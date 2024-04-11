import gc
import copy
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from kneed import KneeLocator
from projection.vip import VIP
from coevolution.ccea import CCEA
from projection.cipls import CIPLS
from sklearn.metrics import silhouette_score
from fitness.penalty import SubsetSizePenalty
from evaluation.wrapper import WrapperEvaluation
from cooperation.best import SingleBestCollaboration
from sklearn.cross_decomposition import PLSRegression
from decomposition.ranking import RankingFeatureGrouping
from cooperation.random import SingleRandomCollaboration
from sklearn.cluster import KMeans, AgglomerativeClustering
from initialization.binary import RandomBinaryInitialization
from decomposition.clustering import ClusteringFeatureGrouping
from optimizers.genetic_algorithm import BinaryGeneticAlgorithm


class CCPFG(CCEA):
    """Cooperative Co-Evolutionary Algorithm with Projection-based Feature Grouping (CCPFG).

    Attributes
    ----------
    n_components : int
        Number of components to keep after dimensionality reduction.
    method : str
        Projection-based decomposition method. It can be 'clustering', 'elitist' and 'distributed'.
    vip_threshold : float
        All features whose importance is less than or equal to this threshold will be removed.
    removed_features : np.ndarray
        Indexes of features that were removed due to their low importance.
    """

    CLUSTERING_METHODS = {
        "k_means": (KMeans, {}),
        "agglomerative_clustering": (AgglomerativeClustering, {"linkage": "ward", "affinity": "euclidean"})
    }

    def _feature_clustering(self, projection_model) -> np.ndarray:
        """Cluster the features according to their contribution to the components of the
        low-dimensional latent space.

        Parameters
        ----------
        projection_model : sklearn model object
            Partial Least Squares regression model. It can be the traditional version (PLS) or the
            Covariance-free version (CIPLS).

        Returns
        -------
        feature_clusters : np.ndarray
            Index of the cluster each feature belongs to.
        """
        projection_model = copy.deepcopy(projection_model)
        X_train_normalized = self.data.X_train - self.data.X_train.mean(axis=0)
        y_train_encoded = pd.get_dummies(self.data.y_train).astype(int)
        projection_model.fit(X=X_train_normalized, Y=y_train_encoded)
        # Get the loadings of features on PLS components
        feature_loadings = abs(projection_model.x_loadings_)

        # Automatically define the number of subcomponents
        self.n_subcomps = self._get_best_number_of_subcomponents(feature_loadings)
        # Update the subpopulation sizes after update the number of subcomponents
        self.subpop_sizes = [self.subpop_sizes[0]] * self.n_subcomps

        # Cluster features based on loadings.
        # Loadings indicate how strongly each feature contributes to each component.
        # Features with similar loadings on the same components are likely to be related.
        clustering_model_class, clustering_params = CCPFG.CLUSTERING_METHODS[self.clustering_model_type]
        clustering_model = clustering_model_class(n_clusters=self.n_subcomps, **clustering_params)
        feature_clusters = clustering_model.fit_predict(feature_loadings)

        return feature_clusters

    def _get_best_number_of_components(
            self,
            projection_class,
            X_train: np.ndarray,
            y_train: np.ndarray
        ):
        """Get the best number of components to keep by PLS decomposition.

        The number of components will occur at the elbow point where the coefficient of
        determination (r2-score) starts to level off.

        Parameters
        ----------
        projection_class : sklearn model class
            Partial Least Squares regression class. It can be the traditional version (PLS) or the
            Covariance-free version (CIPLS).
        X_train : np.ndarray
            Train input data.
        y_train : np.ndarray
            Train output data.

        Returns
        -------
        n_components : int
            Best number of components to keep after the decomposition into a latent space.
        """
        n_components_range = range(2, min(30, X_train.shape[1]))
        r_squared_values = list()

        for n_components in n_components_range:
            projection_model = projection_class(n_components=n_components, copy=True)
            X_train_normalized = X_train - X_train.mean(axis=0)
            y_train_encoded = pd.get_dummies(y_train).astype(int)
            projection_model.fit(X_train_normalized, y_train_encoded)
            # Sum the coefficient of determination of the prediction
            r_squared_values.append(np.sum(projection_model.score(X_train_normalized, y_train_encoded)))
            del projection_model
            gc.collect()

        # Use kneed to find the knee/elbow point
        kneedle = KneeLocator(
            n_components_range, r_squared_values, curve="concave", direction="increasing"
        )
        n_components = kneedle.knee
        logging.info(f"Best number of components: {n_components}.")

        return n_components

    def _get_best_number_of_subcomponents(self, feature_loadings: np.ndarray) -> int:
        """Get the best number of subcomponents.

        The number of subcomponents (clusters) will be the one that maximizes the silhouette
        coefficient of the clustering of X loadings.

        Parameters
        ----------
        feature_loadings : np.ndarray
            The absolute importance of terms to components.

        Returns
        -------
        n_subcomps : int
            Best number of subcomponents to decompose the original problem.
        """
        n_clusters_range = range(2, min(10, feature_loadings.shape[0]))
        silhouette_scores = list()

        for n_clusters in n_clusters_range:
            clustering_model_class, clustering_params = CCPFG.CLUSTERING_METHODS[self.clustering_model_type]
            clustering_model = clustering_model_class(n_clusters=n_clusters, **clustering_params)
            cluster_labels = clustering_model.fit_predict(feature_loadings)
            silhouette_avg = silhouette_score(feature_loadings, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        silhouette_scores = pd.DataFrame(
            list(zip(n_clusters_range, silhouette_scores)),
            columns=["n_clusters", "silhouette_score"]
        )

        n_subcomps = silhouette_scores.loc[
            silhouette_scores["silhouette_score"].idxmax(), "n_clusters"
        ]
        logging.info(f"Best number of subcomponents: {n_subcomps}.")

        return n_subcomps

    def _get_best_quantile_to_remove(self, importances: np.ndarray) -> float:
        """
        Gets the best quantile of the feature importance distribution to remove weak features.

        The best quantile will be the one that, when its features are removed from a context
        vector filled with 1's (selecting all remaining features), gives the best fitness value.

        Parameters
        ----------
        importances : np.ndarray (n_features,)
            Importance of each feature based on its contribution to yield the latent space.

        Returns
        -------
        best_quantile : float
            Best quantile. All features that have their importance in this quantile of the
            feature importance distribution will be removed.
        """
        metrics = dict()
        quantiles = np.arange(0, 0.55, 0.05).round(2)
        for quantile in quantiles:
            data_q = copy.deepcopy(self.data)
            vip_threshold = round(np.quantile(importances, quantile), 4)
            features_to_keep = importances > vip_threshold
            # Removing features from subsets and folds
            data_q.X_train = data_q.X_train[:, features_to_keep].copy()
            data_q.X_test = data_q.X_test[:, features_to_keep].copy()
            for k in range(data_q.kfolds):
                data_q.train_folds[k][0] = data_q.train_folds[k][0][:, features_to_keep].copy()
                data_q.val_folds[k][0] = data_q.val_folds[k][0][:, features_to_keep].copy()
            # Build context vector with all remaining features
            context_vector = np.ones((data_q.X_train.shape[1],)).astype(bool)
            metrics[quantile] = self.fitness_function.evaluate(context_vector, data_q)
            logging.getLogger().disabled = False
        # Get the quantile that gives the best fitness value
        metrics = pd.DataFrame(list(metrics.items()), columns=["quantile", "fitness"])
        best_quantile = metrics.loc[metrics["fitness"].idxmax(), "quantile"]
        logging.info(f"Best quantile: {best_quantile}.")
        return best_quantile

    def _remove_unimportant_features(
            self,
            importances: np.ndarray,
        ) -> np.ndarray:
        """Remove irrelevant or weaken features from folds and subsets.

        Parameters
        ----------
        importances : np.ndarray
            Importance of each feature based on its contribution to yield the latent space.

        Returns
        -------
        importances : np.ndarray
            Importance of the remaining features.
        """
        self.quantile_to_remove = self._get_best_quantile_to_remove(importances)
        self.vip_threshold = round(np.quantile(importances, self.quantile_to_remove), 4)
        logging.info(f"Removing features with VIP less than or equal to {self.vip_threshold}...")
        features_to_keep = importances > self.vip_threshold
        self.removed_features = np.where(features_to_keep == False)[0]
        logging.info(f"{len(self.removed_features)} features were removed.")

        # Removing features from subsets and folds
        self.data.X_train = self.data.X_train[:, features_to_keep].copy()
        self.data.X_test = self.data.X_test[:, features_to_keep].copy()
        for k in range(self.data.kfolds):
            self.data.train_folds[k][0] = self.data.train_folds[k][0][:, features_to_keep].copy()
            self.data.val_folds[k][0] = self.data.val_folds[k][0][:, features_to_keep].copy()

        # Importance of the remaining features
        importances = importances[features_to_keep].copy()

        return importances

    def _compute_variable_importances(self, projection_model) -> np.ndarray:
        """Compute variable importance in projection (VIP).

        Parameters
        ----------
        projection_model : sklearn model object
            Partial Least Squares regression model. It can be the traditional version (PLS) or the
            Covariance-free version (CIPLS).

        Returns
        -------
        importances : np.ndarray (n_features,)
            Importance of each feature based on its contribution to yield the latent space.
        """
        projection_model = copy.deepcopy(projection_model)
        X_train_normalized = self.data.X_train - self.data.X_train.mean(axis=0)
        y_train_encoded = pd.get_dummies(self.data.y_train).astype(int)
        projection_model.fit(X=X_train_normalized, Y=y_train_encoded)
        vip = VIP(model=projection_model)
        vip.compute()
        importances = vip.importances.copy()
        return importances

    def _init_decomposer(self) -> None:
        """Instantiate feature grouping method."""
        # Method used to distribute features into subcomponents
        self.method = self.conf["decomposition"]["method"]
        logging.info(f"Decomposition approach: {self.method}.")

        # Define projection model according to the number of features
        high_dim = self.data.n_features > 100000
        # TODO CIPLS breaks when running for too many dimensions
        projection_class = CIPLS if high_dim else PLSRegression

        if self.conf["decomposition"].get("n_components"):
            self.n_components = self.conf["decomposition"]["n_components"]
        else:
            logging.info("Automatically choosing the number of PLS components...")
            # Get the best number of components to keep after projection using the Elbow method
            self.n_components = self._get_best_number_of_components(
                projection_class, self.data.X_train, self.data.y_train
            )
        # Instantiate projection model object
        projection_model = projection_class(n_components=self.n_components, copy=True)

        # Compute feature importances
        importances = self._compute_variable_importances(projection_model=projection_model)

        # Remove irrelevant or weaken relevant features
        if self.conf["decomposition"].get("drop", False) > 0:
            importances = self._remove_unimportant_features(importances)

        # Instantiate feature grouping
        if self.method == "clustering":
            self.clustering_model_type = self.conf["decomposition"]["clustering_model_type"]
            logging.info(f"Clustering model type: {self.clustering_model_type}")
            feature_clusters = self._feature_clustering(projection_model=projection_model)
            self.decomposer = ClusteringFeatureGrouping(
                n_subcomps=self.n_subcomps,
                clusters=feature_clusters
            )
        else:
            # Ranking feature grouping using variable importances as scores
            self.decomposer = RankingFeatureGrouping(
                n_subcomps=self.n_subcomps,
                subcomp_sizes=self.subcomp_sizes,
                scores=importances,
                method=self.method,
                ascending=False
            )

    def _init_collaborator(self) -> None:
        """Instantiate collaboration method."""
        self.best_collaborator = SingleBestCollaboration()
        self.random_collaborator = SingleRandomCollaboration(seed=self.seed)

    def _init_evaluator(self) -> None:
        """Instantiate evaluation method."""
        evaluator = WrapperEvaluation(task=self.conf["wrapper"]["task"],
                                      model_type=self.conf["wrapper"]["model_type"],
                                      eval_function=self.conf["evaluation"]["eval_function"],
                                      eval_mode=self.eval_mode,
                                      n_classes=self.data.n_classes)
        self.fitness_function = SubsetSizePenalty(evaluator=evaluator,
                                                  weights=self.conf["evaluation"]["weights"])

    def _init_subpop_initializer(self) -> None:
        """Instantiate subpopulation initialization method."""
        self.initializer = RandomBinaryInitialization(data=self.data,
                                                      subcomp_sizes=self.subcomp_sizes,
                                                      subpop_sizes=self.subpop_sizes,
                                                      collaborator=self.random_collaborator,
                                                      fitness_function=self.fitness_function)

    def _init_optimizers(self) -> None:
        """Instantiate evolutionary algorithms to evolve each subpopulation."""
        self.optimizers = list()
        # Instantiate an optimizer for each subcomponent
        for i in range(self.n_subcomps):
            optimizer = BinaryGeneticAlgorithm(subpop_size=self.subpop_sizes[i],
                                               n_features=self.subcomp_sizes[i],
                                               conf=self.conf)
            self.optimizers.append(optimizer)

    def optimize(self) -> None:
        """Solve the feature selection problem through optimization."""
        # Decompose problem
        self._problem_decomposition()
        # Initialize subpopulations
        self._init_subpopulations()
        # Instantiate optimizers
        self._init_optimizers()

        # Get the best individual and context vector from each subpopulation
        self.current_best = self._get_best_individuals(
            subpops=self.subpops,
            fitness=self.fitness,
            context_vectors=self.context_vectors
        )
        # Select the globally best context vector
        self.best_context_vector, self.best_fitness = self._get_global_best()
        self.best_context_vectors.append(self.best_context_vector.copy())
        # Save the order of features considered in the random feature grouping
        self.best_feature_idxs = self.feature_idxs.copy()

        # Set the number of generations counter
        n_gen = 0
        # Number of generations that the best fitness has not improved
        stagnation_counter = 0
        # Initialize the optimization progress bar
        progress_bar = tqdm(total=self.conf["coevolution"]["max_gen"],
                            desc="Generations",
                            leave=False)

        # Iterate up to the maximum number of generations
        while n_gen <= self.conf["coevolution"]["max_gen"]:
            # Append current best fitness
            self.convergence_curve.append(self.best_fitness)

            # Evolve each subpopulation using a genetic algorithm
            current_subpops = list()
            for i in range(self.n_subcomps):
                current_subpop = self.optimizers[i].evolve(
                    subpop=self.subpops[i],
                    fitness=self.fitness[i]
                )
                current_subpops.append(current_subpop)

            # Evaluate each individual of the evolved subpopulations
            current_fitness = list()
            current_context_vectors = list()
            for i in range(self.n_subcomps):
                current_fitness.append(list())
                current_context_vectors.append(list())
                # Use best individuals from the previous generation (`self.current_best`) as
                # collaborators for each individual in the current generation after evolve
                # (`current_subpops`)
                for j in range(self.subpop_sizes[i]):
                    collaborators = self.best_collaborator.get_collaborators(
                        subpop_idx=i,
                        indiv_idx=j,
                        current_subpops=current_subpops,
                        current_best=self.current_best
                    )
                    context_vector = self.best_collaborator.build_context_vector(collaborators)
                    # Update the context vector
                    current_context_vectors[i].append(context_vector.copy())
                    # Update fitness
                    current_fitness[i].append(self.fitness_function.evaluate(context_vector, self.data))
            # Update subpopulations, context vectors and evaluations
            self.subpops = copy.deepcopy(current_subpops)
            self.fitness = copy.deepcopy(current_fitness)
            self.context_vectors = copy.deepcopy(current_context_vectors)
            del current_subpops, current_fitness, current_context_vectors
            gc.collect()

            # Get the best individual and context vector from each subpopulation
            self.current_best = self._get_best_individuals(
                subpops=self.subpops,
                fitness=self.fitness,
                context_vectors=self.context_vectors
            )
            # Select the globally best context vector
            best_context_vector, best_fitness = self._get_global_best()
            # Update best context vector
            if self.best_fitness < best_fitness:
                # Reset stagnation counter because best fitness has improved
                stagnation_counter = 0
                # Enable logger if specified
                logging.getLogger().disabled = False if self.verbose else True
                # Current fitness
                current_best_fitness = round(self.best_fitness, 4)
                # New fitness
                new_best_fitness = round(best_fitness, 4)
                # Show improvement
                logging.info(
                    f"\nUpdate fitness from {current_best_fitness} to {new_best_fitness}.\n"
                )
                # Update best context vector
                self.best_context_vector = best_context_vector.copy()
                self.best_context_vectors.append(self.best_context_vector.copy())
                # Update best fitness
                self.best_fitness = best_fitness
            else:
                # Increase stagnation counter because best fitness has not improved
                stagnation_counter += 1
                # Checks whether the optimization has been stagnant for a long time
                if stagnation_counter >= self.conf["coevolution"]["max_gen_without_improvement"]:
                    # Enable logger
                    logging.getLogger().disabled = False
                    logging.info(
                        "\nEarly stopping because fitness has been stagnant for "
                        f"{stagnation_counter} generations in a row."
                    )
                    break
            # Increase number of generations
            n_gen += 1
            # Update progress bar
            progress_bar.update(1)
        # Close progress bar after optimization
        progress_bar.close()
