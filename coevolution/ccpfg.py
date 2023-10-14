import copy
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from projection.vip import VIP
from coevolution.ccea import CCEA
from projection.cipls import CIPLS
from sklearn.cluster import KMeans
from fitness.penalty import SubsetSizePenalty
from evaluation.wrapper import WrapperEvaluation
from cooperation.best import SingleBestCollaboration
from sklearn.cross_decomposition import PLSRegression
from cooperation.random import SingleRandomCollaboration
from decomposition.ranking import RankingFeatureGrouping
from decomposition.clustering import ClusteringFeatureGrouping
from initialization.random import RandomBinaryInitialization
from optimizers.genetic_algorithm import BinaryGeneticAlgorithm


class CCPFG(CCEA):
    """Cooperative Co-Evolutionary Algorithm with Projection-based Feature Grouping (CCPFG).

    Attributes
    ----------
    subcomp_sizes: list
        Number of features in each subcomponent.
    feature_idxs: np.ndarray
        List of feature indexes.
    """

    def _feature_clustering(self, projection_model, folds):
        """Cluster the features according to their contribution to the components of the
        low-dimensional latent space.

        Note that when the evaluation mode is "kfold_cv", a clustering is done for each fold and
        then a clustering is done on the centroids of these clusterings to perform a consensus
        clustering. Otherwise, only one clustering is done on the training set.

        Parameters
        ----------
        projection_model: sklearn model object
            Partial Least Squares regression model. It can be the traditional version (PLS) or the
            Covariance-free version (CIPLS).
        folds: list, or tuple
            Data for model fit, where index 0 of each value indicates the input X and index 1 the
            output y. In case the evaluation mode is "kfold_cv", this list has k values (input and
            output pairs). Otherwise, it has only one input and output pair.

        Returns
        -------
        feature_clusters: np.ndarray
            Index of the cluster each feature belongs to.
        """
        centroids_folds = list()
        feature_loadings_folds = list()
        for Xk_train, yk_train in folds:
            # For each fold, use an unfitted model and train it
            projection_model_k = copy.deepcopy(projection_model)
            projection_model_k.fit(X=Xk_train, Y=yk_train)
            # Get the loadings of features on PLS components
            feature_loadings = np.abs(projection_model_k.x_loadings_)
            feature_loadings_folds.append(feature_loadings)
            # Cluster features based on loadings.
            # Loadings indicate how strongly each feature contributes to each component.
            # Features with similar loadings on the same components are likely to be related.
            clustering_model_k = KMeans(n_clusters=self.n_subcomps, random_state=self.seed)
            feature_clusters = clustering_model_k.fit_predict(feature_loadings)
            centroids_folds.append(clustering_model_k.cluster_centers_)
            del projection_model_k, clustering_model_k
        # Combine centroids from all folds to create a consensus set of centroids
        if len(folds) > 1:
            clustering_model = KMeans(n_clusters=self.n_subcomps, random_state=self.seed)
            clustering_model.fit(np.concatenate(centroids_folds))
            consensus_centroids = clustering_model.cluster_centers_
            feature_loadings_mean = np.mean(feature_loadings_folds, axis=0)
            distances = np.linalg.norm(
                feature_loadings_mean[:, np.newaxis] - consensus_centroids, axis=2
            )
            feature_clusters = np.argmin(distances, axis=1)

        return feature_clusters

    def _compute_variable_importances(self, projection_model, folds):
        """
        Compute variable importance in projection.

        Note that when the evaluation mode is "kfold_cv", the importances will be averages of the
        importances in the folds. Otherwise, the importances will be calculated only once in the
        training set.

        Parameters
        ----------
        projection_model: sklearn model object
            Partial Least Squares regression model. It can be the traditional version (PLS) or the
            Covariance-free version (CIPLS).
        folds: list, or tuple
            Data for model fit, where index 0 of each value indicates the input X and index 1 the
            output y. In case the evaluation mode is "kfold_cv", this list has k values (input and
            output pairs). Otherwise, it has only one input and output pair.

        Returns
        -------
        importances: np.ndarray (n_features,)
            Importance of each feature based on its contribution to yield the latent space.
        """
        vips = list()
        for Xk_train, yk_train in folds:
            # For each fold, use an unfitted model and train it
            projection_model_k = copy.deepcopy(projection_model)
            projection_model_k.fit(X=Xk_train, Y=yk_train)
            vip = VIP(model=projection_model_k)
            vip.compute()
            vips.append(vip.importances.copy())
            del projection_model_k, vip
        # The importance of a variable will be the average value of its importances in the k-folds
        importances = pd.DataFrame(vips).mean(axis=0).values
        return importances

    def _init_decomposer(self):
        """Instantiate feature grouping method."""
        # Number of components to keep after projection
        self.n_components = self.conf["decomposition"]["n_components"]
        # Method used to distribute features into subcomponents
        self.method = self.conf["decomposition"]["method"]
        logging.info(f"Decomposition approach: {self.method}.")

        # Define projection model according to the number of features
        high_dim = self.data.n_features > 1000
        projection_model = (
            CIPLS(n_components=self.n_components, copy=True)
            if high_dim
            else PLSRegression(n_components=self.n_components, copy=True)
        )

        # Cross-validation
        if self.conf["evaluation"]["eval_mode"] == "kfold_cv":
            folds = self.data.train_folds
        # Train-validation
        else:
            folds = [(self.data.X_train, self.data.y_train)]

        # Instantiate feature grouping
        if self.method == "clustering":
            feature_clusters = self._feature_clustering(
                projection_model=projection_model,
                folds=folds
            )
            self.decomposer = ClusteringFeatureGrouping(
                n_subcomps=self.n_subcomps,
                clusters=feature_clusters
            )
        else:
            importances = self._compute_variable_importances(
                projection_model=projection_model,
                folds=folds
            )
            # Ranking feature grouping using variable importances as scores
            self.decomposer = RankingFeatureGrouping(
                n_subcomps=self.n_subcomps,
                subcomp_sizes=self.subcomp_sizes,
                scores=importances,
                method=self.method,
                ascending=False
            )

    def _init_collaborator(self):
        """Instantiate collaboration method."""
        self.best_collaborator = SingleBestCollaboration()
        self.random_collaborator = SingleRandomCollaboration(seed=self.seed)

    def _init_evaluator(self):
        """Instantiate evaluation method."""
        evaluator = WrapperEvaluation(task=self.conf["wrapper"]["task"],
                                      model_type=self.conf["wrapper"]["model_type"],
                                      eval_function=self.conf["evaluation"]["eval_function"],
                                      eval_mode=self.conf["evaluation"]["eval_mode"])
        self.fitness_function = SubsetSizePenalty(evaluator=evaluator,
                                                  weights=self.conf["evaluation"]["weights"])

    def _init_subpop_initializer(self):
        """Instantiate subpopulation initialization method."""
        self.initializer = RandomBinaryInitialization(data=self.data,
                                                      subcomp_sizes=self.subcomp_sizes,
                                                      subpop_sizes=self.subpop_sizes,
                                                      collaborator=self.random_collaborator,
                                                      fitness_function=self.fitness_function)

    def _init_optimizers(self):
        """Instantiate evolutionary algorithms to evolve each subpopulation."""
        self.optimizers = list()
        # Instantiate an optimizer for each subcomponent
        for i in range(self.n_subcomps):
            optimizer = BinaryGeneticAlgorithm(subpop_size=self.subpop_sizes[i],
                                               n_features=self.subcomp_sizes[i],
                                               conf=self.conf)
            self.optimizers.append(optimizer)

    def optimize(self):
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
            for i in range(self.n_subcomps):
                self.subpops[i] = self.optimizers[i].evolve(
                    subpop=self.subpops[i],
                    fitness=self.fitness[i]
                )
            # For each subpopulation
            for i in range(self.n_subcomps):
                # Find best individuals from the previous generation as collaborators for each
                # individual in the current generation
                for j in range(self.subpop_sizes[i]):
                    collaborators = self.best_collaborator.get_collaborators(
                        subpop_idx=i,
                        indiv_idx=j,
                        subpops=self.subpops,
                        current_best=self.current_best
                    )
                    context_vector = self.best_collaborator.build_context_vector(collaborators)
                    # Update the context vector
                    # TODO Should I store the best context vector of each subpopulation across generations?
                    self.context_vectors[i][j] = context_vector.copy()
                    # Update fitness
                    self.fitness[i][j] = self.fitness_function.evaluate(context_vector, self.data)
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
