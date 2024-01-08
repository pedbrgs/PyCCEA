import gc
import copy
import logging
import numpy as np
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
    subcomp_sizes : list
        Number of features in each subcomponent.
    feature_idxs : np.ndarray
        List of feature indexes.
    n_components : int
        Number of components to keep after dimensionality reduction.
    method : str
        Projection-based decomposition method. It can be 'clustering', 'elitist' and 'distributed'.
    """

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
        projection_model.fit(X=self.data.X_train, Y=self.data.y_train)
        # Get the loadings of features on PLS components
        feature_loadings = abs(projection_model.x_loadings_)
        # Cluster features based on loadings.
        # Loadings indicate how strongly each feature contributes to each component.
        # Features with similar loadings on the same components are likely to be related.
        clustering_model = KMeans(n_clusters=self.n_subcomps, random_state=self.seed)
        feature_clusters = clustering_model.fit_predict(feature_loadings)

        return feature_clusters

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
        projection_model.fit(X=self.data.X_train, Y=self.data.y_train)
        vip = VIP(model=projection_model)
        vip.compute()
        importances = vip.importances.copy()
        return importances

    def _init_decomposer(self) -> None:
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

        # Compute feature importances
        importances = self._compute_variable_importances(projection_model=projection_model)

        # Instantiate feature grouping
        if self.method == "clustering":
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
            next_subpops = list()
            for i in range(self.n_subcomps):
                next_subpop = self.optimizers[i].evolve(
                    subpop=self.subpops[i],
                    fitness=self.fitness[i]
                )
                next_subpops.append(next_subpop)

            # Evaluate each individual of the evolved subpopulations
            next_fitness = list()
            next_context_vectors = list()
            for i in range(self.n_subcomps):
                next_fitness.append(list())
                next_context_vectors.append(list())
                # Use best individuals from the previous generation (`self.current_best`) as
                # collaborators for each individual in the current generation after evolve
                # (`next_subpop`)
                for j in range(self.subpop_sizes[i]):
                    collaborators = self.best_collaborator.get_collaborators(
                        subpop_idx=i,
                        indiv_idx=j,
                        next_subpops=next_subpops,
                        current_best=self.current_best
                    )
                    context_vector = self.best_collaborator.build_context_vector(collaborators)
                    # Update the context vector
                    next_context_vectors[i].append(context_vector.copy())
                    # Update fitness
                    next_fitness[i].append(self.fitness_function.evaluate(context_vector, self.data))
            # Update subpopulations, context vectors and evaluations
            self.subpops = copy.deepcopy(next_subpops)
            self.fitness = copy.deepcopy(next_fitness)
            self.context_vectors = copy.deepcopy(next_context_vectors)
            del next_subpops, next_fitness, next_context_vectors
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
