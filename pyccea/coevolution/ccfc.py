import gc
import copy
import logging
from tqdm import tqdm
from ..coevolution.ccga import CCGA
from sklearn.cluster import KMeans
from ..decomposition.clustering import ClusteringFeatureGrouping


class CCFC(CCGA):
    """Cooperative Co-Evolutionary Algorithm with Feature Clustering.

    Decomposition strategy proposed on:
    Li, Haoran, et al. "MLFS-CCDE: multi-objective large-scale feature selection by cooperative
    coevolutionary differential evolution." Memetic Computing 13 (2021): 1-18.
    """

    def _init_decomposer(self) -> None:
        """Instantiate feature clustering method."""
        # Transpose the training data to switch rows (instances) and columns (features)
        transposed_X_train = self.data.X_train.transpose().copy()
        # Initialize the k-means clustering model to group features into subcomponents
        clustering_model = KMeans(n_clusters=self.n_subcomps)
        # Fit the clustering model on the transposed data (now features are treated as instances)
        clustering_model.fit(X=transposed_X_train, y=self.data.y_train)
        # Extract the cluster labels, which indicate the group each feature belongs to
        feature_clusters = clustering_model.labels_
        # Assign the feature clusters to the decomposer object for further processing
        self.decomposer = ClusteringFeatureGrouping(
            n_subcomps=self.n_subcomps,
            clusters=feature_clusters
        )

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
                # Objective weight
                w1 = self.conf["evaluation"]["weights"][0]
                # Penalty weight
                w2 = self.conf["evaluation"]["weights"][1]
                # Current fitness, performance evaluation and penalty
                current_best_fitness = round(self.best_fitness, 4)
                current_penalty = round(self.best_context_vector.sum()/self.data.n_features, 4)
                current_eval = round((self.best_fitness + w2*current_penalty)/w1, 4)
                # New fitness, performance evaluation and penalty
                new_best_fitness = round(best_fitness, 4)
                new_penalty = round(best_context_vector.sum()/self.data.n_features, 4)
                new_eval = round((best_fitness + w2*new_penalty)/w1, 4)
                # Show improvement
                logging.info(
                    f"\nUpdate fitness from {current_best_fitness} to {new_best_fitness}.\n"
                    f"Update predictive performance from {current_eval} to {new_eval}.\n"
                    f"Update penalty from {current_penalty} to {new_penalty}.\n"
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
