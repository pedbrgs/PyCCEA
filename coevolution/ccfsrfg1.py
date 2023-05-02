import logging
from tqdm import tqdm
from coevolution.ccea import CCEA
from evaluation.wrapper import WrapperEvaluation
from decomposition.random import RandomFeatureGrouping
from cooperation.best import SingleBestCollaboration
from cooperation.random import SingleRandomCollaboration
from initialization.random import RandomBinaryInitialization
from optimizers.genetic_algorithm import BinaryGeneticAlgorithm


class CCFSRFG1(CCEA):
    """ Cooperative Co-Evolutionary-Based Feature Selection with Random Feature Grouping 1.

    Rashid, A. N. M., et al. "Cooperative co-evolution for feature selection in Big Data with
    random feature grouping." Journal of Big Data 7.1 (2020): 1-42.

    Attributes
    ----------
    subcomp_sizes: list
        Number of features in each subcomponent.
    feature_idxs: np.ndarray
        Shuffled list of feature indexes.
    """

    def _init_decomposer(self):
        """Instantiate feature grouping method."""
        self.decomposer = RandomFeatureGrouping(n_subcomps=self.n_subcomps, seed=self.seed)

    def _init_evaluator(self):
        """Instantiate evaluation method."""
        self.evaluator = WrapperEvaluation(task=self.conf["wrapper"]["task"],
                                           model_type=self.conf["wrapper"]["model_type"],
                                           eval_function=self.conf["wrapper"]["eval_function"])

    def _init_collaborator(self):
        """Instantiate collaboration method."""
        self.best_collaborator = SingleBestCollaboration()
        self.random_collaborator = SingleRandomCollaboration(seed=self.seed)

    def _init_subpop_initializer(self):
        """Instantiate subpopulation initialization method."""
        self.initializer = RandomBinaryInitialization(data=self.data,
                                                      subcomp_sizes=self.subcomp_sizes,
                                                      subpop_sizes=self.subpop_sizes,
                                                      evaluator=self.evaluator,
                                                      collaborator=self.random_collaborator,
                                                      penalty=self.conf["coevolution"]["penalty"],
                                                      weights=self.conf["coevolution"]["weights"])

    def _init_optimizers(self):
        """Instantiate evolutionary algorithms to evolve each subpopulation."""
        self.optimizers = list()
        # Instantiate an optimizer for each subcomponent
        for i in range(self.n_subcomps):
            optimizer = BinaryGeneticAlgorithm(subpop_size=self.subpop_sizes[i],
                                               X_train=self.data.S_train[i],
                                               y_train=self.data.y_train,
                                               X_test=self.data.S_val[i],
                                               y_test=self.data.y_val,
                                               evaluator=self.evaluator,
                                               conf=self.conf)
            self.optimizers.append(optimizer)

    def _problem_decomposition(self):
        """Decompose the problem into smaller subproblems."""
        # Decompose features in the training set
        self.data.S_train, self.subcomp_sizes, self.feature_idxs = self.decomposer.decompose(
            X=self.data.X_train)
        # Decompose features in the validation set
        self.data.S_val, _, _ = self.decomposer.decompose(X=self.data.X_val,
                                                          feature_idxs=self.feature_idxs)
        # Reorder the data according to shuffling in feature decomposition
        self.data.X_train = self.data.X_train[:, self.feature_idxs].copy()
        self.data.X_val = self.data.X_val[:, self.feature_idxs].copy()
        self.data.X_test = self.data.X_test[:, self.feature_idxs].copy()

    def _evaluate(self, context_vector):
        """Evaluate the given context vector using the evaluator."""
        # Evaluate the context vector
        fitness = self.evaluator.evaluate(solution=context_vector,
                                          X_train=self.data.X_train,
                                          y_train=self.data.y_train,
                                          X_test=self.data.X_val,
                                          y_test=self.data.y_val)

        # Penalize large subsets of features
        if self.conf["coevolution"]["penalty"]:
            features_p = context_vector.sum()/context_vector.shape[0]
            global_fitness = self.conf["coevolution"]["weights"][0] * fitness -\
                self.conf["coevolution"]["weights"][1] * features_p

        return global_fitness

    def optimize(self):
        """Solve the feature selection problem through optimization."""
        # Decompose problem
        self._problem_decomposition()
        # Initialize subpopulations
        self._init_subpopulations()
        # Instantiate optimizers
        self._init_optimizers()

        # Get the best individual and context vector from each subpopulation
        self.current_best = self.best_collaborator.get_best_individuals(
            subpops=self.subpops,
            local_fitness=self.local_fitness,
            global_fitness=self.global_fitness,
            context_vectors=self.context_vectors
        )
        # Select the globally best context vector
        self.best_context_vector, self.best_global_fitness = self._get_global_best()

        # Set the number of generations counter with the first generation
        n_gen = 1
        # Number of generations that the best global fitness has not improved
        stagnation_counter = 0
        # Initialize the optimization progress bar
        progress_bar = tqdm(total=self.conf["coevolution"]["max_gen"],
                            desc="Generations",
                            leave=False)

        # Iterate up to the maximum number of generations
        while n_gen <= self.conf["coevolution"]["max_gen"]:
            # Append current best global fitness
            self.convergence_curve.append(self.best_global_fitness)
            # Evolve each subpopulation using a genetic algorithm
            for i in range(self.n_subcomps):
                self.subpops[i], self.local_fitness[i] = self.optimizers[i].evolve(
                    self.subpops[i],
                    self.local_fitness[i]
                )
                # Best individuals from the previous generation as collaborators for each
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
                    self.context_vectors[i][j] = context_vector
                    # Update the global evaluation
                    self.global_fitness[i][j] = self._evaluate(context_vector)
            # Get the best individual and context vector from each subpopulation
            self.current_best = self.best_collaborator.get_best_individuals(
                subpops=self.subpops,
                local_fitness=self.local_fitness,
                global_fitness=self.global_fitness,
                context_vectors=self.context_vectors
            )
            # Select the globally best context vector
            best_context_vector, best_global_fitness = self._get_global_best()
            # Update best context vector
            if self.best_global_fitness < best_global_fitness:
                # Reset stagnation counter because best global fitness has improved
                stagnation_counter = 0
                # Enable logger if specified
                logging.getLogger().disabled = False if self.verbose else True
                # Show improvement
                old_best_fitness = round(self.best_global_fitness, 4)
                new_best_fitness = round(best_global_fitness, 4)
                logging.info(f"\nUpdate fitness from {old_best_fitness} to {new_best_fitness}.")
                # Update best context vector
                self.best_context_vector = best_context_vector
                # Update best global fitness
                self.best_global_fitness = best_global_fitness
            else:
                # Increase stagnation counter because best global fitness has not improved
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
