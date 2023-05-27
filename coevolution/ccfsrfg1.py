import logging
from tqdm import tqdm
from coevolution.ccfsrfg import CCFSRFG
from decomposition.random import RandomFeatureGrouping


class CCFSRFG1(CCFSRFG):
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
