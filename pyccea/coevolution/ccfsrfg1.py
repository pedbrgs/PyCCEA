import logging
from tqdm import tqdm
from ..coevolution.ccga import CCGA
from ..utils.memory import force_memory_release
from ..decomposition.random import RandomFeatureGrouping


class CCFSRFG1(CCGA):
    """Cooperative Co-Evolutionary-Based Feature Selection with Random Feature Grouping 1.

    Rashid, A. N. M., et al. "Cooperative co-evolution for feature selection in Big Data with
    random feature grouping." Journal of Big Data 7.1 (2020): 1-42.
    """

    def _init_decomposer(self):
        """Instantiate feature grouping method."""
        self.decomposer = RandomFeatureGrouping(
            n_subcomps=self.n_subcomps,
            subcomp_sizes=self.subcomp_sizes,
            seed=self.seed
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
        self._record_best_context_vector(self.best_context_vector)
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
            current_fitness, current_best_context_vectors = self._evaluate_evolved_subpopulations(
                collaborators_getter=lambda subpop_idx, indiv_idx: self.best_collaborator.get_collaborators(
                    subpop_idx=subpop_idx,
                    indiv_idx=indiv_idx,
                    current_subpops=current_subpops,
                    current_best=self.current_best
                )
                build_context_vector=self.best_collaborator.build_context_vector
            ),
            # Update subpopulations, context vectors and evaluations
            self.subpops = current_subpops
            self.fitness = current_fitness
            self.context_vectors = current_best_context_vectors
            del current_subpops, current_fitness, current_best_context_vectors
            force_memory_release()

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
                self._record_best_context_vector(self.best_context_vector)
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
