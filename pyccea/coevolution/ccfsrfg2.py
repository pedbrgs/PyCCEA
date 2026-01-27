import gc
import logging
import numpy as np
from tqdm import tqdm
from ..coevolution.ccga import CCGA
from ..decomposition.random import RandomFeatureGrouping


class CCFSRFG2(CCGA):
    """Cooperative Co-Evolutionary-Based Feature Selection with Random Feature Grouping 2.

    Rashid, A. N. M., et al. "Cooperative co-evolution for feature selection in Big Data with
    random feature grouping." Journal of Big Data 7.1 (2020): 1-42.

    Attributes
    ----------
    best_feature_idxs : np.ndarray
        List of feature indices corresponding to the best decomposition.
    """

    def _init_decomposer(self):
        """Instantiate feature grouping method."""
        self.decomposer = RandomFeatureGrouping(
            n_subcomps=self.n_subcomps,
            subcomp_sizes=self.subcomp_sizes
        )

    def _get_best_individuals(self, subpops, fitness, context_vectors) -> dict:
        """Get the best individual of each subpopulation and its respective evaluation.

        In CCFSRFG2, each subpopulation can keep multiple context vectors (one per individual),
        so we need to select the context vector aligned with the best fitness
        """
        current_best = dict()
        n_subpops = len(subpops)
        for i in range(n_subpops):
            best_ind_idx = np.argmax(fitness[i])
            current_best[i] = dict()
            current_best[i]["individual"] = subpops[i][best_ind_idx].copy()
            context_vector = context_vectors[i].copy()
            if isinstance(context_vector, list):
                context_vector = context_vector[best_ind_idx].copy()
            current_best[i]["context_vector"] = context_vector
            current_best[i]["fitness"] = fitness[i][best_ind_idx]
        return current_best

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
        # Store the shuffled feature list that generated the best context vector
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

            # Decompose problem
            current_feature_idxs = self.feature_idxs.copy()
            self._problem_decomposition()
            self.feature_idxs = current_feature_idxs[self.feature_idxs].copy()

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
                # Select the 'elite_size' best individuals of the previous generation to be in the
                # current generation (elitism)
                prev_context_vectors = self.context_vectors[i]
                if isinstance(prev_context_vectors, np.ndarray) and prev_context_vectors.ndim == 1:
                    prev_context_vectors = [prev_context_vectors]
                else:
                    prev_context_vectors = list(prev_context_vectors)
                elite_size = min(self.optimizers[i].elite_size, len(prev_context_vectors))
                if elite_size == 0:
                    elite_fitness = list()
                    elite_context_vectors = list()
                elif elite_size == 1 and len(prev_context_vectors) == 1:
                    elite_fitness = [max(self.fitness[i])]
                    elite_context_vectors = [prev_context_vectors[0].copy()]
                else:
                    descending_order = np.argsort(self.fitness[i])[::-1][:elite_size]
                    elite_fitness = [self.fitness[i][idx] for idx in descending_order]
                    elite_context_vectors = [prev_context_vectors[idx].copy() for idx in descending_order]
                current_fitness.append(list(elite_fitness))
                current_context_vectors.append(list(elite_context_vectors))
                # Use random individuals from the previous generation as collaborators for each
                # individual in the current generation. Except the first 'elite_size' individuals
                # from each subpopulation which are being used as elitism and have different
                # features from the individuals of the previous generation
                for j in range(elite_size, self.subpop_sizes[i]):
                    collaborators = self.random_collaborator.get_collaborators(
                        subpop_idx=i,
                        indiv_idx=j,
                        previous_subpops=self.subpops,
                        current_subpops=current_subpops,
                    )
                    context_vector = self.random_collaborator.build_context_vector(collaborators)
                    # Update the context vector
                    current_context_vectors[i].append(context_vector.copy())
                    # Update fitness
                    current_fitness[i].append(self.fitness_function.evaluate(context_vector, self.data))
                    del collaborators, context_vector
            # Update subpopulations, context vectors and evaluations
            self.subpops = current_subpops
            self.fitness = current_fitness
            self.context_vectors = current_context_vectors
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

            # In this particular case, where the problem is decomposed in each generation, the
            # update of the best context vector and feature indices can only be done if the
            # current fitness is greater than the best fitness and not greater than or equal to
            # it. If we include equals in the conditional, generations in which there was no
            # improvement will maintain the same context vector and update the feature indices
            # incorrectly, since the elitist context vector was not necessarily generated in the
            # current generation.
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
                current_best_fitness = round(float(self.best_fitness), 4)
                current_penalty = round(float(sum(self.best_context_vector))/self.data.n_features, 4)
                current_eval = round((float(self.best_fitness) + w2*current_penalty)/w1, 4)
                # New fitness, performance evaluation and penalty
                new_best_fitness = round(float(best_fitness), 4)
                new_penalty = round(float(sum(best_context_vector))/self.data.n_features, 4)
                new_eval = round((float(best_fitness) + w2*new_penalty)/w1, 4)
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
                # Update the shuffled feature list that generated the best context vector
                self.best_feature_idxs = self.feature_idxs.copy()
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
