import gc
import copy
import logging
from tqdm import tqdm
from abc import abstractmethod
from coevolution.ccea import CCEA
from fitness.penalty import SubsetSizePenalty
from evaluation.wrapper import WrapperEvaluation
from cooperation.best import SingleBestCollaboration
from cooperation.random import SingleRandomCollaboration
from initialization.binary import RandomBinaryInitialization
from optimizers.genetic_algorithm import BinaryGeneticAlgorithm


class CCBCGA(CCEA):
    """Simple Cooperative Co-Evolutionary Algorithm with Best Collaboration and Genetic Algorithm.

    The components of this algorithm will remain unchanged to assess different decomposition
    strategies.
    """

    @abstractmethod
    def _init_decomposer(self) -> None:
        """Instantiate feature grouping method."""
        pass

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
