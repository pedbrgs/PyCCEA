import numpy as np


class BinaryGeneticAlgorithm():
    """Binary genetic algorithm.

    Attributes
    ----------
    mutation_rate: float
        Probability of a mutation occurring.
    crossover_rate: float
        Probability of a crossover occurring.
    tournament_sample_size: int
        Sample size of the subpopulation that will be used in Tournament Selection.
    selection_method: str
        Population update method employed in each generation, which can take one of two options:
        (i) generational, where the entire population is replaced in each generation (excluding a
        specified 'elite_size' individuals), or (ii) steady-state, where only the two least fit
        individuals are replaced.
    elite_size: int, optional
        Number of individuals that will be preserved from the current to the next generation. They
        are selected according to the fitness, i.e., if it is N, the N best individuals of the
        current generation will be preserved for the next generation. This parameter is only used
        if the selection_method is generational.
    """

    selection_methods = ["generational", "steady-state"]

    def __init__(self,
                 subpop_size: int,
                 n_features: int,
                 conf: dict):
        """
        Parameters
        ----------
        subpop_size: int
            Number of individuals in the subpopulation.
        n_features: int
            Number of features, that is, decision variables.
        conf: dict
            Configuration parameters of the cooperative coevolutionary algorithm.
        """
        # Subpopulation size
        self.subpop_size = subpop_size
        # Configuration parameters
        self.conf = conf
        # Mutation rate
        self.mutation_rate = self.conf["optimizer"]["mutation_rate"]
        # Crossover rate
        self.crossover_rate = self.conf["optimizer"]["crossover_rate"]
        # Population selection method
        self.selection_method = self.conf["optimizer"]["selection_method"]
        # Check if the chosen population selection method is available
        if not self.selection_method in BinaryGeneticAlgorithm.selection_methods:
            raise AssertionError(
                f"Population selection method '{self.selection_method}' was not found. "
                f"The available methods are {', '.join(BinaryGeneticAlgorithm.selection_methods)}."
            )
        # Elite size
        self.elite_size = self.conf["optimizer"].get("elite_size")
        # Number of features
        self.n_features = n_features
        # Sample size for Tournament Selection
        self.tournament_sample_size = self.conf["optimizer"]["tournament_sample_size"]

    def _single_point_crossover(self, parent_a: np.ndarray, parent_b: np.ndarray):
        """Single point crossover. """
        # Random probability
        prob = float(np.random.uniform(low=0, high=1))

        # If the random probability is less than the crossover probability, then do the crossover
        if (prob < self.crossover_rate) and (self.n_features > 1):
            # Crossover point
            point = np.random.randint(low=1, high=self.n_features)
            # Offspring
            offspring = np.concatenate([parent_b[:point], parent_a[point:]], axis=0)
        else:
            # Offspring will be a copy of a parent
            parent_idx = np.random.choice(range(2))
            offspring = [parent_a, parent_b][parent_idx].copy()

        return offspring

    def _mutation(self, parent: np.ndarray):
        """Bit-flip mutation. """
        # Offspring
        offspring = parent.copy()
        # Random probabilities
        probs = np.random.uniform(low=0, high=1, size=self.n_features)
        # If the random probability is less than the mutation probability, then do the mutation
        pos = probs < self.mutation_rate
        # Flip values only of the selected positions
        offspring[pos] ^= 1

        return offspring

    def _tournament_selection(self, subpop: np.ndarray, fitness: list):
        """Tournament selection. """
        # Indexes of selected parents
        selected_idxs = np.zeros((2,))

        for i in range(2):

            # Candidate individuals (sample of size tournament_sample_size)
            sample_idxs = np.random.choice(range(self.subpop_size),
                                           size=self.tournament_sample_size,
                                           replace=False)
            # Get fitness of candidate individuals
            fitness_candidates = np.array(fitness)[sample_idxs]
            # Get index of best individual in the sample, i.e., individual with highest fitness
            best_idx = np.argmax(fitness_candidates)
            # Get fitness of the best individual in the sample
            best_fitness = fitness_candidates[best_idx]
            # Select index of the best individual with respect to the entire subpopulation, i.e.,
            # fitness instead of fitness_candidates
            selected_idxs[i] = np.where(fitness == best_fitness)[0][0]

        # Selected parents
        parent_a, parent_b = subpop[selected_idxs.astype(int)]

        return parent_a, parent_b

    def _survivor_selection(self, subpop: np.ndarray, fitness: list):
        """Selection of individuals who will remain in the subpopulation. """
        # Select the two worst individuals (lower fitness) in the subpopulation
        worst_idxs = np.argsort(fitness)[:2]
        # Eliminate the two worst individuals in the subpopulation
        subpop = np.delete(subpop, worst_idxs, axis=0)

        return subpop

    def evolve(self, subpop: np.ndarray, fitness: list):
        """Evolve a subpopulation for a single generation.

        Parameters
        ----------
        subpop: np.ndarray
            Individuals of the subpopulation, where each individual is an array of size equal to
            the number of features.
        fitness: list
            Evaluation of all individuals in the subpopulation.

        Returns
        -------
        next_subpop: np.ndarray
            Individuals in the subpopulation of the next generation.
        """

        if self.selection_method == "generational":

            # Maximization problem
            descending_order = np.argsort(fitness)[::-1]
            # Select the 'elite_size' best individuals of the current generation to be in the next
            # generation (elitism)
            n_bests = descending_order[:self.elite_size]
            next_subpop = subpop[n_bests].copy()

            # Perform (subpop_size - elite_size) tournament selections to build the next population
            for i in range(self.elite_size, self.subpop_size):
                # Select parents through Tournament Selection
                parent_a, parent_b = self._tournament_selection(subpop, fitness)
                # Recombine pairs of parents
                offspring = self._single_point_crossover(parent_a, parent_b)
                # Mutate the resulting offspring
                offspring = self._mutation(offspring)
                # Add new individual to the subpopulation
                next_subpop = np.vstack([next_subpop, offspring])

        elif self.selection_method == "steady-state":

            # Select parents through Tournament Selection
            parent_a, parent_b = self._tournament_selection(subpop, fitness)
            # Recombine pairs of parents
            offspring_a = self._single_point_crossover(parent_a, parent_b)
            offspring_b = self._single_point_crossover(parent_a, parent_b)
            # Mutate the resulting offspring
            offspring_a = self._mutation(offspring_a)
            offspring_b = self._mutation(offspring_b)
            # Select individuals for the next generation
            subpop = self._survivor_selection(subpop, fitness)
            # Add new individuals to the subpopulation
            next_subpop = np.vstack([subpop, [offspring_a, offspring_b]])

        return next_subpop
