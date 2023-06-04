import numpy as np
from tqdm import tqdm


class BinaryGeneticAlgorithm():
    """ Binary genetic algorithm.

    Attributes
    ----------
    mutation_rate: float
        Probability of a mutation occurring.
    crossover_rate: float
        Probability of a crossover occurring.
    elite_size: int
        Number of individuals that will be preserved from the current to the next generation. They
        are selected according to the fitness, i.e., if it is N, the N best individuals of the
        current generation will be preserved for the next generation.
    tournament_sample_size: int
        Sample size of the subpopulation that will be used in Tournament Selection.
    """

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
        # Elite size
        self.elite_size = self.conf["optimizer"]["elite_size"]
        # Number of features
        self.n_features = n_features
        # Sample size for Tournament Selection
        self.tournament_sample_size = self.conf["optimizer"]["tournament_sample_size"]

    def _single_point_crossover(self, parent_a, parent_b):
        """ Single point crossover. """
        # Random probability
        prob = float(np.random.uniform(low=0, high=1))

        # If the random probability is less than the crossover probability, then do the crossover
        if prob < self.crossover_rate:
            # Crossover point
            point = np.random.randint(low=1, high=self.n_features)
            # Offspring
            offspring = np.concatenate([parent_b[:point], parent_a[point:]], axis=0)
        else:
            # Offspring will be a copy of a parent
            parent_idx = np.random.choice(range(2))
            offspring = [parent_a, parent_b][parent_idx].copy()

        return offspring

    def _mutation(self, parent):
        """ Bit-flip mutation. """
        # Offspring
        offspring = parent.copy()
        # Random probabilities
        probs = np.random.uniform(low=0, high=1, size=self.n_features)
        # If the random probability is less than the mutation probability, then do the mutation
        pos = probs < self.mutation_rate
        # Flip values only of the selected positions
        offspring[pos] ^= 1

        return offspring

    def _tournament_selection(self, subpop, fitness):
        """ Tournament selection. """
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

    def evolve(self, subpop, fitness):
        """ Evolve a subpopulation in a single generation.

        Parameters
        ----------
        subpop: np.ndarray
            Individuals of the subpopulation, where each individual is an array of size equal to
            the number of features.
        fitness: list
            Evaluation of all individuals in the subpopulation.

        Returns
        -------
        subpop: np.ndarray
            Individuals of the subpopulation, where each individual is an array of size equal to
            the number of features.
        fitness: list
            Evaluation of all individuals in the subpopulation.
        """

        # Elitism: select the elite_size best individuals of the current generation to be in the
        # next generation
        n_bests = np.argsort(fitness)[::-1][:self.elite_size]
        next_subpop = subpop[n_bests].copy()

        # Perform (subpop_size - elite_size) Tournament Selections to build the next generation
        for i in range(self.elite_size, self.subpop_size):

            # Select parents through Tournament Selection
            parent_a, parent_b = self._tournament_selection(subpop, fitness)
            # Recombine pairs of parents
            offspring = self._single_point_crossover(parent_a, parent_b)
            # Mutate the resulting offspring
            offspring = self._mutation(offspring)
            # Add new individual to the subpopulation
            next_subpop = np.vstack([next_subpop, offspring])

        return next_subpop
