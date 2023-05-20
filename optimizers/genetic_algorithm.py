import numpy as np
from tqdm import tqdm


class BinaryGeneticAlgorithm():
    """ Binary genetic algorithm.

    Attributes
    ----------
    n_features: int
        Number of features.
    best_individual: np.ndarray
        Individual from the subpopulation with the best evaluation.
    best_fitness: float
        Evaluation of the best individual of the subpopulation.
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
                 subpop_size,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 evaluator,
                 conf: dict):
        """
        Parameters
        ----------
        subpop_size: int
            Number of individuals in the subpopulation.
        X_train: np.ndarray
            Train input data.
        y_train: np.ndarray
            Train output data.
        X_val: np.ndarray
            Validation input data.
        y_val: np.ndarray
            Validation output data.
        evaluator: object of one of the evaluation classes
            Responsible for evaluating individuals, that is, subsets of features.
        conf: dict
            Configuration parameters of the cooperative coevolutionary algorithm.
        """
        # Subpopulation size
        self.subpop_size = subpop_size
        # Training set
        self.X_train = X_train
        self.y_train = y_train
        # Validation set
        self.X_val = X_val
        self.y_val = y_val
        # Evaluator
        self.evaluator = evaluator
        # Configuration parameters
        self.conf = conf
        # Mutation rate
        self.mutation_rate = self.conf["optimizer"]["mutation_rate"]
        # Crossover rate
        self.crossover_rate = self.conf["optimizer"]["crossover_rate"]
        # Elite size
        self.elite_size = self.conf["optimizer"]["elite_size"]
        # Number of features
        self.n_features = X_train.shape[1]
        # Sample size for Tournament Selection
        self.tournament_sample_size = self.conf["optimizer"]["tournament_sample_size"]
        # Initialization of the best individual is utopian
        # Its representation is equivalent to selecting no features
        self.best_individual = np.zeros((self.n_features,))
        # Its evaluation, therefore, is zero (i.e., lowest possible value)
        self.best_fitness = 0

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

    def _survivor_selection(self, subpop, fitness):
        """ Selection of individuals who will remain in the subpopulation. """
        # Number of individuals that will be removed to keep the subpop_size constant
        n_worst_individuals = len(subpop) - self.subpop_size
        # If it is non-zero, remove the extra individuals
        if n_worst_individuals > 0:
            # Select the N worst individuals (lower fitness) in the current subpopulation
            idxs = np.argsort(fitness)[:n_worst_individuals]
            # Eliminate the two worst individuals in the subpopulation
            subpop = np.delete(subpop, idxs, axis=0)
            fitness = np.delete(fitness, idxs)
        return subpop, fitness

    def _evaluate(self, individual):
        """ Evaluate the current individual using the evaluator. """
        # Evaluate individual
        fitness = self.evaluator.evaluate(solution=individual,
                                          X_train=self.X_train,
                                          y_train=self.y_train,
                                          X_val=self.X_val,
                                          y_val=self.y_val)

        # Penalize large subsets of features
        if self.conf["coevolution"]["penalty"]:
            features_p = individual.sum()/individual.shape[0]
            fitness = self.conf["coevolution"]["weights"][0] * fitness -\
                self.conf["coevolution"]["weights"][1] * features_p

        return fitness

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
        next_subpop = subpop[n_bests]
        next_fitness = np.array(fitness)[n_bests].tolist()

        # Perform (subpop_size - elite_size) Tournament Selections to build the next generation
        for i in range(self.elite_size, self.subpop_size):

            # Select parents through Tournament Selection
            parent_a, parent_b = self._tournament_selection(subpop, fitness)
            # Recombine pairs of parents
            offspring = self._single_point_crossover(parent_a, parent_b)
            # Mutate the resulting offspring
            offspring = self._mutation(offspring)
            # Evaluate new candidate
            fvalue = self._evaluate(offspring)
            # Add new individual to the subpopulation
            next_subpop = np.vstack([next_subpop, offspring])
            next_fitness = np.append(next_fitness, fvalue)

        # Update new best individual
        self.best_fitness = np.max(next_fitness)
        self.best_individual = next_subpop[np.argmax(next_fitness)]

        return next_subpop, next_fitness
