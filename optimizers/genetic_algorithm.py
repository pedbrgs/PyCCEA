import numpy as np
from tqdm import tqdm


class BinaryGeneticAlgorithm():
    """ Binary genetic algorithm.

    Attributes
    ----------
    n_features: int
        Number of features.
    sample_size: int
        Sample size of subpopulation to be used in Tournament Selection.
    best_individual: np.ndarray
        Individual from the subpopulation with the best evaluation.
    best_fitness: float
        Evaluation of the best individual of the subpopulation.
    mut_prob: float
        Probability of a mutation occurring.
    cross_prob: float
        Probability of a crossover occurring.
    sample_perc: float
        Percentage of subpopulation size that will be sampled for Tournament Selection.
    """

    def __init__(self,
                 subpop_size,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
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
        X_test: np.ndarray
            Test input data.
        y_test: np.ndarray
            Test output data.
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
        # Test set
        self.X_test = X_test
        self.y_test = y_test
        # Evaluator
        self.evaluator = evaluator
        # Configuration parameters
        self.conf = conf
        # Mutation probability
        self.mut_prob = self.conf["optimizer"]["mut_prob"]
        # Crossover probability
        self.cross_prob = self.conf["optimizer"]["cross_prob"]
        # Number of features
        self.n_features = X_train.shape[1]
        # Sample size for Tournament Selection
        sample_size = self.conf["optimizer"]["sample_perc"] * self.subpop_size
        # At least a sample of 2 individuals
        self.sample_size = int(max(sample_size, 2))
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
        if prob < self.cross_prob:
            # Crossover point
            point = np.random.randint(low=1, high=self.n_features)
            # Offspring
            offspring_a = np.concatenate([parent_a[:point], parent_b[point:]], axis=0)
            offspring_b = np.concatenate([parent_b[:point], parent_a[point:]], axis=0)                
        else:
            # Offspring will be a copy of their parents
            offspring_a = parent_a.copy()
            offspring_b = parent_b.copy()

        return offspring_a, offspring_b

    def _mutation(self, parent):
        """ Bit-flip mutation. """
        # Offspring
        offspring = parent.copy()
        # Random probabilities
        probs = np.random.uniform(low=0, high=1, size=self.n_features)
        # If the random probability is less than the mutation probability, then do the mutation
        pos = probs < self.mut_prob
        # Flip values only of the selected positions
        offspring[pos] ^= 1

        return offspring

    def _tournament_selection(self, subpop, fitness):
        """ Tournament selection. """
        # Indexes of selected parents
        selected_idxs = np.zeros((2,))

        for i in range(2):

            # Candidate individuals
            sample_idxs = np.random.choice(range(self.subpop_size),
                                           size=self.sample_size,
                                           replace=False)
            candidate_fitness = np.array(fitness)[sample_idxs]

            # Select the best individual in the tournament
            selected_idxs[i] = np.argmax(candidate_fitness)

        # Selected parents
        parent_a, parent_b = subpop[selected_idxs.astype(int)]

        return parent_a, parent_b

    def _survivor_selection(self, subpop, fitness):
        """ Selection of individuals who will remain in the subpopulation. """
        # Select the two worst individuals (lower fitness) in the subpopulation
        idxs = np.argsort(fitness)[:2]
        
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
                                          X_test=self.X_test,
                                          y_test=self.y_test)

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
        # Select parents through Tournament Selection
        parent_a, parent_b = self._tournament_selection(subpop, fitness)

        # Recombine pairs of parents
        offspring_a, offspring_b = self._single_point_crossover(parent_a, parent_b)

        # Mutate the resulting offspring
        offspring_a = self._mutation(offspring_a)
        offspring_b = self._mutation(offspring_b)

        # Evaluate new candidates
        fitness_a = self._evaluate(offspring_a)
        fitness_b = self._evaluate(offspring_b)

        # Add new individuals to the subpopulation
        subpop = np.vstack([subpop, [offspring_a, offspring_b]])
        fitness = np.append(fitness, [fitness_a, fitness_b])
        
        # Select individuals for the next generation
        subpop, fitness = self._survivor_selection(subpop, fitness)

        # Update new best individual
        self.best_fitness = np.max(fitness)
        self.best_individual = subpop[np.argmax(fitness)]

        return subpop, fitness
