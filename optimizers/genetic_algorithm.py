import numpy as np
from tqdm import tqdm


class BinaryGeneticAlgorithm():
    """ Binary genetic algorithm.

    Attributes
    ----------
    n_features: int
        Number of features, that is, decision variables.
    sample_size: int
        Sample size of subpopulation to be used in Tournament Selection.
    best_individual: np.ndarray
        Individual from the subpopulation with the best evaluation.
    best_fitness: float
        Evaluation of the best individual of the subpopulation.
    """

    def __init__(self,
                 subpop_size: int,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 evaluator,
                 mut_prob: float = 0.05,
                 cross_prob: float = 0.80,
                 sample_perc: float = 0.10):
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
            mut_prob: float, default 0.05
                Probability of a mutation occurring.
            cross_prob: float, default 0.80
                Probability of a crossover occurring.
            sample_perc: float, default 0.10
                Percentage of subpopulation size that will be sampled for Tournament Selection.
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
        # Mutation probability
        self.mut_prob = mut_prob
        # Crossover probability
        self.cross_prob = cross_prob
        # Number of features
        self.n_features = X_train.shape[1]
        # At least a sample of 2 individuals
        self.sample_size = max(round(sample_perc * subpop_size), 2)
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

    def _tournament_selection(self, subpop, fobjs):
        """ Tournament selection. """

        # Indexes of selected parents
        selected_idxs = np.zeros((2,))

        for i in range(2):

            # Candidate individuals
            sample_idxs = np.random.choice(range(self.subpop_size),
                                           size=self.sample_size,
                                           replace=False)
            candidate_fitness = np.array(fobjs)[sample_idxs]        

            # Select the best individual in the tournament
            selected_idxs[i] = np.argmax(candidate_fitness)

        # Selected parents
        parent_a, parent_b = subpop[selected_idxs.astype(int)]

        return parent_a, parent_b

    def _survivor_selection(self, subpop, fobjs):
        """ Selection of individuals who will remain in the subpopulation. """
        
        # Select the two worst individuals (lower fitness) in the subpopulation
        idxs = np.argsort(fobjs)[:2]
        
        # Eliminate the two worst individuals in the subpopulation
        subpop = np.delete(subpop, idxs, axis=0)
        fobjs = np.delete(fobjs, idxs)
        
        return subpop, fobjs

    def _evaluate(self, individual):
        """ Evaluate the current individual using the evaluator. """

        # If no feature is selected
        if individual.sum() == 0:
            # Select one at random
            pos = np.random.choice(np.arange(individual.shape[0]))
            individual[pos] = 1
        # Evaluate individual
        metric = self.evaluator.evaluate(solution=individual,
                                         X_train=self.X_train,
                                         y_train=self.y_train,
                                         X_test=self.X_test,
                                         y_test=self.y_test)

        return metric

    def evolve(self, subpop, fobjs):
        """ Evolve a subpopulation in a single generation.

        Parameters
        ----------
        subpop: np.ndarray
            Individuals of the subpopulation, where each individual is an array of size equal to
            the number of features.
        fobjs: list
            Evaluation of all individuals in the subpopulation.

        Returns
        -------
        subpop: np.ndarray
            Individuals of the subpopulation, where each individual is an array of size equal to
            the number of features.
        fobjs: list
            Evaluation of all individuals in the subpopulation.
        """

        # Select parents through Tournament Selection
        parent_a, parent_b = self._tournament_selection(subpop, fobjs)

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
        fobjs = np.append(fobjs, [fitness_a, fitness_b])
        
        # Select individuals for the next generation
        subpop, fobjs = self._survivor_selection(subpop, fobjs)

        # Update new best individual
        self.best_fitness = np.max(fobjs)
        self.best_individual = subpop[np.argmax(fobjs)]

        return subpop, fobjs
