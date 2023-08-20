import copy
import logging
import numpy as np
from abc import ABC, abstractmethod
from utils.datasets import DataLoader


class CCEA(ABC):
    """ An abstract class for a Cooperative Co-Evolutionary-Based Feature Selection Algorithm.

    Attributes
    ----------
    subpop_sizes: list
        Subpopulation sizes, that is, the number of individuals in each subpopulation.
    decomposer: object of one of the decomposition classes
        Responsible for decompose the problem into smaller subproblems.
    collaborator: object of one of the collaboration classes.
        Responsible for selecting collaborators for individuals.
    fitness_function: object of one of the fitness classes.
        Responsible for evaluating individuals, that is, subsets of features.
    initializer: object of one of the subpopulation initializers
        Responsible for initializing all individuals of all subpopulations.
    optimizers: list of objects of optimizer classes
        Responsible for evolving each of the subpopulations individually.
    subpops: list
        Individuals from all subpopulations. Each individual is represented by a binary
        n-dimensional array, where n is the number of features. If there is a 1 in the i-th
        position of the array, it indicates that the i-th feature should be considered and if
        there is a 0, it indicates that the feature should not be considered.
    fitness: list
        Evaluation of all context vectors from all subpopulations.
    context_vectors: list
        Complete problem solutions.
    convergence_curve: list
        Best global fitness in each generation.
    current_best: dict
        Current best individual of each subpopulation and its respective evaluation.
    best_context_vector: np.ndarray
        Best solution of the complete problem.
    best_fitness: float
        Evaluation of the best solution of the complete problem.
    """

    def __init__(self,
                 data: DataLoader,
                 conf: dict,
                 verbose: bool = True):
        """
        Parameters
        ----------
        data: DataLoader
            Container with process data and training and test sets.
        conf: dict
            Configuration parameters of the cooperative coevolutionary algorithm.
        verbose: bool, default True
            If True, show the improvements obtained from the optimization process.
        """
        # Seed
        self.seed = conf["coevolution"].get("seed")
        # Verbose
        self.verbose = verbose
        # Data
        self.data = data
        # Number of features
        self.n_features = self.data.n_features
        # Size of each subpopulation
        self.subpop_sizes = conf["coevolution"]["subpop_sizes"]
        # Number of subcomponents
        self.n_subcomps = conf["coevolution"].get("n_subcomps")
        if self.n_subcomps:
            if self.n_subcomps != len(self.subpop_sizes):
                if len(self.subpop_sizes) == 1:
                    subpop_size = self.subpop_sizes[0]
                    logging.info(f"Considering all subpopulations with size {subpop_size}.")
                    self.subpop_sizes = [subpop_size] * self.n_subcomps
                else:
                    raise AssertionError(
                        f"The number of subcomponents ({self.n_subcomps}) is not equal to the "
                        f"number of subpopulations ({len(self.subpop_sizes)}). Check parameters "
                        "'n_subcomps' and 'subpop_sizes' in the configuration file."
                    )
        # Number of features in each subcomponent
        self.subcomp_sizes = conf["coevolution"].get("subcomp_sizes")
        if self.subcomp_sizes:
            if len(self.subcomp_sizes) != len(self.subpop_sizes):
                raise AssertionError(
                    f"The number of subcomponents ({len(self.subcomp_sizes)}) is not equal to the"
                    f" number of subpopulations ({len(self.subpop_sizes)}). Check parameters "
                    "'subcomp_sizes' and 'subpop_sizes' in the configuration file."
                )
        # Configuration parameters
        self.conf = conf
        # Initializes the components of the cooperative co-evolutionary algorithm
        self._init_decomposer()
        self._init_evaluator()
        self._init_collaborator()
        # List to store the best global fitness in each generation
        self.convergence_curve = list()

        # Initialize logger with info level
        logging.basicConfig(encoding="utf-8", level=logging.INFO)
        # Reset handlers
        logging.getLogger().handlers = []
        # Add a custom handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(handler)

    @abstractmethod
    def _init_decomposer(self):
        """Instantiate feature grouping method."""
        pass

    @abstractmethod
    def _init_evaluator(self):
        """Instantiate evaluation method."""
        pass

    @abstractmethod
    def _init_collaborator(self):
        """Instantiate collaboration method."""
        pass

    @abstractmethod
    def _init_subpop_initializer(self):
        """Instantiate subpopulation initialization method."""
        pass

    @abstractmethod
    def _init_optimizers(self):
        """Instantiate evolutionary algorithms to evolve each subpopulation."""
        pass

    @abstractmethod
    def optimize(self):
        """Solve the feature selection problem through optimization."""
        pass

    def _get_best_individuals(self,
                              subpops: list,
                              fitness: list,
                              context_vectors: list):
        """
        Get the best individual from each subpopulation.

        Parameters
        ----------
        subpops: list
            Individuals from all subpopulations. Each individual is represented by a binary
            n-dimensional array, where n is the number of features. If there is a 1 in the i-th
            position of the array, it indicates that the i-th feature should be considered and if
            there is a 0, it indicates that the feature should not be considered.
        fitness: list
            Evaluation of all context vectors from all subpopulations.
        context_vectors: list
            Complete problem solutions.

        Returns
        -------
        current_best: dict
            Current best individual of each subpopulation and its respective evaluation.
        """
        # Current best individual of each subpopulation
        current_best = dict()
        # Number of subpopulations
        n_subpops = len(subpops)
        # For each subpopulation
        for i in range(n_subpops):
            best_ind_idx = np.argmax(fitness[i])
            current_best[i] = dict()
            current_best[i]["individual"] = subpops[i][best_ind_idx].copy()
            current_best[i]["context_vector"] = context_vectors[i][best_ind_idx].copy()
            current_best[i]["fitness"] = fitness[i][best_ind_idx].copy()

        return current_best

    def _get_global_best(self):
        """Get the globally best context vector."""
        best_idx = np.argmax([best["fitness"] for best in self.current_best.values()])
        best_fitness = self.current_best[best_idx]["fitness"]
        best_context_vector = self.current_best[best_idx]["context_vector"].copy()
        return best_context_vector, best_fitness

    def _init_subpopulations(self):
        """Initialize all subpopulations according to their respective sizes."""
        # Instantiate subpopulation initialization method
        self._init_subpop_initializer()
        # Build subpopulations
        # Number of subpopulations is equal to the number of subcomponents
        self.initializer.build_subpopulations()
        # Evaluate all individuals in each subpopulation
        # Number of individuals in each subpopulation is in the list of subcomponent sizes
        self.initializer.evaluate_individuals()
        # Subpopulations
        self.subpops = copy.deepcopy(self.initializer.subpops)
        # Context vectors
        self.context_vectors = copy.deepcopy(self.initializer.context_vectors)
        # Evaluations of context vectors
        self.fitness = copy.deepcopy(self.initializer.fitness)

    def _problem_decomposition(self):
        """Decompose the problem into smaller subproblems."""
        # Train-validation
        if self.conf["evaluation"]["eval_mode"] == 'train_val':
            # Decompose features in the training set
            _, self.feature_idxs = self.decomposer.decompose(X=self.data.X_train)
            # Reorder training, validation and test sets according to shuffling in the feature
            # decomposition
            self.data.X_train = self.data.X_train[:, self.feature_idxs].copy()
            self.data.X_val = self.data.X_val[:, self.feature_idxs].copy()
            self.data.X_test = self.data.X_test[:, self.feature_idxs].copy()
        # K-fold cross-validation
        else:
            for k in range(self.data.kfolds):
                Xk_train = self.data.train_folds[k][0]
                # Decompose only once to use the same feature indexes on all k-folds
                if k == 0:
                    _, self.feature_idxs = self.decomposer.decompose(X=Xk_train.copy())
                # Reorder training and validation folds built from the training set according to
                # the shuffling in the feature decomposition
                self.data.train_folds[k][0] = Xk_train[:, self.feature_idxs].copy()
                Xk_val = self.data.val_folds[k][0]
                self.data.val_folds[k][0] = Xk_val[:, self.feature_idxs].copy()
                # Reorder training and validation folds built from the test set according to the
                # shuffling in the feature decomposition
                if self.data.test_size > 0:
                    Xk_eval_train = self.data.eval_train_folds[k][0]
                    self.data.eval_train_folds[k][0] = Xk_eval_train[:, self.feature_idxs].copy()
                    Xk_eval_val = self.data.eval_val_folds[k][0]
                    self.data.eval_val_folds[k][0] = Xk_eval_val[:, self.feature_idxs].copy()
        # Update 'n_subcomps' when it starts with NoneType
        self.n_subcomps = self.decomposer.n_subcomps
        # Update 'subcomp_sizes' when it starts with an empty list
        self.subcomp_sizes = self.decomposer.subcomp_sizes.copy()
