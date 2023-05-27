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
    evaluator: object of one of the evaluation classes
        Responsible for evaluating individuals, that is, subsets of features.
    collaborator: object of one of the collaboration classes.
        Responsible for selecting collaborators for individuals.
    initializer: object of one of the subpopulation initializers
        Responsible for initializing all individuals of all subpopulations.
    optimizers: list of objects of optimizer classes
        Responsible for evolving each of the subpopulations individually.
    subpops: list
        Individuals from all subpopulations. Each individual is represented by a binary
        n-dimensional array, where n is the number of features. If there is a 1 in the i-th
        position of the array, it indicates that the i-th feature should be considered and if
        there is a 0, it indicates that the feature should not be considered.
    local_fitness: list
        Evaluation of all individuals from all subpopulations.
    global_fitness: list
        Evaluation of all context vectors from all subpopulations.
    context_vectors: list
        Complete problem solutions.
    convergence_curve: list
        Best global fitness in each generation.
    current_best: dict
        Current best individual of each subpopulation and its respective evaluation.
    best_context_vector: np.ndarray
        Best solution of the complete problem.
    best_global_fitness: float
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
        self.seed = conf["coevolution"]["seed"]
        # Verbose
        self.verbose = verbose
        # Data
        self.data = data
        # Number of subcomponents
        self.n_subcomps = conf["coevolution"]["n_subcomps"]
        # Size of each subpopulation
        self.subpop_sizes = conf["coevolution"]["subpop_sizes"]
        if self.n_subcomps != len(self.subpop_sizes):
            raise AssertionError(
                f"The number of components ({self.n_subcomps}) is not equal to the number of "
                f"subpopulations ({len(self.subpop_sizes)}). Check parameters 'n_subcomps' and "
                "'subpop_sizes' in the configuration file."
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
    def _problem_decomposition(self):
        """Decompose the problem into smaller subproblems."""
        pass

    @abstractmethod
    def _evaluate(self, context_vector):
        """Evaluate the given context vector using the evaluator."""
        pass

    @abstractmethod
    def optimize(self):
        """Solve the feature selection problem through optimization."""
        pass

    def _get_global_best(self):
        """Get the globally best context vector."""
        best_idx = np.argmax([best["global_fitness"] for best in self.current_best.values()])
        best_global_fitness = self.current_best[best_idx]["global_fitness"]
        best_context_vector = self.current_best[best_idx]["context_vector"]
        return best_context_vector, best_global_fitness

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
        # Evaluations of individuals
        self.local_fitness = copy.deepcopy(self.initializer.local_fitness)
        # Context vectors
        self.context_vectors = copy.deepcopy(self.initializer.context_vectors)
        # Evaluations of context vectors
        self.global_fitness = copy.deepcopy(self.initializer.global_fitness)
