import os
import gc
import logging
import numpy as np
from abc import ABC, abstractmethod
from ..utils.datasets import DataLoader


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
    feature_idxs : np.ndarray
        List of feature indexes.
    best_context_vectors: list
        Best context vector in each generation.
    """

    def __init__(self, data: DataLoader, conf: dict, verbose: bool = True):
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
        # Evaluation mode
        self.eval_mode = self.data.splitter_type
        # Configuration parameters
        self.conf = conf
        # Initializes the components of the cooperative co-evolutionary algorithm
        self._init_evaluator()
        self._init_decomposer()
        self._init_collaborator()
        # List to store the best global fitness in each generation
        self.convergence_curve = list()
        # List to store the best context vector in each generation
        self.best_context_vectors = list()
        # Maximum number of best context vectors to store
        self.max_best_context_vectors = conf["coevolution"].get("max_best_context_vectors")
        # Optional memory profiling
        self.memory_profile = conf["coevolution"].get("memory_profile", False)
        self.memory_log_every = conf["coevolution"].get("memory_log_every", 0)

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
            current_best[i]["fitness"] = fitness[i][best_ind_idx]

        return current_best

    def _get_global_best(self):
        """Get the globally best context vector."""
        best_idx = np.argmax([best["fitness"] for best in self.current_best.values()])
        best_fitness = self.current_best[best_idx]["fitness"]
        best_context_vector = self.current_best[best_idx]["context_vector"].copy()
        return best_context_vector, best_fitness

    def _record_best_context_vector(self, context_vector: np.ndarray) -> None:
        """Store best context vectors while keeping memory bounded."""
        self.best_context_vectors.append(context_vector.copy())
        max_keep = self.max_best_context_vectors
        if max_keep is None:
            return
        if isinstance(max_keep, int) and max_keep > 0:
            if len(self.best_context_vectors) > max_keep:
                self.best_context_vectors = self.best_context_vectors[-max_keep:]
        if isinstance(max_keep, int) and max_keep <= 0:
            self.best_context_vectors.clear()

    def _get_rss_mb(self) -> float:
        """Get resident set size (RSS) in megabytes (MB) when available."""
        if os.name != "posix":
            return None
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as status_file:
                for line in status_file:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        return round(int(parts[1]) / 1024, 2)  # Convert from kB to MB
        except FileNotFoundError:
            return None
        return None

    def _sum_nbytes(self, obj) -> int:
        """Recursively sum the number of bytes of a given object and its contents."""
        total_size = 0
        if isinstance(obj, np.ndarray):
            total_size += obj.nbytes
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                total_size += self._sum_nbytes(item)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                total_size += self._sum_nbytes(key)
                total_size += self._sum_nbytes(value)
        return total_size

    def _log_memory_usage(self, stage: str, n_gen: int = None) -> None:
        """Log memory usage of objects to help find bottenecks."""
        if not self.memory_profile:
            return
        if n_gen is not None:
            if not self.memory_log_every or (n_gen % self.memory_log_every) != 0:
                return
        rss_mb = self._get_rss_mb()
        subpops_mb = round(self._sum_nbytes(getattr(self, "subpops", None)) / (1024 ** 2), 2)
        context_vectors_mb = round(self._sum_nbytes(getattr(self, "context_vectors", None)) / (1024 ** 2), 2)
        best_context_vectors_mb = round(self._sum_nbytes(getattr(self, "best_context_vectors", None)) / (1024 ** 2), 2)
        training_mb = round(self._sum_nbytes(getattr(self.data, "X_train", None)) / (1024 ** 2), 2)
        test_mb = round(self._sum_nbytes(getattr(self.data, "X_test", None)) / (1024 ** 2), 2)
        raw_training_mb = round(self._sum_nbytes(getattr(self.data, "_raw_X_train", None)) / (1024 ** 2), 2)
        message = (
            f"[Memory Usage] Stage: {stage} | "
            f"{'' if n_gen is None else f' Generation: {n_gen} | '}"
            f"RSS: {rss_mb} MB | "
            f"Subpopulations: {subpops_mb} MB | "
            f"Context Vectors: {context_vectors_mb} MB | "
            f"Best Context Vectors: {best_context_vectors_mb} MB | "
            f"Training data: {training_mb} MB | "
            f"Test data: {test_mb} MB | "
            f"Raw Training data: {raw_training_mb} MB"
        )
        # Log message even if logging is disabled
        logger = logging.getLogger()
        was_disabled = logger.disabled
        logger.disabled = False
        logging.info(message)
        logger.disabled = was_disabled

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
        self.subpops = self.initializer.subpops
        # Context vectors
        self.context_vectors = self.initializer.context_vectors
        # Evaluations of context vectors
        self.fitness = self.initializer.fitness
        delattr(self, "initializer")
        gc.collect()

    def _problem_decomposition(self):
        """Decompose the problem into smaller subproblems."""
        # Decompose only once to use the same feature indexes on all k-folds
        Xk_train, _, _, _ = self.data.get_fold(0, normalize=False)
        _, self.feature_idxs = self.decomposer.decompose(X=Xk_train)
        self.feature_importances = self.feature_importances[self.feature_idxs]
        # Reorder training set according to the shuffling in the feature decomposition
        self.data.X_train = self.data.X_train[:, self.feature_idxs]
        # Reorder test set according to the shuffling in the feature decomposition
        self.data.X_test = self.data.X_test[:, self.feature_idxs]
        # Keep raw training data aligned for fold normalization if it exists
        if hasattr(self.data, "_raw_X_train"):
            self.data._raw_X_train = self.data._raw_X_train[:, self.feature_idxs]
        # Update 'n_subcomps' when it starts with NoneType
        self.n_subcomps = self.decomposer.n_subcomps
        # Update 'subcomp_sizes' when it starts with an empty list
        self.subcomp_sizes = self.decomposer.subcomp_sizes