import copy
import logging
import numpy as np
from tqdm import tqdm
from utils.datasets import DataLoader
from evaluation.wrapper import WrapperEvaluation
from decomposition.random import RandomFeatureGrouping
from collaboration.random import SingleRandomCollaboration
from initialization.random import RandomBinaryInitialization
from optimizers.genetic_algorithm import BinaryGeneticAlgorithm


class CCFSRFG():
    """ Cooperative Co-Evolutionary-Based Feature Selection with Random Feature Grouping.

    Rashid, A. N. M., et al. "Cooperative co-evolution for feature selection in Big Data with
    random feature grouping." Journal of Big Data 7.1 (2020): 1-42.

    Attributes
    ----------
    subcomp_sizes: list
        Number of features in each subcomponent.
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
    feature_idxs: np.ndarray
        Shuffled list of feature indexes.
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

    def _init_decomposer(self):
        """Instantiate feature grouping method."""
        self.decomposer = RandomFeatureGrouping(n_subcomps=self.n_subcomps, seed=self.seed)

    def _init_evaluator(self):
        """Instantiate evaluation method."""
        self.evaluator = WrapperEvaluation(task=self.conf["wrapper"]["task"],
                                           model_type=self.conf["wrapper"]["model_type"],
                                           eval_function=self.conf["wrapper"]["eval_function"])

    def _init_collaborator(self):
        """Instantiate collaboration method."""
        self.collaborator = SingleRandomCollaboration(seed=self.seed)

    def _init_subpop_initializer(self):
        """Instantiate subpopulation initialization method."""
        self.initializer = RandomBinaryInitialization(data=self.data,
                                                      subcomp_sizes=self.subcomp_sizes,
                                                      subpop_sizes=self.subpop_sizes,
                                                      evaluator=self.evaluator,
                                                      collaborator=self.collaborator,
                                                      penalty=self.conf["coevolution"]["penalty"],
                                                      weights=self.conf["coevolution"]["weights"])

    def _init_optimizers(self):
        """Instantiate evolutionary algorithms to evolve each subpopulation."""
        self.optimizers = list()
        # Instantiate an optimizer for each subcomponent
        for i in range(self.n_subcomps):
            optimizer = BinaryGeneticAlgorithm(subpop_size=self.subpop_sizes[i],
                                               X_train=self.data.S_train[i],
                                               y_train=self.data.y_train,
                                               X_test=self.data.S_val[i],
                                               y_test=self.data.y_val,
                                               evaluator=self.evaluator,
                                               conf=self.conf)
            self.optimizers.append(optimizer)

    def _problem_decomposition(self):
        """Decompose the problem into smaller subproblems."""
        # Decompose features in the training set
        self.data.S_train, self.subcomp_sizes, self.feature_idxs = self.decomposer.decompose(
            X=self.data.X_train)
        # Decompose features in the validation set
        self.data.S_val, _, _ = self.decomposer.decompose(X=self.data.X_val,
                                                          feature_idxs=self.feature_idxs)
        # Reorder the data according to shuffling in feature decomposition
        self.data.X_train = self.data.X_train[:, self.feature_idxs].copy()
        self.data.X_val = self.data.X_val[:, self.feature_idxs].copy()
        self.data.X_test = self.data.X_test[:, self.feature_idxs].copy()

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

    def _get_best_individuals(self):
        """Get the best individual from each subpopulation."""
        # Current best individual of each subpopulation
        self.current_best = dict()
        # For each subpopulation
        for i in range(self.n_subcomps):
            # TODO Would the best individual be the one with the highest global fitness or the highest local fitness?
            # best_ind_idx = np.argmax(self.local_fitness[i])
            best_ind_idx = np.argmax(self.global_fitness[i])
            self.current_best[i] = dict()
            self.current_best[i]["individual"] = self.subpops[i][best_ind_idx]
            self.current_best[i]["local_fitness"] = self.local_fitness[i][best_ind_idx]
            self.current_best[i]["context_vector"] = self.context_vectors[i][best_ind_idx]
            self.current_best[i]["global_fitness"] = self.global_fitness[i][best_ind_idx]

    def _get_global_best(self):
        """Get the globally best context vector."""
        best_idx = np.argmax([best["global_fitness"] for best in self.current_best.values()])
        best_fitness = self.current_best[best_idx]["global_fitness"]
        best_context_vector = self.current_best[best_idx]["context_vector"]

        return best_context_vector, best_fitness

    def _evaluate(self, context_vector):
        """Evaluate the given context vector using the evaluator."""
        # Evaluate the context vector
        fitness = self.evaluator.evaluate(solution=context_vector,
                                          X_train=self.data.X_train,
                                          y_train=self.data.y_train,
                                          X_test=self.data.X_val,
                                          y_test=self.data.y_val)

        # Penalize large subsets of features
        if self.conf["coevolution"]["penalty"]:
            features_p = context_vector.sum()/context_vector.shape[0]
            global_fitness = self.conf["coevolution"]["weights"][0] * fitness -\
                self.conf["coevolution"]["weights"][1] * features_p

        return global_fitness

    def optimize(self):
        """Solve the feature selection problem through optimization."""
        # Decompose problem
        self._problem_decomposition()
        # Initialize subpopulations
        self._init_subpopulations()
        # Instantiate optimizers
        self._init_optimizers()

        # Find the best individual and context vector from each subpopulation
        self._get_best_individuals()
        # Best individuals in a list
        best_individuals = [best["individual"] for best in self.current_best.values()]
        # Select the globally best context vector
        self.best_context_vector, self.best_global_fitness = self._get_global_best()

        # Set the number of generations counter with the first generation
        n_gen = 1
        # Initialize the optimization progress bar
        progress_bar = tqdm(total=self.conf["coevolution"]["max_gen"], desc="Generations")

        # Iterate up to the maximum number of generations
        while n_gen <= self.conf["coevolution"]["max_gen"]:
            # Append current best global fitness
            self.convergence_curve.append(self.best_global_fitness)
            # Evolve each subpopulation using a genetic algorithm
            for i in range(self.n_subcomps):
                self.subpops[i], self.local_fitness[i] = self.optimizers[i].evolve(
                    self.subpops[i],
                    self.local_fitness[i])
                # Best individuals from the previous generation as collaborators for each
                # individual in the current generation
                for j in range(self.subpop_sizes[i]):
                    collaborators = copy.deepcopy(best_individuals)
                    collaborators[i] = self.subpops[i][j]
                    context_vector = self.collaborator.build_context_vector(collaborators)
                    # Update the context vector
                    # TODO Should I store the best context vector of each subpopulation across generations?
                    self.context_vectors[i][j] = context_vector
                    # Update the global evaluation
                    self.global_fitness[i][j] = self._evaluate(context_vector)
            # Find the best individual from each subpopulation
            self._get_best_individuals()
            # Update list of best individuals
            best_individuals = [best["individual"] for best in self.current_best.values()]
            # Select the globally best context vector
            best_context_vector, best_global_fitness = self._get_global_best()
            # Update best context vector
            if self.best_global_fitness < best_global_fitness:
                # Enable logger if specified
                logging.getLogger().disabled = False if self.verbose else True
                # Show improvement
                old_best_fitness = round(self.best_global_fitness, 4)
                new_best_fitness = round(best_global_fitness, 4)
                logging.info(f"\nUpdate fitness from {old_best_fitness} to {new_best_fitness}.")
                # Update best context vector
                self.best_context_vector = best_context_vector
                # Update best global fitness
                self.best_global_fitness = best_global_fitness
            # Increase number of generations
            n_gen += 1
            # Update progress bar
            progress_bar.update(1)
        # Close progress bar after optimization
        progress_bar.close()
