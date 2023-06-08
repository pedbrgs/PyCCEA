import logging
import numpy as np
from tqdm import tqdm
from coevolution.ccea import CCEA
from evaluation.wrapper import WrapperEvaluation
from cooperation.best import SingleBestCollaboration
from cooperation.random import SingleRandomCollaboration
from initialization.random import RandomBinaryInitialization
from optimizers.genetic_algorithm import BinaryGeneticAlgorithm


class CCFSRFG(CCEA):
    """ Cooperative Co-Evolutionary-Based Feature Selection with Random Feature Grouping.

    Rashid, A. N. M., et al. "Cooperative co-evolution for feature selection in Big Data with
    random feature grouping." Journal of Big Data 7.1 (2020): 1-42.

    Attributes
    ----------
    subcomp_sizes: list
        Number of features in each subcomponent.
    feature_idxs: np.ndarray
        Shuffled list of feature indexes.
    """

    def _init_collaborator(self):
        """Instantiate collaboration method."""
        self.best_collaborator = SingleBestCollaboration()
        self.random_collaborator = SingleRandomCollaboration(seed=self.seed)

    def _init_evaluator(self):
        """Instantiate evaluation method."""
        self.evaluator = WrapperEvaluation(task=self.conf["wrapper"]["task"],
                                           model_type=self.conf["wrapper"]["model_type"],
                                           eval_function=self.conf["wrapper"]["eval_function"],
                                           eval_mode=self.conf["wrapper"]["eval_mode"],
                                           kfolds=self.conf["wrapper"].get("kfolds"))

    def _init_subpop_initializer(self):
        """Instantiate subpopulation initialization method."""
        self.initializer = RandomBinaryInitialization(data=self.data,
                                                      subcomp_sizes=self.subcomp_sizes,
                                                      subpop_sizes=self.subpop_sizes,
                                                      evaluator=self.evaluator,
                                                      collaborator=self.random_collaborator,
                                                      penalty=self.conf["coevolution"]["penalty"],
                                                      weights=self.conf["coevolution"]["weights"])

    def _init_optimizers(self):
        """Instantiate evolutionary algorithms to evolve each subpopulation."""
        self.optimizers = list()
        # Instantiate an optimizer for each subcomponent
        for i in range(self.n_subcomps):
            optimizer = BinaryGeneticAlgorithm(subpop_size=self.subpop_sizes[i],
                                               n_features=self.subcomp_sizes[i],
                                               conf=self.conf)
            self.optimizers.append(optimizer)

    def _problem_decomposition(self):
        """Decompose the problem into smaller subproblems."""
        # Decompose features in the training set
        self.data.S_train, self.feature_idxs = self.decomposer.decompose(X=self.data.X_train)
        # Update 'n_subcomps' when it starts with NoneType
        self.n_subcomps = self.decomposer.n_subcomps
        # Update 'subcomp_sizes' when it starts with an empty list
        self.subcomp_sizes = self.decomposer.subcomp_sizes.copy()
        # Reorder train and test data according to shuffling in feature decomposition
        self.data.X_train = self.data.X_train[:, self.feature_idxs].copy()
        if self.data.X_test:
            self.data.X_test = self.data.X_test[:, self.feature_idxs].copy()

        # Train-validation
        if self.conf["wrapper"]["eval_mode"] == 'train_val':
            # Decompose features in the validation set
            self.data.S_val, _ = self.decomposer.decompose(X=self.data.X_val,
                                                           feature_idxs=self.feature_idxs)
            # Reorder validation data according to shuffling in feature decomposition
            self.data.X_val = self.data.X_val[:, self.feature_idxs].copy()
        # Cross-validation
        else:
            # It is just to avoid crashing when initializing the optimizers.
            # It will not be used in the cross-validation mode.
            self.data.S_val = np.full(shape=(self.n_subcomps), fill_value=None)

    def _evaluate(self, context_vector):
        """Evaluate the given context vector using the evaluator."""
        # Evaluate the context vector
        evaluation = self.evaluator.evaluate(solution=context_vector.copy(),
                                             X_train=self.data.X_train,
                                             y_train=self.data.y_train,
                                             X_val=self.data.X_val,
                                             y_val=self.data.y_val)

        # Penalize large subsets of features
        if self.conf["coevolution"]["penalty"]:
            penalty = context_vector.sum()/self.n_features
            fitness = self.conf["coevolution"]["weights"][0] * evaluation -\
                self.conf["coevolution"]["weights"][1] * penalty

        return fitness
