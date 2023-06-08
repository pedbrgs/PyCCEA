import random
import  numpy as np
from decomposition.grouping import FeatureGrouping


class RandomFeatureGrouping(FeatureGrouping):
    """
    Decompose the problem (a collection of features) randomly.
    """

    def __init__(self, n_subcomps: int = None, subcomp_sizes: list = list(), seed: int = None):
        super().__init__(n_subcomps, subcomp_sizes)
        """
        Parameters
        ----------
        n_subcomps: int
            Number of subcomponents, where each subcomponent is a subset of features.
        subcomp_sizes: list
            Number of features in each subcomponent.
        seed: int
            Numerical value that generates a new set or repeats pseudo-random numbers. It is
            defined in stochastic processes to ensure reproducibility.
        """
        # Set the seed value
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed=seed)

    def decompose(self, X: np.ndarray, feature_idxs: np.ndarray = None):
        """
        Divide an n-dimensional problem into m subproblems.

        Parameters
        ----------
        X: np.ndarray
            n-dimensional input data.
        feature_idxs: np.ndarray, default None
            Shuffled list of feature indexes.

        Returns
        -------
        subcomponents: list
            Subcomponents, where each subcomponent is an array that can be accessed by indexing
            the list.
        subcomp_sizes: list
            Number of features in each subcomponent.
        feature_idxs: np.ndarray
            Shuffled list of feature indexes.
        """
        if not isinstance(feature_idxs, np.ndarray):
            # List of feature indexes
            feature_idxs = np.arange(X.shape[1])
            # Shuffle the list of feature indexes
            np.random.shuffle(feature_idxs)
        # Shuffle the data features according to the indexes
        X = X[:, feature_idxs].copy()
        # Decompose the problem according to the given parameters
        if self.subcomp_sizes:
            if X.shape[1] != sum(self.subcomp_sizes):
                raise AssertionError(
                    f"The sum of subcomponent sizes ({sum(self.subcomp_sizes)}) is not equal to "
                    f"the number of features ({X.shape[1]}). Check parameter 'subcomp_sizes' "
                    "in the configuration file."
                )
            # Indices to partition the problem
            indices = np.cumsum(self.subcomp_sizes)[:-1]
            # Decompose n-dimensional problem into subproblems, where the i-th subproblem has
            # 'subcomp_sizes[i]' features
            subcomponents = np.split(X, indices, axis=1)
            # Number of subcomponents
            self.n_subcomps = len(subcomponents)
        else:
            # Decompose n-dimensional problem into 'n_subcomps' subproblems
            subcomponents = np.array_split(X, indices_or_sections=self.n_subcomps, axis=1)
            # Number of features in each subcomponent
            self.subcomp_sizes = [subcomp.shape[1] for subcomp in subcomponents]

        return subcomponents, feature_idxs
