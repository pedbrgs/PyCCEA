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
        seed: int, default None
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
            Shuffled list of feature indexes. It is passed as a parameter if it has been
            previously generated.

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
        if feature_idxs is None:
            # List of feature indexes
            feature_idxs = np.arange(X.shape[1])
            # Shuffle the list of feature indexes
            np.random.shuffle(feature_idxs)
        # Shuffle the data features according to the indexes
        X = X[:, feature_idxs].copy()
        # Decompose the problem
        subcomponents = self._get_subcomponents(X=X)   
   
        return subcomponents, feature_idxs
