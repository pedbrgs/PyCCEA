import random
import  numpy as np


class RandomFeatureGrouping():
    """
    Decompose the problem (a collection of features) randomly.
    """

    def __init__(self, n_subcomps: int, seed: int = None):
        """
        Parameters
        ----------
        n_subcomps: int
            Number of subcomponents, where each subcomponent is a subset of features.
        seed: int
            Numerical value that generates a new set or repeats pseudo-random numbers. It is
            defined in stochastic processes to ensure reproducibility.
        """

        self.n_subcomps = n_subcomps
        # Set the seed value
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed=seed)

    def decompose(self, X: np.array, feature_idxs: np.array = None):
        """
        Divide an n-dimensional problem into m subproblems.

        Parameters
        ----------
        X: np.array
            n-dimensional input data.
        feature_idxs: np.array, default None
            Shuffled list of feature indexes.

        Returns
        -------
        subcomponents: list
            Subcomponents, where each subcomponent is an array that can be accessed by indexing
            the list.
        subcomp_sizes: list
            Number of features in each subcomponent.
        feature_idxs: np.array
            Shuffled list of feature indexes.
        """
        if not isinstance(feature_idxs, np.ndarray):
            # List of feature indexes
            feature_idxs = np.arange(X.shape[1])
            # Shuffle the list of feature indexes
            np.random.shuffle(feature_idxs)
        # Shuffle the data features according to the indexes
        X = X[:, feature_idxs].copy()
        # Decompose n-dimensional problem into m subproblems
        subcomponents = np.array_split(X, indices_or_sections=self.n_subcomps, axis=1)
        # Number of features in each subcomponent
        subcomp_sizes = [subcomp.shape[1] for subcomp in subcomponents]

        return subcomponents, subcomp_sizes, feature_idxs
