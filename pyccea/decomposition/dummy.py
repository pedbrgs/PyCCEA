import numpy as np
from ..decomposition.grouping import FeatureGrouping


class DummyFeatureGrouping(FeatureGrouping):
    """Decompose the problem (a collection of features) according to preset indexes."""

    def __init__(self,
                 n_subcomps: int = None,
                 subcomp_sizes: list = list(),
                 feature_idxs: np.ndarray = None):
        super().__init__(n_subcomps, subcomp_sizes)
        """
        Parameters
        ----------
        n_subcomps : int
            Number of subcomponents, where each subcomponent is a subset of features.
        subcomp_sizes : list
            Number of features in each subcomponent.
        feature_idxs : np.ndarray
            Indexes of features sorted according to a predetermined method.
        """
        self.feature_idxs = feature_idxs.copy()

    def decompose(self, X: np.ndarray):
        """Divide an n-dimensional problem into m subproblems.

        Parameters
        ----------
        X: np.ndarray
            n-dimensional input data.

        Returns
        -------
        subcomponents : list
            Subcomponents, where each subcomponent is an array that can be accessed by indexing
            the list.
        feature_idxs : np.ndarray, default None
            Indexes of features sorted according to a predetermined method.
        """
        # Shuffle the data features according to the indexes
        X = X[:, self.feature_idxs].copy()
        # Decompose the problem
        subcomponents = self._get_subcomponents(X=X)

        return subcomponents, self.feature_idxs
