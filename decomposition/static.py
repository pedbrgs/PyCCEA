import  numpy as np
from decomposition.grouping import FeatureGrouping


class SequentialFeatureGrouping(FeatureGrouping):
    """Decompose the problem (a collection of features) sequentially."""

    def decompose(self, X: np.ndarray):
        """Divide an n-dimensional problem into m subproblems.

        Parameters
        ----------
        X : np.ndarray
            n-dimensional input data.

        Returns
        -------
        subcomponents : list
            Subcomponents, where each subcomponent is an array that can be accessed by indexing
            the list.
        feature_idxs : np.ndarray
            List of feature indexes starting from 0 to n_features-1.
        """
        # Decompose sequentially according to the original dataset order
        subcomponents = self._get_subcomponents(X=X)
        # Starting from the feature indexed at 0
        feature_idxs = np.arange(start=0, stop=X.shape[1], step=1)

        return subcomponents, feature_idxs
