import  numpy as np
from decomposition.grouping import FeatureGrouping


class SequentialFeatureGrouping(FeatureGrouping):
    """
    Decompose the problem (a collection of features) sequentially.
    """

    def decompose(self, X: np.ndarray):
        """
        Divide an n-dimensional problem into m subproblems.

        Parameters
        ----------
        X: np.ndarray
            n-dimensional input data.

        Returns
        -------
        subcomponents: list
            Subcomponents, where each subcomponent is an array that can be accessed by indexing
            the list.
        subcomp_sizes: list
            Number of features in each subcomponent.
        """
        # Decompose sequentially according to the original dataset order
        subcomponents = self._get_subcomponents(X=X)

        return subcomponents
