import  numpy as np


class SequentialFeatureGrouping():
    """
    Decompose the problem (a collection of features) sequentially.
    """

    def __init__(self, n_subcomps: int):
        """
        Parameters
        ----------
        n_subcomps: int
            Number of subcomponents, where each subcomponent is a subset of features.
        """

        self.n_subcomps = n_subcomps

    def decompose(self, X: np.array):
        """
        Divide an n-dimensional problem into m subproblems.

        Parameters
        ----------
        X: np.array
            n-dimensional input data.

        Returns
        -------
        subcomponents: list
            Subcomponents, where each subcomponent is an array that can be accessed by indexing
            the list.
        subcomp_sizes: list
            Number of features in each subcomponent.
        """
        # Decompose n-dimensional problem into m subproblems
        subcomponents = np.array_split(X, indices_or_sections=self.n_subcomps, axis=1)
        # Number of features in each subcomponent
        subcomp_sizes = [subcomp.shape[1] for subcomp in subcomponents]

        return subcomponents, subcomp_sizes
