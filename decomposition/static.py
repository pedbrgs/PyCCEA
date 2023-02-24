import  numpy as np


class UniformStaticGrouping():
    """
    Decompose the problem (a collection of features) uniformly.

    Attributes
    ----------
    n_features: int
        Number of features.
    subcomp_size: int
        Number of features in a subcomponent.
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
        Divide an n-dimensional problem into m n/m-dimensional subproblems.

        Parameters
        ----------
        X: np.array
            n-dimensional input data.

        Returns
        -------
        subproblems: list
            Subproblems, where each subproblem is an array that can be accessed by indexing the
            list.
        """
        # Number of features
        self.n_features = X.shape[1]
        if self.n_features % self.n_subcomps != 0:
            raise AssertionError(
                f"{self.n_features} features is not divisible by {self.n_subcomps} subcomponents"
                )
        # Number of features in each subcomponent
        self.subcomp_size = int(self.n_features/self.n_subcomps)
        # Decompose n-dimensional problem into m n/m-dimensional subproblems
        subproblems = np.array_split(X, indices_or_sections=self.n_subcomps, axis=1)

        return subproblems
