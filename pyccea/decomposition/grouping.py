import  numpy as np
from abc import ABC, abstractmethod


class FeatureGrouping(ABC):
    """
    An abstract class for a feature grouping approach.
    """

    def __init__(self, n_subcomps: int = None, subcomp_sizes: list = list()):
        """
        Parameters
        ----------
        n_subcomps: int
            Number of subcomponents, where each subcomponent is a subset of features.
        subcomp_sizes: list
            Number of features in each subcomponent.
        """
        self.n_subcomps = n_subcomps
        self.subcomp_sizes = subcomp_sizes
        if self.n_subcomps and self.subcomp_sizes:
            raise AssertionError(
                f"Provide only one of the parameters: n_subcomps or subcomp_sizes."
            )

    def _get_subcomponents(self, X: np.ndarray):
        """
        Group features into subcomponents.

        When applying this method it is expected that the original order of the features has
        already been changed according to a feature grouping strategy that should based on the
        feature interactions or importances.

        Parameters
        ----------
        X: np.ndarray
            n-dimensional input data.

        Returns
        -------
        subcomponents: list
            Subcomponents, where each subcomponent is an array that can be accessed by indexing
            the list.
        """
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

        return subcomponents
