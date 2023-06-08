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
        if self.n_subcomps and subcomp_sizes:
            raise AssertionError(
                f"Provide only one of the parameters: n_subcomps or subcomp_sizes."
            )

    @abstractmethod
    def decompose(self):
        """Divide an n-dimensional problem into m subproblems."""
        pass
