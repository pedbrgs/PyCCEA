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
