import  numpy as np


class UniformStaticGrouping():
    """
    Decompose the set of decision variables uniformly.

    Attributes
    ----------
    n: int
        Number of decision variables.
    s: int
        Number of variables in a subcomponent.
    """

    def __init__(self, m: int):
        """
        Parameters
        ----------
        m: int
            Number of subcomponents, where each subcomponent is a subset of decision variables.
        """

        self.m = m

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
        # Number of decision variables
        self.n = X.shape[1]
        if self.n % self.m != 0:
            raise AssertionError(
                f"{self.n} decision variables is not divisible by {self.m} subcomponents"
                )
        # Number of decision variables in each subcomponent
        self.s = int(self.n/self.m)
        # Decompose n-dimensional problem into m n/m-dimensional subproblems
        subproblems = np.array_split(X, indices_or_sections=self.m, axis=1)

        return subproblems
