import  numpy as np
import pandas as pd

class UniformStaticGrouping():
    """
    Decompose the set of decision variables evenly.

    Attributes
    ----------
    n: int
        Number of decision variables.
    s: int
        Number of variables in a subcomponent.
    """

    def __init__(self, X: pd.DataFrame, m: int):
        """
        Parameters
        ----------
        X: pd.DataFrame
            n-dimensional input data.
        m: int
            Number of subcomponents, where each subcomponent is a subset of decision variables.
        """

        self.X = X
        self.n = X.shape[1]
        self.m = m

    def _decompose(self):
        # Number of variables in each subcomponent
        if int(self.n/self.m) % 2 == 0:
            raise AssertionError(
                f"{self.n} decision variables is not divisible by {self.m} subcomponents"
                )
        self.s = int(self.n/self.m)
        # Split problem into m n/m-dimensional subproblems
        subproblems = np.array_split(self.X.to_numpy(), indices_or_sections=self.m, axis=1)

        return subproblems

    def run(self):
        """
        Divide an n-dimensional problem into m n/m-dimensional subproblems.

        Returns
        -------
        subproblems: list
            Subproblems, where each subproblem is an numpy array that can be accessed by indexing
            the list.
        """
        # Decompose n-dimensional problem into m n/m-dimensional subproblems
        subproblems = self._decompose()

        return subproblems
