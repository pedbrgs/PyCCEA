import numpy as np
from abc import ABC


class Collaboration(ABC):
    """
    An abstract class for a collaborative method between individuals from different subpopulations
    """

    def __init__(self):
        pass

    def build_context_vector(self, collaborators):
        """
        Build a context vector, i.e., a complete problem solution composed of representative
        solutions from each subpopulation.

        Parameters
        ----------
        collaborators: list
            A single individual from each subpopulation that will collaborate with an individual.

        Returns
        -------
        context_vector: np.ndarray
            Complete problem solution composed of one individual from each subpopulation.
        """
        context_vector = np.hstack(collaborators)

        return context_vector
