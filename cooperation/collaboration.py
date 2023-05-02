import random
import numpy as np


class Collaboration():
    """
    A template for a collaborative approach between individuals from different subpopulations.
    """

    def __init__(self, seed: int = None):
        """
        Parameters
        ----------
        seed: int
            Numerical value that generates a new set or repeats pseudo-random numbers. It is
            defined in stochastic processes to ensure reproducibility.
        """
        # Set the seed value
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed=seed)

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
