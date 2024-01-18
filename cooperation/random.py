import random
import numpy as np
from cooperation.collaboration import Collaboration


class SingleRandomCollaboration(Collaboration):
    """Set a random individual from each subpopulation as a collaborator to an individual."""

    def __init__(self, seed: int = None):
        """
        Parameters
        ----------
        seed : int
            Numerical value that generates a new set or repeats pseudo-random numbers. It is
            defined in stochastic processes to ensure reproducibility.
        """
        # Set the seed value
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed=seed)

    def get_collaborators(
            self,
            subpop_idx: int,
            indiv_idx: int,
            previous_subpops: list,
            current_subpops: list,
        ):
        """Randomly find a single collaborator from each subpopulation in the previous generation
        for the individual in the current generation given as a parameter.

        Parameters
        ----------
        subpop_idx : int
            Index of the subpopulation to which the individual belongs.
        indiv_idx : int
            Index of the individual in its respective subpopulation.
        previous_subpops : list
            Individuals from all subpopulations in the previous generation (already evaluated).
        current_subpops : list
            Individuals from all subpopulations of the current generation (will be evaluated).

        Returns
        -------
        collaborators : list
            A single individual from each subpopulation that will collaborate with the individual
            given as a parameter.
        """
        # For an individual, randomly select a single collaborator from each previous subpopulation
        collaborators = [random.choices(subpop, k=1)[0] for subpop in previous_subpops]
        # Assign the individual itself to the subpopulation to which it belongs
        collaborators[subpop_idx] = current_subpops[subpop_idx][indiv_idx].copy()

        return collaborators
