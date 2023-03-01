import random
import numpy as np


class SingleRandomCollaboration():
    """
    Randomly find a single collaborator from each subpopulation for individuals.

    Attributes
    ----------
    seed: int
        Numerical value that generates a new set or repeats pseudo-random numbers. It is defined
        in stochastic processes to ensure reproducibility.
    """

    def __init__(self, seed: int = None):

        # Set the seed value
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed=seed)

    def get_collaborators(self, subpop_idx: int, indiv_idx: int, subpops: list):
        """
        Randomly find a single collaborator from each subpopulation for the individual given as a
        parameter.

        Parameters
        ----------
        subpop_idx: int
            Index of the subpopulation to which the individual belongs.
        indiv_idx: int
            Index of the individual in its respective subpopulation.
        subpops: list
            Individuals from all subpopulations.

        Returns
        -------
        collaborators: list
            A single individual from each subpopulation that will collaborate with the individual
            given as a parameter.
        """
        # Number of subpopulations
        n_subpops = len(subpops)
        # Number of individuals in each subpopulation
        subpop_sizes = list(map(lambda subpop: subpop.shape[0], subpops))
        # For an individual, randomly select a single collaborator from each subpopulation
        collaborators = [random.choices(subpop, k=1)[0] for subpop in subpops]
        # Assign the individual itself to the subpopulation to which it belongs
        collaborators[subpop_idx] = subpops[subpop_idx][indiv_idx]

        return collaborators

    def build_complete_solution(self, collaborators):
        """
        Build a complete solution to the problem using an individual selected from each
        subpopulation.

        Parameters
        ----------
        collaborators: list
            A single individual from each subpopulation that will collaborate with an individual.

        Returns
        -------
        complete_solution: np.array
            Complete problem solution composed of one individual from each subpopulation.
        """
        complete_solution = np.hstack(collaborators)

        return complete_solution
