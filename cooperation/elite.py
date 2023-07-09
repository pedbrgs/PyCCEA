import random
import numpy as np
from cooperation.collaboration import Collaboration


class SingleEliteCollaboration(Collaboration):
    """
    Randomly selects the collaborator from among the best k individuals in the subpopulation.
    """

    def __init__(self, sample_size: int, seed: int = None):
        """
        Parameters
        ----------
        sample_size: int
            Number of best individuals to compose the subpopulation sample.
        seed: int
            Numerical value that generates a new set or repeats pseudo-random numbers. It is
            defined in stochastic processes to ensure reproducibility.
        """
        self.sample_size = sample_size
        # Set the seed value
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed=seed)

    def get_collaborators(self,
                          subpop_idx: int,
                          indiv_idx: int,
                          subpops: list,
                          ranking: list):
        """
        Set the collaborator of the individual given as a parameter as a random individual from
        the k best individuals of the subpopulation.

        Parameters
        ----------
        subpop_idx: int
            Index of the subpopulation to which the individual belongs.
        indiv_idx: int
            Index of the individual in its respective subpopulation.
        subpops: list
            Individuals from all subpopulations.
        ranking: list
            Ranking of subpopulations.

        Returns
        -------
        collaborators: list
            A single individual from each subpopulation that will collaborate with the individual
            given as a parameter.
        """
        collaborators = [None] * len(subpops)
        # Find a collaborator per subpopulation
        for i in range(len(subpops)):
            # If the current subpopulation is the same as the individual's subpopulation, the
            # individual's collaborator is himself
            if i == subpop_idx:
                collaborators[i] = subpops[subpop_idx][indiv_idx]
            # Otherwise, the collaborator will be a random individual among the 'sample_size' best
            # individuals of the subpopulation
            else:
                collaborator_pool = subpops[i][ranking[i][:self.sample_size]]
                collaborators[i] = random.choices(collaborator_pool, k=1)[0]

        return collaborators
