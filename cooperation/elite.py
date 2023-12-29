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
                          next_subpops: list,
                          fitness: list):
        """
        Set the collaborators of the individual given as a parameter as random individuals among
        the k best individuals in each subpopulation.

        In the scenario of n subpopulations, the elite collaboration method involves choosing n
        collaborators, with one selected from each subpopulation. Notably, the individual
        becomes their own collaborator within their specific subpopulation, while the remaining
        collaborators are chosen from the top k individuals in each respective subpopulation.

        Parameters
        ----------
        subpop_idx: int
            Index of the subpopulation to which the individual belongs.
        indiv_idx: int
            Index of the individual in its respective subpopulation. The vector that represents
            the individual is obtained from the `next_subpops`, where the individuals were evolved
            and not evaluated.
        subpops: list
            Individuals from all subpopulations.
        next_subpops: list
            Individuals from all subpopulations of the next generation.
        fitness: list
            Evaluation of the individuals in all subpopulations.

        Returns
        -------
        collaborators: list
            A single individual from each subpopulation that will collaborate with the individual
            given as a parameter.
        """
        collaborators = [None] * len(subpops)
        # Find a collaborator per subpopulation
        for i in range(len(subpops)):
            # If the i-th subpopulation is the same as the individual's subpopulation, the
            # individual's collaborator is itself, since we want to evaluate it
            if i == subpop_idx:
                collaborators[i] = next_subpops[subpop_idx][indiv_idx].copy()
            # Otherwise, the collaborator will be a random individual among the 'sample_size' best
            # individuals of the subpopulation in the previous generation
            else:
                ranking = np.argsort(fitness[i])[::-1]
                collaborator_pool = subpops[i][ranking][:self.sample_size].copy()
                collaborators[i] = random.choices(collaborator_pool, k=1)[0].copy()

        return collaborators
