import numpy as np
from utils.datasets import DataLoader
from utils.mapping import shifted_heaviside_function
from initialization.build import SubpopulationInitialization


class RandomContinuousInitialization(SubpopulationInitialization):
    """Randomly initialize subpopulations with continuous representation.

    For certain Evolutionary Algorithms, like Differential Evolution, which operate on continuous
    solutions, an appropriate initialization is required based on this representation.
    """

    def __init__(
            self,
            data: DataLoader,
            subcomp_sizes: list,
            subpop_sizes: list,
            collaborator,
            fitness_function,
            bounds: tuple = (0, 1)
    ):
        super().__init__(data, subcomp_sizes, subpop_sizes, collaborator, fitness_function)
        """
        Parameters
        ----------
        bounds : tuple[int, int], default (0, 1)
            Bounds of continuous variables.
        """
        self.bounds = bounds

    def _get_subpop(self, subcomp_size, subpop_size) -> np.ndarray:
        """Initialize 'subpop_size' individuals with 'subcomp_size' continuous values between
        bounds.

        Parameters
        ----------
        subcomp_size : int
            Number of individuals in the subpopulation.
        subpop_size : int
            Size of each individual in the subpopulation.

        Returns
        -------
        subpop : np.ndarray
            A subpopulation.
        """
        subpop = np.random.uniform(
                low=self.bounds[0],
                high=self.bounds[1],
                size=(subpop_size, subcomp_size)
        )
        return subpop

    def _build_context_vector(
            self,
            subpop_idx: int,
            indiv_idx: int,
            subpops: list
        ) -> np.ndarray:
        """Build a complete solution from an individual and their collaborators.

        Parameters
        ----------
        subpop_idx : int
            Index of the subpopulation to which the individual belongs.
        indiv_idx : int
            Index of the individual in its respective subpopulation.
        subpops : list
            Individuals from all subpopulations of the first generation (will be evaluated).

        Returns
        -------
        context_vector : np.ndarray
            Complete solution.
        """
        # Find collaborators for the individual
        encoded_collaborators = self.collaborator.get_collaborators(
            subpop_idx=subpop_idx,
            indiv_idx=indiv_idx,
            # As it is the first generation, individuals will be used as collaborators with each
            # other for evaluation
            previous_subpops=subpops,
            current_subpops=subpops,
        )
        # Collaborators in the continuous space is transformed into the binary space
        collaborators = [
            shifted_heaviside_function(collaborator)
            for collaborator in encoded_collaborators
        ]
        # Build a context vector to evaluate a complete solution
        context_vector = self.collaborator.build_context_vector(collaborators)
        return context_vector
