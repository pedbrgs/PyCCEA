import numpy as np
from ..initialization.build import SubpopulationInitialization


class RandomBinaryInitialization(SubpopulationInitialization):
    """Randomly initialize subpopulations with binary representation."""

    def _get_subpop(self, subcomp_size, subpop_size) -> np.ndarray:
        """Initialize 'subpop_size' individuals of size 'subcomp_size' with only 0's and 1's.

        Parameters
        ----------
        subcomp_size : int
            Number of individuals in the subpopulation.
        subpop_size : int
            Size of each individual in the subpopulation.

        Returns
        -------
        subpop : np.ndarray
            A subpopulation. Each individual is represented by a binary n-dimensional array, where
            n is the number of features in the specific subcomponent. If there is a 1 in the i-th
            position of the array, it indicates that the i-th feature should be considered and if
            there is a 0, it indicates that the feature should not be considered.
        """
        subpop = np.random.randint(2, size=(subpop_size, subcomp_size))
        return subpop

    def _build_context_vector(
            self,
            subpop_idx: int,
            indiv_idx: int,
            subpops: list,
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
        collaborators = self.collaborator.get_collaborators(
            subpop_idx=subpop_idx,
            indiv_idx=indiv_idx,
            # As it is the first generation, individuals will be used as collaborators with each
            # other for evaluation
            previous_subpops=subpops,
            current_subpops=subpops,
        )
        # Build a context vector to evaluate a complete solution
        context_vector = self.collaborator.build_context_vector(collaborators)
        return context_vector
