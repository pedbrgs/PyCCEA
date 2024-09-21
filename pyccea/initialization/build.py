import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from ..utils.datasets import DataLoader


class SubpopulationInitialization(ABC):
    """An abstract class for subpopulation initialization.

    Attributes
    ----------
    subpops : list
        Individuals from all subpopulations.
    fitness : list
        Evaluation of all context vectors from all subpopulations.
    context_vectors: list
        Complete problem solutions that were randomly initialized.
    """

    def __init__(
            self,
            data: DataLoader,
            subcomp_sizes: list,
            subpop_sizes: list,
            collaborator,
            fitness_function
    ):
        """
        Parameters
        ----------
        data : DataLoader
            Container with processed data and training and test sets.
        subcomp_sizes : list
            Number of features in each subcomponent.
        subpop_sizes : list
            Subpopulation sizes, that is, the number of individuals in each subpopulation.
        collaborator : object of one of the collaboration classes.
            Responsible for selecting collaborators for individuals.
        fitness_function : object of one of the fitness classes.
            Responsible for evaluating individuals, that is, subsets of features.
        """
        self.data = data
        self.subpop_sizes = subpop_sizes
        self.fitness_function = fitness_function
        self.collaborator = collaborator
        # Complete problem solutions
        self.context_vectors = list()
        # Individuals of all subpopulations
        self.subpops = list()
        # List to store the fitness of all context vectors
        self.fitness = list()
        # Number of subcomponents
        self.n_subcomps = len(subcomp_sizes)
        # Number of features in each subcomponent
        self.subcomp_sizes = subcomp_sizes

    @abstractmethod
    def _get_subpop(self, subcomp_size, subpop_size) -> np.ndarray:
        """Get a single subpopulation according to the domain of the search space and their
        respective boundaries.

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
        pass

    @abstractmethod
    def _build_context_vector(self, subpop_idx: int, indiv_idx: int, subpops: np.ndarray) -> np.ndarray:
        """Build a complete solution from an individual and their collaborators.

        Parameters
        ----------
        subpop_idx : int
            Index of the subpopulation to which the individual belongs.
        indiv_idx : int
            Index of the individual in its respective subpopulation.
        subpops : np.ndarray
            Subpopulations.

        Returns
        -------
        context_vector : np.ndarray
            Complete solution.
        """
        pass        

    def build_subpopulations(self):
        """Initialize individuals from all subpopulations."""
        # Initialize the progress bar
        progress_bar = tqdm(total=self.n_subcomps, desc="Building subpopulations")
        # For each subcomponent with a specific number of features, build a subpopulation
        for subcomp_size, subpop_size in zip(self.subcomp_sizes, self.subpop_sizes):
            # Initialize subpop_size individuals of size subcomp_size
            subpop = self._get_subpop(subcomp_size, subpop_size)
            # Store all individuals of the current subpopulation
            self.subpops.append(subpop)
            # Update progress bar
            progress_bar.update(1)
        # Close progress bar
        progress_bar.close()

    def evaluate_individuals(self):
        """Evaluate all individuals from all subpopulations."""
        # Initialize the progress bar
        progress_bar = tqdm(total=self.n_subcomps, desc="Evaluating individuals")
        # For each subpopulation
        for i, subpop in enumerate(self.subpops):
            # List to store the context vectors in the current subpopulation
            subpop_context_vectors = list()
            # List to store the evaluations of these context vectors
            subpop_fitness = list()
            # Evaluate each individual in the subpopulation
            for j, _ in enumerate(subpop):
                # Build a context vector to evaluate a complete solution
                context_vector = self._build_context_vector(
                    subpop_idx=i,
                    indiv_idx=j,
                    subpops=self.subpops
                )
                # Evaluate the context vector
                fitness = self.fitness_function.evaluate(context_vector, self.data)
                # Store the complete problem solution related to the current individual
                subpop_context_vectors.append(context_vector.copy())
                # Store evaluation of the current context vector
                subpop_fitness.append(fitness)
            # Store all complete problem solutions related to the current subpopulation
            self.context_vectors.append(np.vstack(subpop_context_vectors))
            # Store evaluation of all context vectors of the current subpopulation
            self.fitness.append(subpop_fitness)
            # Update progress bar
            progress_bar.update(1)
            # Delete variables related to the current subpopulation
            del subpop_context_vectors, subpop_fitness
        # Close progress bar
        progress_bar.close()
