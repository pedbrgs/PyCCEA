import numpy as np
from tqdm import tqdm
from utils.datasets import DataLoader


class RandomBinaryInitialization():
    """
    Randomly initialize subpopulations with a binary representation.

    Attributes
    ----------
    subpops: list
        Individuals from all subpopulations. Each individual is represented by a binary
        n-dimensional array, where n is the number of features. If there is a 1 in the i-th
        position of the array, it indicates that the i-th feature should be considered and if
        there is a 0, it indicates that the feature should not be considered.
    fobjs: list
        Evaluation of all individuals from all subpopulations.
    context_vectors: list
        Complete problem solutions that were randomly initialized.
    """

    def __init__(self,
                 data: DataLoader,
                 subcomp_sizes: list,
                 subpop_sizes: list,
                 evaluator,
                 collaborator):
        """
        Parameters
        ----------
        data: DataLoader
            Container with process data and training and test sets.
        subcomp_sizes: list
            Number of features in each subcomponent.
        subpop_sizes: list
            Subpopulation sizes, that is, the number of individuals in each subpopulation.
        evaluator: object of one of the evaluation classes
            Responsible for evaluating individuals, that is, subsets of features.
        collaborator: object of one of the collaboration classes.
            Responsible for selecting collaborators for individuals.
        """
        # Parameters as attributes
        self.data = data
        self.subpop_sizes = subpop_sizes
        self.evaluator = evaluator
        self.collaborator = collaborator
        # Complete problem solutions
        self.context_vectors = list()
        # Best individual of each subpopulation
        self.best_individuals = list()
        # Individuals of all subpopulations
        self.subpops = list()
        # List to store the evaluations of the individuals of all subpopulations
        self.fobjs = list()
        # Number of subcomponents
        self.n_subcomps = len(subcomp_sizes)
        # Number of features in each subcomponent
        self.subcomp_sizes = subcomp_sizes

    def build_subpopulations(self):
        """
        Randomly initializes individuals from all subpopulations.
        """
        # Initialize the progress bar
        progress_bar = tqdm(total=self.n_subcomps, desc="Building subpopulations")
        # For each subcomponent with a specific number of features, build a subpopulation
        for subcomp_size, subpop_size in zip(self.subcomp_sizes, self.subpop_sizes):
            # Initialize subpop_size individuals of size subcomp_size with only 0's and 1's
            subpop = np.random.choice([0, 1], size=(subpop_size, subcomp_size))
            # Store all individuals of the current subpopulation
            self.subpops.append(subpop)
            # Update progress bar
            progress_bar.update(1)
        # Close progress bar
        progress_bar.close()

    def evaluate_individuals(self):
        """
        Evaluate all individuals from all subpopulations.
        """
        # Initialize the progress bar
        progress_bar = tqdm(total=self.n_subcomps, desc="Evaluating individuals")
        # For each subpopulation
        for i, subpop in enumerate(self.subpops):
            subpop_fobjs = list()
            subpop_context_vectors = list()
            # Evaluate each individual in the subpopulation
            for j, indiv in enumerate(subpop):
                # Find random collaborator(s) for the current individual
                collaborators = self.collaborator.get_collaborators(subpop_idx=i,
                                                                    indiv_idx=j,
                                                                    subpops=self.subpops)
                # Build a complete solution to evaluate the individual
                complete_solution = self.collaborator.build_complete_solution(collaborators)
                # If no feature is selected
                if complete_solution.sum() == 0:
                    # Select one at random
                    pos = np.random.choice(np.arange(complete_solution.shape[0]))
                    complete_solution[pos] = 1
                # Evaluate the current individual
                metric = self.evaluator.evaluate(solution=complete_solution,
                                                 X_train=self.data.X_train,
                                                 y_train=self.data.y_train,
                                                 X_test=self.data.X_val,
                                                 y_test=self.data.y_val)
                # Store the complete problem solution related to the current individual
                subpop_context_vectors.append(complete_solution)
                # Store evaluation of the current individual
                subpop_fobjs.append(metric)
            # Store all complete problem solutions related to the current subpopulation
            self.context_vectors.append(np.vstack(subpop_context_vectors))
            # Store evaluation of all individuals of the current subpopulation
            self.fobjs.append(subpop_fobjs)
            # Update progress bar
            progress_bar.update(1)
        # Close progress bar
        progress_bar.close()
