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
    """

    def __init__(self,
                 S_train: list,
                 S_test: list,
                 subpop_sizes: list,
                 data: DataLoader,
                 evaluator,
                 collaborator):
        """
        Parameters
        ----------
        S_train: list
           Subproblems related to the training set, where each subproblem is an numpy array that
           can be accessed by indexing the list.
        S_test: list
           Subproblems related to the test set, where each subproblem is an numpy array that can
           be accessed by indexing the list.
        subpop_sizes: list
            Subpopulation sizes, that is, the number of individuals in each subpopulation.
        data: DataLoader
            Container with process data and training and test sets.
        evaluator: object of one of the evaluation classes
            Responsible for evaluating individuals, that is, subsets of features.
        collaborator: object of one of the collaboration classes.
            Responsible for selecting collaborators for individuals.
        """
        self.S_train = S_train
        self.S_test = S_test
        self.subpop_sizes = subpop_sizes
        self.data = data
        self.evaluator = evaluator
        self.collaborator = collaborator
        # Individuals of all subpopulations
        self.subpops = list()
        # List to store the evaluations of the individuals of all subpopulations
        self.fobjs = list()
        # Number of subproblems
        self.n_subprobs = len(self.S_train)
        # Number of variables in each subproblem
        self.nvars = list(map(lambda subprob: subprob.shape[1], self.S_train))

    def build_subpopulations(self):
        """
        Randomly initializes individuals from all subpopulations.
        """
        # Initialize the progress bar
        progress_bar = tqdm(total=self.n_subprobs, desc="Building subpopulations")
        # For each subproblem with a specific number of decision variables, build a subpopulation
        for i, (nvar, subpop_size) in enumerate(zip(self.nvars, self.subpop_sizes)):
            # Initialize subpop_size individuals of size nvar with only 0's and 1's
            subpop = np.random.choice([0, 1], size=(subpop_size, nvar))
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
        progress_bar = tqdm(total=self.n_subprobs, desc="Evaluating individuals")
        # For each subpopulation
        for i, subpop in enumerate(self.subpops):
            subpop_fobjs = list()
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
                                                 X_test=self.data.X_test,
                                                 y_test=self.data.y_test)
                # Store evaluation of the current individual
                subpop_fobjs.append(metric)
            # Store evaluation of all individuals of the current subpopulation
            self.fobjs.append(subpop_fobjs)
            # Update progress bar
            progress_bar.update(1)
        # Close progress bar
        progress_bar.close()
