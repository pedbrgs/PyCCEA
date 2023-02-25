import numpy as np
from tqdm import tqdm
from utils.datasets import DataLoader
from evaluation.wrapper import WrapperEvaluation

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
                 evaluator):
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
        """
        self.S_train = S_train
        self.S_test = S_test
        self.subpop_sizes = subpop_sizes
        self.data = data
        self.evaluator = evaluator
        self.subpops = list()
        self.fobjs = list()

    def run(self):
        """
        Randomly initializes individuals from all subpopulations.
        """
        # Number of subproblems
        n_subprobs = len(self.S_train)
        # Initialize the progress bar
        progress_bar = tqdm(total=n_subprobs, desc="Building subpopulations")
        # Number of variables in each subproblem
        nvars = list(map(lambda subprob: subprob.shape[1], self.S_train))
        # For each subproblem with a specific number of decision variables, build a subpopulation
        for i, (nvar, subpop_size) in enumerate(zip(nvars, self.subpop_sizes)):
            # Initialize subpop_size individuals of size nvar with only 0's and 1's
            subpop = np.random.choice([0, 1], size=(subpop_size, nvar))
            # Evaluate each individual in the subpopulation
            current_fobjs = list()
            for j, indiv in enumerate(subpop):
                # If no feature is selected
                if indiv.sum() == 0:
                    # Select one at random
                    pos = np.random.choice(np.arange(indiv.shape[0]))
                    indiv[pos] = 1
                    subpop[j] = indiv.copy()
                # Evaluate the current individual
                metric = self.evaluator.evaluate(indiv=indiv,
                                                 X_train=self.S_train[i],
                                                 y_train=self.data.y_train,
                                                 X_test=self.S_test[i],
                                                 y_test=self.data.y_test)
                # Store evaluation of the current individual
                current_fobjs.append(metric)
            # Store all individuals of the current subpopulation
            self.subpops.append(subpop)
            # Store evaluation of all individuals of the current subpopulation
            self.fobjs.append(current_fobjs)
            # Update progress bar
            progress_bar.update(1)
        # Close progress bar
        progress_bar.close()
