import numpy as np
from abc import ABC
from utils.datasets import DataLoader
from evaluation.wrapper import WrapperEvaluation


class WrapperFitnessFunction(ABC):
    """ An abstract class for a wrapper objective function.

    It measures the quality of a solution according to the predictive performance of a machine
    learning model.

    Attributes
    ----------
    evaluator: object of one of the evaluation classes
        Responsible for evaluating individuals, that is, subsets of features.   
    """

    def __init__(self, evaluator: WrapperEvaluation):
        self.evaluator = evaluator

    def _evaluate_predictive_performance(self,
                                         context_vector: np.ndarray,
                                         data: DataLoader):
        """""
        Evaluate predictive performance of a machine learning model trained on the selected set.

        Each context vector must be represented by a binary n-dimensional array, where n is the
        number of features. If there is a 1 in the i-th position of the array, it indicates that
        the i-th feature should be considered and if there is a 0, it indicates that the feature
        should not be considered.
        """
        return self.evaluator.evaluate(solution=context_vector.copy(), data=data)
