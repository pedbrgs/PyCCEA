import logging
import numpy as np
from sklearn.metrics import (precision_score, recall_score, accuracy_score, f1_score,
                             confusion_matrix)


class ClassificationMetrics():
    """
    Evaluate machine learning model trained for a classification problem.

    Attributes
    ----------
    values: dict()
        Values of classification metrics.
    """

    metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'specificity']

    def __init__(self):
        self.values = dict()
        # Initialize logger with info level
        logging.basicConfig(encoding='utf-8', level=logging.INFO)

    def compute(self, y_pred: np.ndarray, y_test: np.ndarray, verbose: bool=False):
        """
        Parameters
        ----------
        y_pred: np.ndarray
            Output predicted by the machine learning model in the test set.
        y_test: np.ndarray
            Test output data.
        verbose: bool, default False
            If True, show evaluation metrics in the test set.
        """
        # Type of aggregation used in the evaluation metrics according to the classification task
        avg = 'macro' if np.unique(y_test).shape[0] > 2 else 'binary'
        
        # Square matrix (CxC), where C is the number of classes, containing the number of true
        # positives (tp), number of false positives (fp), number of true negatives (tn) and
        # number of false negatives (fn)
        self.values['conf_matrix'] = confusion_matrix(y_test, y_pred)
        # Ratio of the correctly identified positive cases to all the predicted positive cases,
        # i.e., tp/(tp+fp).
        self.values['precision'] = round(precision_score(y_test, y_pred, average=avg), 4)
        # Also known as sensitivity, is the ratio of the correctly identified positive cases to
        # all the actual positive cases, i.e., tp/(tp+fn)
        self.values['recall'] = round(recall_score(y_test, y_pred, average=avg), 4)
        # Harmonic mean of precision and recall, i.e., 2.(precision.recall)/(precision+recall)
        self.values['f1_score'] = round(f1_score(y_test, y_pred, average=avg), 4)
        # Ratio of number of correct predictions to the total number of input samples, i.e.,
        # (tp+tn)/(tp+fp+tn+fn)
        self.values['accuracy'] = round(accuracy_score(y_test, y_pred), 4)
        # Ratio of the correctly identified negative cases to all the predicted negative cases,
        # i.e., (tn)/(tn + fp)
        tn, fp, fn, tp = self.values['conf_matrix'].ravel()
        self.values['specificity'] = round(tn / (tn + fp), 4)

        # Show evaluation metrics
        if verbose:
            logging.info(f"Precision: {self.values['precision']}")
            logging.info(f"Recall/Sensitivity/TPR: {self.values['recall']}")
            logging.info(f"F1-score: {self.values['f1_score']}")
            logging.info(f"Accuracy: {self.values['accuracy']}")
            logging.info(f"Specificity/TNR: {self.values['specificity']}")
