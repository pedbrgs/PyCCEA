import logging
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from imblearn.metrics import specificity_score
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
)


class ClassificationMetrics():
    """
    Evaluate machine learning model trained for a classification problem.

    Attributes
    ----------
    values: dict()
        Values of classification metrics.
    """

    metrics = [
        # Ratio of number of correct predictions to the total number of input samples, i.e.,
        # (tp+tn)/(tp+fp+tn+fn)
        "accuracy",
        # It is equivalent to accuracy with class-balanced sample weights
        "balanced_accuracy",
        # Ratio of the correctly identified positive cases to all the predicted positive cases,
        # i.e., tp/(tp+fp).
        "precision",
        # Also known as sensitivity, is the ratio of the correctly identified positive cases to
        # all the actual positive cases, i.e., tp/(tp+fn)
        "recall",
        # Harmonic mean of precision and recall, i.e., 2.(precision.recall)/(precision+recall)
        "f1_score",
        # Ratio of the correctly identified negative cases to all the predicted negative cases,
        # i.e., (tn)/(tn + fp)
        "specificity"
    ]

    def __init__(self, n_classes):
        """
        Parameters
        ----------
        n_classes: int
            Number of classes.
        """
        self.values = dict()
        self.n_classes = n_classes
        # Initialize logger with info level
        logging.basicConfig(encoding="utf-8", level=logging.INFO)

    def compute(self,
                estimator,
                X_test: np.ndarray,
                y_test: np.ndarray,
                verbose: bool=False):
        """
        Parameters
        ----------
        estimator: sklearn model object
            Model that will be evaluated.
        X_test: np.ndarray
            Input test data.
        y_test: np.ndarray
            Output test data.
        verbose: bool, default False
            If True, show evaluation metrics.
        """

        # Type of aggregation used in the evaluation metrics according to the classification task
        avg = "macro" if self.n_classes > 2 else "binary"

        # Predictions
        y_pred = estimator.predict(X_test)
        # Measures
        self.values["precision"] = round(precision_score(y_test, y_pred, average=avg), 4)
        self.values["recall"] = round(recall_score(y_test, y_pred, average=avg), 4)
        self.values["f1_score"] = round(f1_score(y_test, y_pred, average=avg), 4)
        self.values["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
        self.values["balanced_accuracy"] = round(balanced_accuracy_score(y_test, y_pred), 4)
        self.values["specificity"] = round(specificity_score(y_test, y_pred, average=avg), 4)

        # Show evaluation metrics
        if verbose:
            logging.getLogger().disabled = False
            logging.info(f"Precision: {self.values['precision']}")
            logging.info(f"Balanced accuracy: {self.values['balanced_accuracy']}")
            logging.info(f"Accuracy: {self.values['accuracy']}")
            logging.info(f"Recall/Sensitivity/TPR: {self.values['recall']}")
            logging.info(f"Specificity/TNR: {self.values['specificity']}")
            logging.info(f"F1-score: {self.values['f1_score']}")
