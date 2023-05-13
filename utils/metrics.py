import logging
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from imblearn.metrics import specificity_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


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
        'accuracy', 
        # Ratio of the correctly identified positive cases to all the predicted positive cases,
        # i.e., tp/(tp+fp).
        'precision',
        # Also known as sensitivity, is the ratio of the correctly identified positive cases to
        # all the actual positive cases, i.e., tp/(tp+fn)
        'recall',
        # Harmonic mean of precision and recall, i.e., 2.(precision.recall)/(precision+recall)
        'f1_score',
        # Ratio of the correctly identified negative cases to all the predicted negative cases,
        # i.e., (tn)/(tn + fp)
        'specificity'
    ]

    def __init__(self):
        self.values = dict()
        # Initialize logger with info level
        logging.basicConfig(encoding='utf-8', level=logging.INFO)

    def compute(self,
                estimator,
                eval_mode: str,
                kfolds: int,
                X: np.ndarray,
                y: np.ndarray,
                verbose: bool=False):
        """
        Parameters
        ----------
        estimator: sklearn model object
            Model that will be evaluated.
        eval_mode: str
            Evaluation mode. It can be 'train_val' or 'kfold_cv'.
        kfolds: int, default 10
            Number of folds in the k-fold cross validation.
        X: np.ndarray
            Input data. It is from the validation set in 'train_val' mode and from the training
            set in 'kfold_cv' mode.
        y: np.ndarray
            Output data. It is from the validation set in 'train_val' mode and from the training
            set in 'kfold_cv' mode.
        verbose: bool, default False
            If True, show evaluation metrics.
        """

        # Type of aggregation used in the evaluation metrics according to the classification task
        avg = 'macro' if np.unique(y).shape[0] > 2 else 'binary'

        # Train-validation
        if eval_mode == 'train_val':
            # Predictions
            y_pred = estimator.predict(X)
            # Measures
            self.values['precision'] = round(precision_score(y, y_pred, average=avg), 4)
            self.values['recall'] = round(recall_score(y, y_pred, average=avg), 4)
            self.values['f1_score'] = round(f1_score(y, y_pred, average=avg), 4)
            self.values['accuracy'] = round(accuracy_score(y, y_pred), 4)
            self.values['specificity'] = round(specificity_score(y, y_pred, average=avg), 4)
        # Cross-validation
        else:
            scoring = {'precision': make_scorer(precision_score, average=avg),
                       'recall': make_scorer(recall_score, average=avg),
                       'f1_score': make_scorer(f1_score, average=avg),
                       'accuracy': make_scorer(accuracy_score),
                       'specificity': make_scorer(specificity_score, average=avg)}
            evaluation = cross_validate(estimator, X, y, scoring=scoring, cv=kfolds)
            # Measures
            self.values = {
                f'{metric}': round(evaluation[f'test_{metric}'].mean(), 4)
                for metric in scoring.keys()
            }

        # Show evaluation metrics
        if verbose:
            logging.getLogger().disabled = False
            logging.info(f"Precision: {self.values['precision']}")
            logging.info(f"Accuracy: {self.values['accuracy']}")
            logging.info(f"Recall/Sensitivity/TPR: {self.values['recall']}")
            logging.info(f"Specificity/TNR: {self.values['specificity']}")
            logging.info(f"F1-score: {self.values['f1_score']}")
