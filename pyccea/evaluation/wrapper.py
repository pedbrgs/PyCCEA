import copy
import logging
import warnings
import numpy as np
from ..utils.datasets import DataLoader
from ..utils.models import ClassificationModel
from ..utils.metrics import ClassificationMetrics

warnings.filterwarnings(action="ignore", category=UserWarning, message="y_pred contains classes")

class WrapperEvaluation():
    """Evaluate selected features based on the predictive performance of a machine learning model.

    Attributes
    ----------
    model_evaluator : object of one of the metrics classes
        Responsible for computing performance metrics to evaluate models.
    base_model : sklearn model object
        Model that has not been fitted. Works as a template to avoid multiple model
        initializations. As each model evaluates a subset of features (individual), the base model
        is copied and fitted for each individual.
    model : sklearn model object
        Model that has been fitted to evaluate the current individual.
    estimators : list of sklearn model objects
        Estimators used in the current evaluation. It is one when 'eval_mode' is set to "hold_out"
        and k when 'eval_mode' is set to "k_fold" or "leave_one_out".
    """

    models = {"classification": ClassificationModel}
    metrics = {"classification": ClassificationMetrics}
    eval_modes = ["hold_out", "k_fold", "leave_one_out"]

    def __init__(
            self,
            task: str,
            model_type: str,
            eval_function: str,
            eval_mode: str,
            n_classes: int = None
    ):
        """
        Parameters
        ----------
        task : str
            Name of the supervised learning task (e.g., regression, classification).
        model_type : str
            Name of the machine learning model that will be fitted for the task.
        eval_function : str
            Metric that will be used to evaluate the performance of the model trained with the
            selected subset of features (makes up the fitness of the individual).
        eval_mode : str
            Evaluation mode. It can be 'hold_out', 'leave_one_out', or 'k_fold'.
        n_classes : int, default None
            Number of classes when task parameter is set to 'classification'.
        """
        # Check if the chosen task is available
        if not task in WrapperEvaluation.metrics.keys():
            raise NotImplementedError(
                f"Task '{task}' is not implemented. "
                f"The available tasks are {', '.join(WrapperEvaluation.metrics.keys())}."
            )
        # Initialize the model evaluator
        self.model_evaluator = WrapperEvaluation.metrics[task](n_classes=n_classes)
        # Check if the chosen evaluation function is available
        if not eval_function in self.model_evaluator.metrics:
            raise NotImplementedError(
                f"Evaluation function '{eval_function}' is not implemented. "
                f"The available {task} metrics are "
                f"{', '.join(self.model_evaluator.metrics)}."
            )
        self.eval_function = eval_function
        # Initialize the model present in the wrapper model_evaluator
        self.base_model = WrapperEvaluation.models[task](model_type=model_type)
        self.model_type = model_type
        # Check if the chosen evaluation mode is available
        if not eval_mode in WrapperEvaluation.eval_modes:
            raise NotImplementedError(
                f"Evaluation mode '{eval_mode}' is not implemented. "
                f"The available evaluation modes are {', '.join(WrapperEvaluation.eval_modes)}."
            )
        self.eval_mode = eval_mode
        # Initialize logger with info level
        logging.basicConfig(encoding="utf-8", level=logging.INFO)

    def _hold_out_validation(self, solution_mask: np.ndarray, data: DataLoader) -> None:
        """Evaluate an individual using hold_out validation (train/test)."""

        # Get model that has not been previously fitted
        self.model = copy.deepcopy(self.base_model)
        # Select subset of features in the training set
        X_train = data.X_train[:, solution_mask].copy()
        y_train = data.y_train.copy()
        # Select subset of features in the test set
        X_test = data.X_test[:, solution_mask].copy()
        y_test = data.y_test.copy()
        # Train model with the current subset of features
        self.model.train(X_train=X_train, y_train=y_train, optimize=False, verbose=False)
        self.estimators.append(copy.deepcopy(self.model.estimator))
        # Evaluate the individual
        self.model_evaluator.compute(
            estimator=self.model.estimator,
            X_test=X_test,
            y_test=y_test,
            verbose=False
        )
        # Get evaluation in the test set
        self.evaluations = self.model_evaluator.values

    def _cross_validation(self, solution_mask: np.ndarray, data: DataLoader) -> None:
        """Evaluate an individual using cross-validation (leave-one-out or k-fold)."""
        for k in range(data.kfolds):
            # Get training and validations subsets built from the full training set
            X_train, y_train = data.train_folds[k]
            X_val, y_val = data.val_folds[k]
            # Select subset of features in the training subset
            X_train = X_train[:, solution_mask].copy()
            # Select subset of features in the validation subset
            X_val = X_val[:, solution_mask].copy()
            # Get model that has not been previously fitted
            self.model = copy.deepcopy(self.base_model)
            # Train model with the current subset of features
            self.model.train(X_train=X_train, y_train=y_train, optimize=False, verbose=False)
            self.estimators.append(copy.deepcopy(self.model.estimator))
            # Evaluate the individual
            self.model_evaluator.compute(
                estimator=self.model.estimator,
                X_test=X_val,
                y_test=y_val,
                verbose=False
            )
            for metric in self.evaluations.keys():
                self.evaluations[metric] += self.model_evaluator.values[metric]
            del self.model
        # Calculate average performance over k folds
        for metric in self.evaluations.keys():
            self.evaluations[metric] = round(self.evaluations[metric]/data.kfolds, 4)

    def evaluate(self, solution: np.ndarray, data: DataLoader) -> dict:
        """
        Evaluate an individual represented by a complete solution through the predictive
        performance of a machine learning model.

        Parameters
        ----------
        solution : np.ndarray
            Solution represented by a binary n-dimensional array, where n is the number of
            features.
        data : DataLoader
            Container with process data and training and test sets.

        Returns
        -------
        : dict
            Evaluation metrics.
        """
        # Estimator(s) used for the current evaluation
        self.estimators = list()
        # If no feature is selected
        self.evaluations = {metric: 0 for metric in self.model_evaluator.metrics}
        if solution.sum() == 0:
            return self.evaluations
        # Boolean array used to filter which features will be used to fit the model
        solution_mask = solution.astype(bool).copy()

        # Hold-out validation
        if self.eval_mode == "hold_out":
            self._hold_out_validation(
                solution_mask=solution_mask,
                data=data,
            )
        # K-fold cross validation or leave-one-out cross validation
        elif self.eval_mode in ["k_fold", "leave_one_out"]:
            self._cross_validation(
                solution_mask=solution_mask,
                data=data,
            )

        return self.evaluations
