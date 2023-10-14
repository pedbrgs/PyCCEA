import copy
import logging
import numpy as np
from utils.datasets import DataLoader
from utils.models import ClassificationModel
from utils.metrics import ClassificationMetrics


class WrapperEvaluation():
    """
    Evaluate selected features based on the predictive performance of a machine learning model.

    Attributes
    ----------
    model_evaluator: object of one of the metrics classes
        Responsible for computing performance metrics to evaluate models.
    base_model: sklearn model object
        Model that has not been fitted. Works as a template to avoid multiple model
        initializations. As each model evaluates a subset of features (individual), the base model
        is copied and fitted for each individual.
    model: sklearn model object
        Model that has been fitted to evaluate the individual.
    """

    models = {"classification": ClassificationModel}
    metrics = {"classification": ClassificationMetrics}
    eval_modes = ["train_val", "kfold_cv"]

    def __init__(self,
                 task: str,
                 model_type: str,
                 eval_function: str,
                 eval_mode: str):
        """
        Parameters
        ----------
        task: str
            Name of the supervised learning task (e.g., regression, classification).
        model_type: str
            Name of the machine learning model that will be fitted for the task.
        eval_function: str
            Metric that will be used to evaluate the performance of the model trained with the
            selected subset of features.
        eval_mode: str
            Evaluation mode. It can be 'train_val' or 'kfold_cv'.
        """
        # Check if the chosen task is available
        if not task in WrapperEvaluation.metrics.keys():
            raise AssertionError(
                f"Task '{task}' was not found. "
                f"The available tasks are {', '.join(WrapperEvaluation.metrics.keys())}."
            )
        # Initialize the model evaluator
        self.model_evaluator = WrapperEvaluation.metrics[task]()
        # Check if the chosen evaluation function is available
        if not eval_function in self.model_evaluator.metrics:
            raise AssertionError(
                f"Evaluation function '{eval_function}' was not found. "
                f"The available {task} metrics are "
                f"{', '.join(self.model_evaluator.metrics)}."
            )
        self.eval_function = eval_function
        # Initialize the model present in the wrapper model_evaluator
        self.base_model = WrapperEvaluation.models[task](model_type=model_type)
        self.model_type = model_type
        # Check if the chosen evaluation mode is available
        if not eval_mode in WrapperEvaluation.eval_modes:
            raise AssertionError(
                f"Evaluation mode '{eval_mode}' was not found. "
                f"The available evaluation modes are {', '.join(WrapperEvaluation.eval_modes)}."
            )
        self.eval_mode = eval_mode
        # Initialize logger with info level
        logging.basicConfig(encoding="utf-8", level=logging.INFO)

    def _hold_out_validation(self,
                             solution_mask: np.ndarray,
                             data: DataLoader,
                             test_mode: bool=False,
                             return_train_eval: bool=False):
        """Evaluate an individual using hold-out validation (train/validation/test)."""

        # Get model that has not been previously fitted
        self.model = copy.deepcopy(self.base_model)
        # Select subset of features in the training set
        X_train = data.X_train[:, solution_mask].copy()
        # Select subset of features in the validation or test set
        if test_mode:
            X_val = data.X_test[:, solution_mask].copy()
            y_val = data.y_test.copy()
        else:
            X_val = data.X_val[:, solution_mask].copy()
            y_val = data.y_val.copy()
        # Train model with the current subset of features
        self.model.train(X_train=X_train,
                         y_train=data.y_train,
                         optimize=False,
                         verbose=False)
        # Evaluate the individual
        self.model_evaluator.compute(estimator=self.model.estimator,
                                     X_test=X_val,
                                     y_test=y_val,
                                     verbose=False)
        # Get evaluation in the validation or test set
        self.evaluations = self.model_evaluator.values

        # Get evaluation in the training set
        if return_train_eval:
            self.model_evaluator.compute(estimator=self.model.estimator,
                                         X_test=X_train,
                                         y_test=data.y_train,
                                         verbose=False)
            self.train_evaluations = self.model_evaluator.values

    def _kfold_cross_validation(self,
                                solution_mask: np.ndarray,
                                data: DataLoader,
                                test_mode: bool=False,
                                return_train_eval: bool=False):
        """Evaluate an individual using k-fold cross-validation."""

        for k in range(data.kfolds):
            if test_mode:
                X_train, y_train = data.eval_train_folds[k]
                X_val, y_val = data.eval_val_folds[k]
            else:
                X_train, y_train = data.train_folds[k]
                X_val, y_val = data.val_folds[k]
            # Select subset of features in the training set
            X_train = X_train[:, solution_mask].copy()
            # Select subset of features in the validation set
            X_val = X_val[:, solution_mask].copy()
            # Get model that has not been previously fitted
            self.model = copy.deepcopy(self.base_model)
            # Train model with the current subset of features
            self.model.train(X_train=X_train, y_train=y_train, optimize=False, verbose=False)
            # Evaluate the individual
            self.model_evaluator.compute(estimator=self.model.estimator,
                                         X_test=X_val,
                                         y_test=y_val,
                                         verbose=False)
            for metric in self.evaluations.keys():
                self.evaluations[metric] += self.model_evaluator.values[metric]
            # Get evaluation in the training set
            if return_train_eval:
                self.model_evaluator.compute(estimator=self.model.estimator,
                                             X_test=X_train,
                                             y_test=y_train,
                                             verbose=False)
                for metric in self.evaluations.keys():
                    self.train_evaluations[metric] += self.model_evaluator.values[metric]
            del self.model
        # Calculate average performance over k folds
        for metric in self.evaluations.keys():
            self.evaluations[metric] = round(self.evaluations[metric]/data.kfolds, 4)
            if return_train_eval:
                self.train_evaluations[metric] = round(self.train_evaluations[metric]/data.kfolds, 4)

    def evaluate(self,
                 solution: np.ndarray,
                 data: DataLoader,
                 test_mode: bool=False,
                 return_gap: bool=False):
        """
        Evaluate an individual represented by a complete solution through the predictive
        performance of a model.

        Parameters
        ----------
        solution: np.ndarray
            Solution represented by a binary n-dimensional array, where n is the number of
            features.
        data: DataLoader
            Container with process data and training and test sets.
        test_mode: bool, default False
            In test mode, the training will be performed using the training set and the evaluation
            using the test set (when eval_mode is 'train_val') or the training will be performed
            using training folds built from the testing set and the evaluation using validation
            folds built from the testing set (when eval_mode is 'kfold_cv'). Otherwise, the
            training will be performed using the training set and the evaluation using the 
            validation set (when eval_mode is 'train_val') or the training will be performed
            using training folds built from the training set and the evaluation using validation
            folds built from the training set (when eval_mode is 'kfold_cv').
        return_gap: bool, default False
            If True, it also calculates evaluation metrics on the training set and computes the
            generalization gap, which is the difference between the evaluation metrics on the
            validation/test set (depending on the value of 'test_mode') and the training set.

        Returns
        -------
        float
            Evaluation metrics.
        """
        # If no feature is selected
        self.evaluations = {metric: 0 for metric in self.model_evaluator.metrics}
        if return_gap:
            self.train_evaluations = {metric: 0 for metric in self.model_evaluator.metrics}
        if solution.sum() == 0:
            return 0
        # Boolean array used to filter which features will be used to fit the model
        solution_mask = solution.astype(bool)
        # Train-validation
        if self.eval_mode == "train_val":
            self._hold_out_validation(
                solution_mask=solution_mask,
                data=data,
                test_mode=test_mode,
                return_train_eval=return_gap
            )
        # K-fold cross-validation
        else:
            self._kfold_cross_validation(
                solution_mask=solution_mask,
                data=data,
                test_mode=test_mode,
                return_train_eval=return_gap
            )
        if return_gap:
            self.evaluations["generalization_gap"] = abs(
                self.train_evaluations[self.eval_function] - self.evaluations[self.eval_function]
            )

        return self.evaluations
