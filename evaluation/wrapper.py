import copy
import logging
import numpy as np
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

    models = {'classification': ClassificationModel}
    metrics = {'classification': ClassificationMetrics}
    eval_modes = ['train_val', 'kfold_cv']

    def __init__(self,
                 task: str,
                 model_type: str,
                 eval_function: str,
                 eval_mode: str,
                 kfolds: int = 10):
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
        kfolds: int, default 10
            Number of folds in the k-fold cross validation.
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
        self.kfolds = kfolds
        # Initialize logger with info level
        logging.basicConfig(encoding='utf-8', level=logging.INFO)

    def evaluate(self,
                 solution: np.ndarray,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray):
        """
        Evaluate an individual represented by a complete solution through the predictive
        performance of a model and according to an evaluation metric.

        Parameters
        ----------
        solution: np.ndarray
            Solution represented by a binary n-dimensional array, where n is the number of
            features.
        X_train: np.ndarray
            Train input data.
        X_val: np.ndarray
            Validation input data.
        y_train: np.ndarray
            Train output data.
        y_val: np.ndarray
            Validation output data.
        """
        # If no feature is selected
        if solution.sum() == 0:
            return 0
        # Get model that has not been previously fitted
        self.model = copy.deepcopy(self.base_model)
        # Boolean array used to filter which features will be used to fit the model
        solution_mask = solution.astype(bool)
        # Select subset of features in the training set
        X_train = X_train[:, solution_mask].copy()
        # Train-validation
        if self.eval_mode == 'train_val':
            # Select subset of features in the validation set
            X_val = X_val[:, solution_mask].copy()
            # Train model with the current subset of features
            self.model.train(X_train=X_train, y_train=y_train, optimize=False, verbose=False)
            # Evaluate the individual
            self.model_evaluator.compute(estimator=self.model.estimator,
                                         eval_mode=self.eval_mode,
                                         X=X_val,
                                         y=y_val,
                                         kfolds=self.kfolds,
                                         verbose=False)
        # Cross-validation
        else:
            # Evaluate the individual
            self.model_evaluator.compute(estimator=self.model.estimator,
                                         eval_mode=self.eval_mode,
                                         X=X_train,
                                         y=y_train,
                                         kfolds=self.kfolds,
                                         verbose=False)
        # Get evaluation
        evaluation = self.model_evaluator.values[self.eval_function]

        return evaluation
