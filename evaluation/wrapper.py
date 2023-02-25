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

    def __init__(self, task: str, model_type: str, eval_function: str):
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
                f"The available {task} metrics are {', '.join(self.model_evaluator.metrics)}."
            )
        self.eval_function = eval_function
        # Initialize the model present in the wrapper model_evaluator
        self.base_model = WrapperEvaluation.models[task](model_type=model_type)
        self.model_type = model_type
        # Initialize logger with info level
        logging.basicConfig(encofing='utf-8', level=logging.INFO)

    def evaluate(self,
                 indiv: np.array,
                 X_train: np.array,
                 y_train: np.array,
                 X_test: np.array,
                 y_test: np.array):
        """
        Evaluate an individual from a subpopulation through the predictive performance of a model
        and according to an evaluation metric.

        Parameters
        ----------
        indiv: np.array
            Individual represented by a binary n-dimensional array, where n is the number of
            features.
        X_train: np.array
            Train input data.
        X_test: np.array
            Test input data.
        y_train: np.array
            Train output data.
        y_test: np.array
            Test output data.
        """
        # Boolean array used to filter which features will be used to fit the model
        var_mask = indiv.astype(bool)
        X_train = X_train[:, var_mask].copy()
        X_test = X_test[:, var_mask].copy()
        # Get model that has not been previously fitted
        self.model = copy.deepcopy(self.base_model)
        # Train model with the current subset of features
        self.model.train(X_train=X_train, y_train=y_train, optimize=False, verbose=False)
        # Predict
        y_pred = self.model.estimator.predict(X_test)
        # Evaluate individual
        self.model_evaluator.compute(y_pred=y_pred, y_test=y_test, verbose=False)
        # Get evaluation
        evaluation = self.model_evaluator.values[self.eval_function]

        return evaluation
