import os
import pickle
import logging
import warnings
import numpy as np
from collections import OrderedDict
from ..utils.datasets import DataLoader
from ..utils.models import ClassificationModel, RegressionModel
from ..utils.metrics import ClassificationMetrics, RegressionMetrics
from ..utils.memory import force_memory_release

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
    estimators : list of sklearn model objects, optional
        Estimators used in the current evaluation. It is one when 'eval_mode' is set to "hold_out"
        and k when 'eval_mode' is set to "k_fold" or "leave_one_out". If 'store_estimators' is 
        False, this attribute is not created.
    _cache : collections.OrderedDict
        Internal LRU cache mapping a packed-bit representation of 'solution' to the evaluation
        dict. The cache is bounded by 'cache_size'.
    """

    models = {"classification": ClassificationModel, "regression": RegressionModel}
    metrics = {"classification": ClassificationMetrics, "regression": RegressionMetrics}
    eval_modes = ["hold_out", "k_fold", "leave_one_out"]

    def __init__(
            self,
            task: str,
            model_type: str,
            eval_function: str,
            eval_mode: str,
            n_classes: int = None,
            store_estimators: bool = True,
            cache_size: int = 0,
            use_subprocess: bool = False
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
        store_estimators : bool, default True
            Whether to store the estimators used in the evaluation.
        cache_size : int, default 0
            Maximum number of distinct feature-subset evaluations to keep in an in-memory LRU
            cache. Set to 0 or None to disable caching. This is useful when the evolutionary loop
            revisits the same solution multiple time (e.g., due to elitism/crossover), avoiding
            redundant model fits. Note: caching assumes evaluation is deterministic for a given
            solution.
        use_subprocess : bool, default False
            Whether to evaluate in a subprocess to release native memory back to the OS.
        """
        # Check if the chosen task is available
        if not task in WrapperEvaluation.metrics.keys():
            raise NotImplementedError(
                f"Task '{task}' is not implemented. "
                f"The available tasks are {', '.join(WrapperEvaluation.metrics.keys())}."
            )
        # Initialize the model evaluator according to the task
        task_kwargs = {
            "classification": {"n_classes": n_classes},
            "regression": {},
        }
        self.model_evaluator = WrapperEvaluation.metrics[task](**task_kwargs[task])
        self.task = task
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
        self.store_estimators = store_estimators
        self.use_subprocess = use_subprocess
        if self.use_subprocess and self.store_estimators:
            raise ValueError("Subprocess evaluation does not support storing estimators.")
        # Optional bounded cache: avoids reffiting identifical feature subsets many times
        self.cache_size = int(cache_size) if cache_size is not None else 0
        self._cache = OrderedDict() if self.cache_size > 0 else None
        # Initialize logger with info level
        logging.basicConfig(encoding="utf-8", level=logging.INFO)

    def _hold_out_validation(self, solution_mask: np.ndarray, data: DataLoader) -> None:
        """Evaluate an individual using hold_out validation (train/test)."""

        # Get model that has not been previously fitted
        model = self.base_model.clone()
        # Train model with the current subset of features
        model.train(
            X_train=data.X_train[:, solution_mask],
            y_train=data.y_train,
            optimize=False,
            verbose=False
        )
        if self.store_estimators:
            self.estimators.append(model.estimator)
        # Evaluate the individual
        self.model_evaluator.compute(
            estimator=model.estimator,
            X_test=data.X_test[:, solution_mask],
            y_test=data.y_test,
            verbose=False
        )
        # Get evaluation in the test set
        self.evaluations = self.model_evaluator.values

        del model
        force_memory_release()

    def _cross_validation(self, solution_mask: np.ndarray, data: DataLoader) -> None:
        """Evaluate an individual using cross-validation (leave-one-out or k-fold)."""
        for k in range(data.kfolds):
            # Get training and validations subsets built from the full training set
            X_train, y_train, X_val, y_val = data.get_fold(k)
            # Get model that has not been previously fitted
            model = self.base_model.clone()
            # Train model with the current subset of features
            model.train(
                X_train=X_train[:, solution_mask],
                y_train=y_train,
                optimize=False,
                verbose=False
            )
            if self.store_estimators:
                self.estimators.append(model.estimator)
            # Evaluate the individual
            self.model_evaluator.compute(
                estimator=model.estimator,
                X_test=X_val[:, solution_mask],
                y_test=y_val,
                verbose=False
            )
            for metric in self.evaluations.keys():
                self.evaluations[metric] += self.model_evaluator.values[metric]
            del model

        # Calculate average performance over k folds
        for metric in self.evaluations.keys():
            self.evaluations[metric] = round(self.evaluations[metric]/data.kfolds, 4)

        del X_train, X_val, y_train, y_val
        force_memory_release()

    def _evaluate_core(self, solution: np.ndarray, data: DataLoader) -> dict:
        """Evaluate an individual without cache or subprocess."""
       # If no feature is selected
        self.evaluations = {metric: 0 for metric in self.model_evaluator.metrics}
        if solution.sum() == 0:
            return self.evaluations

        # Boolean array used to filter which features will be used to fit the model
        solution_mask = solution.astype(bool)

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

    def _evaluate_in_subprocess(self, solution: np.ndarray, data: DataLoader) -> dict:
        """Evaluate in a forked subprocess and return evaluations."""
        if os.name != "posix":
            return self._evaluate_core(solution=solution, data=data)

        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(read_fd)
                result = self._evaluate_core(solution=solution, data=data)
                payload = pickle.dumps(result)
                os.write(write_fd, payload)
            except Exception as e:
                payload = pickle.dumps({"__error__": repr(e)})
                os.write(write_fd, payload)
            finally:
                os.close(write_fd)
                os._exit(0)

        os.close(write_fd)
        chunks = list()
        while True:
            chunk = os.read(read_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        os.close(read_fd)
        os.waitpid(pid, 0)
        data_bytes = b"".join(chunks)
        if not data_bytes:
            return {}
        result = pickle.loads(data_bytes)
        if "__error__" in result:
            raise RunetimeError(result["__error__"])
        self.evaluations = result
        return result

    def evaluate(self, solution: np.ndarray, data: DataLoader) -> dict:
        """Evaluate an individual represented by a complete solution through the predictive
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
        # Cache lookup (key is unique for fixed-length solutions)
        cache_key = None
        if self.cache_size > 0:
            packed = np.packbits(solution.astype(np.uint8, copy=False))
            cache_key = (solution.shape[0], packed.tobytes())
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached.copy()

        # Estimator(s) used for the current evaluation
        if self.store_estimators:
            self.estimators = list()

        # Evaluate in subprocess (if enabled)
        if self.use_subprocess:
            result = self._evaluate_in_subprocess(solution=solution, data=data)
        else:
            result = self._evaluate_core(solution=solution, data=data)

        # Cache store (bounded)
        if cache_key is not None and self.cache_size > 0:
            self._cache[cache_key] = result.copy()
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return result
