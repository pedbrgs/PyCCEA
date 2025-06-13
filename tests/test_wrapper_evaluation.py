import pytest
import numpy as np
import pandas as pd
from pyccea.utils.datasets import DataLoader
from pyccea.evaluation.wrapper import WrapperEvaluation


def test_invalid_task() -> None:
    """Test invalid task."""
    with pytest.raises(NotImplementedError):
        WrapperEvaluation(
            task="invalid_task",
            n_classes=2,
            eval_function="accuracy",
            model_type="random_forest",
            eval_mode="hold_out"
        )


def test_invalid_eval_function() -> None:
    """Test invalid evaluation function."""
    with pytest.raises(NotImplementedError):
        WrapperEvaluation(
            task="classification",
            n_classes=2,
            eval_function="invalid_eval_function",
            model_type="random_forest",
            eval_mode="hold_out"
        )


def test_invalid_eval_mode() -> None:
    """Test invalid evaluation mode."""
    with pytest.raises(NotImplementedError):
        WrapperEvaluation(
            task="regression",
            eval_function="rmse",
            model_type="ridge",
            eval_mode="invalid_eval_mode"
        )


def test_hold_out_validation(monkeypatch, classification_data, complete_data_conf: dict) -> None:
    """Test hold out validation."""
    dl = DataLoader("dummy", complete_data_conf)
    monkeypatch.setitem(DataLoader.DATASETS, "dummy", {"task": "classification"})
    X, y = classification_data
    dl.data = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]))
    # Create "train" and "test" labels
    n_samples = len(dl.data)
    split_idx = int(0.7 * n_samples)
    train_test_split = np.array(["train"] * split_idx + ["test"] * (n_samples - split_idx))
    dl.data[6] = train_test_split
    dl.data = dl.data.rename(columns={
        dl.data.columns[-2]: "label",
        dl.data.columns[-1]: "subset"
    })
    dl.X = dl._get_input()
    dl.y = dl._get_output()
    dl._split()
    dl.X_train, dl.X_test = dl._normalize_subsets(X_train=dl.X_train, X_test=dl.X_test)
    evaluator = WrapperEvaluation(
        task="classification",
        n_classes=2,
        eval_function="accuracy",
        model_type="logistic_regression",
        eval_mode="hold_out"
    )
    # Create a mock solution mask
    solution = np.array([1, 0, 1, 1, 0])
    # Evaluate the individual
    evaluator.evaluate(solution=solution, data=dl)
    # Assert that only one estimator was trained and stored (as expected from hold-out validation)
    assert len(evaluator.estimators) == 1
    # Assert that the estimator was trained using exactly 3 features (according to the solution mask)
    assert evaluator.estimators[0].n_features_in_ == solution.sum()
    # Assert that all expected evaluation metrics are present in the evaluator's output dictionary
    for metric in evaluator.model_evaluator.metrics:
        assert metric in evaluator.evaluations.keys()


def test_evaluate_with_empty_solution(monkeypatch, complete_data_conf) -> None:
    """Test evaluate with empty solution."""
    evaluator = WrapperEvaluation(
        task="classification",
        n_classes=2,
        eval_function="accuracy",
        model_type="logistic_regression",
        eval_mode="hold_out"
    )
    # Create a mock solution mask
    solution = np.array([])
    # Create a mock dataset
    dl = DataLoader("dummy", complete_data_conf)
    monkeypatch.setitem(DataLoader.DATASETS, "dummy", {"task": "classification"})
    # Evaluate the individual
    evaluator.evaluate(solution=solution, data=dl)
    # Construct expected evaluations dictionary
    expected = {metric: 0.0 for metric in evaluator.model_evaluator.metrics}
    # Assert evaluation and estimator results
    assert evaluator.evaluations == expected
    assert evaluator.estimators == []


def test_kfold_cross_validation(monkeypatch, classification_data, complete_data_conf: dict) -> None:
    """Test k-fold cross validation."""
    dl = DataLoader("dummy", complete_data_conf)
    monkeypatch.setitem(DataLoader.DATASETS, "dummy", {"task": "classification"})
    X, y = classification_data
    dl.data = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]))
    # Create "train" and "test" labels
    n_samples = len(dl.data)
    split_idx = int(0.7 * n_samples)
    train_test_split = np.array(["train"] * split_idx + ["test"] * (n_samples - split_idx))
    dl.data[6] = train_test_split
    dl.data = dl.data.rename(columns={
        dl.data.columns[-2]: "label",
        dl.data.columns[-1]: "subset"
    })
    dl.X = dl._get_input()
    dl.y = dl._get_output()
    dl._split()
    dl._model_selection()
    if dl.normalize:
        dl.X_train, dl.X_test = dl._normalize_subsets(X_train=dl.X_train, X_test=dl.X_test)
    evaluator = WrapperEvaluation(
        task="classification",
        n_classes=2,
        eval_function="accuracy",
        model_type="logistic_regression",
        eval_mode="k_fold"
    )
    # Create a mock solution mask
    solution = np.array([1, 0, 1, 1, 0])
    # Evaluate the individual
    evaluator.evaluate(solution=solution, data=dl)
    # Assert that 5 estimators were stored (from k folds)
    assert len(evaluator.estimators) == complete_data_conf["splitter"]["kfolds"]
    # Assert that each trained estimator used exactly 3 features (as expected by the mask)
    for estimator in evaluator.estimators:
        assert estimator.n_features_in_ == solution.sum()
    # Assert that all expected evaluation metrics are present in the evaluator's output dictionary
    for metric in evaluator.model_evaluator.metrics:
        assert metric in evaluator.evaluations.keys()
