import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from pyccea.utils.datasets import DataLoader
from pyccea.evaluation import wrapper as wrapper_module
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


def test_cache_hit_avoids_recompute(monkeypatch) -> None:
    """Test when cache hit avoids recompute the same solution."""
    calls = {"n": 0}

    def fake_hold_out(self, solution_mask, data):
        calls["n"] += 1
        self.evaluations = {m: float(calls["n"]) for m in self.model_evaluator.metrics}

    monkeypatch.setattr(WrapperEvaluation, "_hold_out_validation", fake_hold_out)

    evaluator = WrapperEvaluation(
        task="classification",
        n_classes=2,
        eval_function="accuracy",
        model_type="logistic_regression",
        eval_mode="hold_out",
        cache_size=128,
        store_estimators=False
    )

    solution = np.array([0, 1, 0, 1, 1, 0, 0, 0], dtype=np.int8)

    r1 = evaluator.evaluate(solution=solution, data=None)
    r2 = evaluator.evaluate(solution=solution, data=None)

    assert r1 == r2
    assert calls["n"] == 1
    assert r2 is not r1  # second return is a copy from cache (should not be the same object)


def test_cache_disabled(monkeypatch) -> None:
    """Test when cache is disabled."""
    calls = {"n": 0}

    def fake_hold_out(self, solution_mask, data):
        calls["n"] += 1
        self.evaluations = {m: float(calls["n"]) for m in self.model_evaluator.metrics}

    monkeypatch.setattr(WrapperEvaluation, "_hold_out_validation", fake_hold_out)

    evaluator = WrapperEvaluation(
        task="classification",
        n_classes=2,
        eval_function="accuracy",
        model_type="logistic_regression",
        eval_mode="hold_out",
        cache_size=0,
        store_estimators=False
    )

    solution = np.array([0, 1, 0, 1, 1, 0, 0, 0], dtype=np.int8)

    _ = evaluator.evaluate(solution=solution, data=None)
    _ = evaluator.evaluate(solution=solution, data=None)

    assert calls["n"] == 2


def test_cache_eviction(monkeypatch) -> None:
    """Test cache eviction when max size is reached."""
    calls = {"n": 0}

    def fake_hold_out(self, solution_mask, data):
        calls["n"] += 1
        self.evaluations = {m: float(calls["n"]) for m in self.model_evaluator.metrics}

    monkeypatch.setattr(WrapperEvaluation, "_hold_out_validation", fake_hold_out)

    evaluator = WrapperEvaluation(
        task="classification",
        n_classes=2,
        eval_function="accuracy",
        model_type="logistic_regression",
        eval_mode="hold_out",
        cache_size=2,
        store_estimators=False
    )

    solution1 = np.array([0, 1, 0, 1, 1, 0, 0, 0], dtype=np.int8)
    solution2 = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8)
    solution3 = np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=np.int8)

    _ = evaluator.evaluate(solution=solution1, data=None)  # cached
    _ = evaluator.evaluate(solution=solution2, data=None)  # cached
    _ = evaluator.evaluate(solution=solution1, data=None)  # cache hit
    _ = evaluator.evaluate(solution=solution3, data=None)  # causes eviction of solution2
    _ = evaluator.evaluate(solution=solution2, data=None)  # recompute

    assert calls["n"] == 4


def test_subprocess_fallback_on_non_posix(monkeypatch) -> None:
    """Test subprocess evaluation falls back when not on POSIX system."""
    evaluator = WrapperEvaluation(
        task="classification",
        n_classes=2,
        eval_function="accuracy",
        model_type="logistic_regression",
        eval_mode="hold_out",
        use_subprocess=True,
        store_estimators=False
    )

    def fake_core(self, solution, data):
        return {"accuracy": 0.87}

    monkeypatch.setattr(wrapper_module.os, "name", "nt", raising=False)
    monkeypatch.setattr(WrapperEvaluation, "_evaluate_core", fake_core)

    result = evaluator._evaluate_in_subprocess(solution=np.array([1, 0]), data=None)

    assert result == {"accuracy": 0.87}


def test_evaluate_uses_subprocess(monkeypatch) -> None:
    """Test evaluate delegates to subprocess path when enabled."""
    evaluator = WrapperEvaluation(
        task="classification",
        n_classes=2,
        eval_function="accuracy",
        model_type="logistic_regression",
        eval_mode="hold_out",
        use_subprocess=True,
        store_estimators=False
    )

    expected = {"accuracy": 0.92}
    mock = MagicMock(return_value=expected)
    monkeypatch.setattr(evaluator, "_evaluate_in_subprocess", mock)

    result = evaluator.evaluate(solution=np.array([1, 0]), data=None)
    assert result == expected
    assert mock.call_count == 1


def test_subprocess_store_estimators_raises() -> None:
    """Test subprocess evaluation forbids storing estimators."""
    with pytest.raises(ValueError, match="Subprocess evaluation does not support storing estimators."):
        _ = WrapperEvaluation(
            task="classification",
            n_classes=2,
            eval_function="accuracy",
            model_type="logistic_regression",
            eval_mode="hold_out",
            use_subprocess=True,
            store_estimators=True
        )


def test_wrapper_evaluation_clone_copies_config() -> None:
    """Test clone returns new evaluator with the same configuration."""
    evaluator = WrapperEvaluation(
        task="classification",
        n_classes=2,
        eval_function="accuracy",
        model_type="logistic_regression",
        eval_mode="hold_out",
        cache_size = 256,
        use_subprocess=True,
        store_estimators=False
    )

    cloned = evaluator.clone()

    assert cloned is not evaluator
    assert cloned.task == evaluator.task
    assert cloned.model_type == evaluator.model_type
    assert cloned.eval_function == evaluator.eval_function
    assert cloned.eval_mode == evaluator.eval_mode
    assert cloned.store_estimators == evaluator.store_estimators
    assert cloned.cache_size == evaluator.cache_size
    assert cloned.use_subprocess == evaluator.use_subprocess
