import pytest
import numpy as np
from pyccea.utils.metrics import ClassificationMetrics


class DummyClassifier:
    """A simple mock classifier returning pre-defined predictions."""
    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, X):
        return self._predictions


def test_classification_metrics_binary() -> None:
    """Test binary classification metrics."""
    X_test = np.array([[0], [1], [2], [3]])
    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])

    estimator = DummyClassifier(predictions=y_pred)

    metrics = ClassificationMetrics(n_classes=2)
    metrics.compute(estimator, X_test, y_test)

    expected_values = {
        "precision": 1.0,
        "recall": 0.5,
        "f1_score": 0.6667,
        "accuracy": 0.75,
        "balanced_accuracy": 0.75,
        "specificity": 1.0
    }

    for key, expected in expected_values.items():
        assert metrics.values[key] == pytest.approx(expected, abs=1e-4)


def test_classification_metrics_multiclass() -> None:
    """Test multiclass classification metrics."""
    X_test = np.array([[0], [1], [2], [3], [4], [5]])
    y_test = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 1, 2])

    estimator = DummyClassifier(predictions=y_pred)

    metrics = ClassificationMetrics(n_classes=3)
    metrics.compute(estimator, X_test, y_test)

    expected_values = {
        "accuracy": 4 / 6,
        "balanced_accuracy": (1.0 + 0.5 + 0.5) / 3,
        "recall": (1.0 + 0.5 + 0.5) / 3,
        "precision": (1.0 + 0.5 + 0.5) / 3,
        "f1_score": (1.0 + 0.5 + 0.5) / 3
    }

    for metric_name, expected in expected_values.items():
        assert metrics.values[metric_name] == pytest.approx(expected, abs=1e-4), f"{metric_name} mismatch"


def test_classification_metrics_invalid_input() -> None:
    """Test invalid input for classification metrics."""
    X_test = np.array([[0], [1]])
    y_test = np.array([0, 1])
    # Wrong length on purpose
    y_pred = np.array([0, 1, 0])

    estimator = DummyClassifier(predictions=y_pred)
    metrics = ClassificationMetrics(n_classes=2)

    with pytest.raises(ValueError):
        metrics.compute(estimator, X_test, y_test)


def test_classification_metrics_verbose_output(caplog) -> None:
    """Test verbose output of classification metrics."""
    X_test = np.array([[0], [1]])
    y_test = np.array([0, 1])
    y_pred = np.array([0, 1])

    estimator = DummyClassifier(predictions=y_pred)
    metrics = ClassificationMetrics(n_classes=2)

    with caplog.at_level("INFO"):
        metrics.compute(estimator, X_test, y_test, verbose=True)
        assert "Precision" in caplog.text
        assert "Accuracy" in caplog.text
        assert "F1-score" in caplog.text
