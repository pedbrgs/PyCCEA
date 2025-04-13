import pytest
from sklearn.base import ClassifierMixin

from pyccea.utils.models import ClassificationModel


def test_model_initialization_valid() -> None:
    """Test model instantiation for each available model."""
    for model_type in ClassificationModel.models.keys():
        model = ClassificationModel(model_type)
        assert isinstance(model.estimator, ClassifierMixin)


def test_model_initialization_invalid() -> None:
    """Test invalid model type raises AssertionError."""
    with pytest.raises(AssertionError):
        ClassificationModel("invalid_model_name")


def test_train_without_optimization(classification_data) -> None:
    """Test model training without hyperparameter optimization."""
    X_train, y_train = classification_data
    model = ClassificationModel("random_forest")
    model.train(X_train, y_train, optimize=False)
    
    # Check that the estimator is fitted and hyperparams available
    assert hasattr(model.estimator, "predict")
    assert isinstance(model.hyperparams, dict)
    assert "n_estimators" in model.hyperparams  # for RandomForestClassifier


def test_train_with_optimization(classification_data) -> None:
    """Test model training with hyperparameter optimization."""
    X_train, y_train = classification_data
    model = ClassificationModel("random_forest")
    model.train(X_train, y_train, optimize=True, n_iter=5, kfolds=3, seed=42)
    
    # Check estimator is fitted and hyperparams are available after optimization
    assert hasattr(model.estimator, "predict")
    assert isinstance(model.hyperparams, dict)
    assert "n_estimators" in model.hyperparams


def test_model_selection_returns_expected(classification_data) -> None:
    """Test _model_selection returns estimator and hyperparams."""
    X_train, y_train = classification_data
    model = ClassificationModel("random_forest")
    estimator, hyperparams = model._model_selection(
        X_train,
        y_train,
        n_iter=3,
        kfolds=2,
        seed=42
    )
    
    assert hasattr(estimator, "predict")
    assert isinstance(hyperparams, dict)
    assert "n_estimators" in hyperparams


@pytest.mark.parametrize("model_type,expected_keys", [
    ("support_vector_machine", ["kernel", "degree", "gamma", "class_weight"]),
    ("random_forest", ["n_estimators", "criterion", "min_samples_split"]),
    ("complement_naive_bayes", ["alpha", "fit_prior"]),
    ("multinomial_naive_bayes", ["alpha", "fit_prior"]),
    ("gaussian_naive_bayes", ["var_smoothing"]),
    ("k_nearest_neighbors", ["n_neighbors"])
])
def test_model_selection_grid_setup(classification_data, model_type, expected_keys):
    X, y = classification_data
    X_abs = abs(X)
    model = ClassificationModel(model_type)
    model._model_selection(X_abs, y, n_iter=2, seed=42, kfolds=2)
    for key in expected_keys:
        assert key in model.grid