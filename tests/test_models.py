import pytest
from sklearn.base import ClassifierMixin, RegressorMixin

from pyccea.utils.models import ClassificationModel, RegressionModel


def test_classification_model_initialization_valid() -> None:
    """Test model instantiation for each available classification model."""
    for model_type in ClassificationModel.models.keys():
        model = ClassificationModel(model_type)
        assert isinstance(model.estimator, ClassifierMixin)


def test_regression_model_initialization_valid() -> None:
    """Test model instantiation for each available regression model."""
    for model_type in RegressionModel.models.keys():
        model = RegressionModel(model_type)
        assert isinstance(model.estimator, RegressorMixin)


def test_classification_model_initialization_invalid() -> None:
    """Test invalid classification model type raises AssertionError."""
    with pytest.raises(AssertionError):
        ClassificationModel("linear_regression")


def test_regression_model_initialization_invalid() -> None:
    """Test invalid regression model type raises AssertionError."""
    with pytest.raises(AssertionError):
        RegressionModel("logistic_regression")


def test_train_classifier_without_optimization(classification_data) -> None:
    """Test classification model training without hyperparameter optimization."""
    X_train, y_train = classification_data
    model = ClassificationModel("random_forest")
    model.train(X_train, y_train, optimize=False)
    
    # Check that the estimator is fitted and hyperparams available
    assert hasattr(model.estimator, "predict")
    assert isinstance(model.estimator, ClassifierMixin)
    assert isinstance(model.hyperparams, dict)


def test_train_regressor_without_optimization(regression_data) -> None:
    """Test regression model training without hyperparameter optimization."""
    X_train, y_train = regression_data
    model = RegressionModel("random_forest")
    model.train(X_train, y_train, optimize=False)
    
    # Check that the estimator is fitted and hyperparams available
    assert hasattr(model.estimator, "predict")
    assert isinstance(model.estimator, RegressorMixin)
    assert isinstance(model.hyperparams, dict)


def test_train_classifier_with_optimization(classification_data) -> None:
    """Test classification model training with hyperparameter optimization."""
    X_train, y_train = classification_data
    model = ClassificationModel("random_forest")
    model.train(X_train, y_train, optimize=True, n_iter=5, kfolds=3, seed=42)
    
    # Check estimator is fitted and hyperparams are available after optimization
    assert hasattr(model.estimator, "predict")
    assert isinstance(model.estimator, ClassifierMixin)
    assert isinstance(model.hyperparams, dict)


def test_train_regressor_with_optimization(regression_data) -> None:
    """Test regression model training with hyperparameter optimization."""
    X_train, y_train = regression_data
    model = RegressionModel("random_forest")
    model.train(X_train, y_train, optimize=True, n_iter=5, kfolds=3, seed=42)
    
    # Check estimator is fitted and hyperparams are available after optimization
    assert hasattr(model.estimator, "predict")
    assert isinstance(model.estimator, RegressorMixin)
    assert isinstance(model.hyperparams, dict)


@pytest.mark.parametrize("model_type,expected_keys", [
    ("support_vector_machine", ["kernel", "degree", "gamma", "class_weight"]),
    ("random_forest", ["n_estimators", "criterion", "min_samples_split"]),
    ("complement_naive_bayes", ["alpha", "fit_prior"]),
    ("multinomial_naive_bayes", ["alpha", "fit_prior"]),
    ("gaussian_naive_bayes", ["var_smoothing"]),
    ("k_nearest_neighbors", ["n_neighbors"])
])
def test_classification_model_selection_grid_setup(classification_data, model_type, expected_keys):
    """Test classification model selection grid setup."""
    X, y = classification_data
    X_abs = abs(X)
    model = ClassificationModel(model_type)
    model._model_selection(X_abs, y, n_iter=2, seed=42, kfolds=2)
    for key in expected_keys:
        assert key in model.grid


@pytest.mark.parametrize("model_type,expected_keys", [
    ("linear", ["fit_intercept"]),
    ("elastic_net", ["alpha", "l1_ratio"]),
    ("ridge", ["alpha"]),
    ("lasso", ["alpha"]),
    ("random_forest", ["n_estimators", "max_features", "min_samples_split", "min_samples_leaf", "bootstrap"]),
    ("support_vector_machine", ["C", "epsilon", "kernel"]),
])
def test_regression_model_selection_grid_setup(regression_data, model_type, expected_keys):
    """Test regression model selection grid setup."""
    X, y = regression_data
    model = RegressionModel(model_type)
    model._model_selection(X, y, n_iter=2, seed=42, kfolds=2)
    for key in expected_keys:
        assert key in model.grid


def test_classification_model_clone_is_unfitted(classification_data) -> None:
    """Test clone returns a new, unfitted classification model with same hyperparameters."""
    X_train, y_train = classification_data
    model = ClassificationModel("random_forest")
    model.train(X_train, y_train, optimize=False)

    cloned = model.clone()

    assert cloned is not model
    assert cloned.model_type == model.model_type
    assert cloned.estimator is not model.estimator
    assert cloned.estimator.get_params() == model.estimator.get_params()
    assert not hasattr(cloned.estimator, "n_features_in_")


def test_regression_model_clone_is_unfitted(regression_data) -> None:
    """Test clone returns a new, unfitted regression model with same hyperparameters."""
    X_train, y_train = regression_data
    model = RegressionModel("random_forest")
    model.train(X_train, y_train, optimize=False)

    cloned = model.clone()

    assert cloned is not model
    assert cloned.model_type == model.model_type
    assert cloned.estimator is not model.estimator
    assert cloned.estimator.get_params() == model.estimator.get_params()
    assert not hasattr(cloned.estimator, "n_features_in_")