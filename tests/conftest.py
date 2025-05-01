import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def complete_data_conf():
    """Fixture for complete data configuration."""
    return {
        "general": {"splitter_type": "k_fold", "verbose": True},
        "splitter": {"preset": True, "kfolds": 5, "stratified": True},
        "normalization": {"normalize": True, "method": "min_max"}
    }


@pytest.fixture
def classification_data():
    """Fixture for simple classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        n_informative=3,
        random_state=42
    )
    return X, y


@pytest.fixture
def regression_data():
    """Fixture for simple regression dataset."""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        noise=5.0,
        random_state=42
    )
    return X, y


@pytest.fixture
def generation():
    """Fixture for subpopulations."""
    # Current subpopulation is the subpopulation after evolve
    current_subpops = [
        np.array([np.array([1, 1, 0, 0, 1]), np.array([0, 0, 0, 1, 0])]),
        np.array([np.array([1, 1])]),
        np.array([np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 1, 1])])
    ]
    # Previous subpopulation is the subpopulation before evolve (already evaluated)
    previous_subpops = [
        np.array([np.array([0, 1, 0, 1, 1]), np.array([1, 0, 1, 1, 0])]),
        np.array([np.array([1, 0])]),
        np.array([np.array([1, 0, 1]), np.array([0, 1, 1]), np.array([0, 0, 1])])
    ]
    # Fitness values of the individuals in the previous subpopulations
    fitness = [
        [0.63, 0.86],
        [0.31],
        [0.44, 0.15, 0.81]
    ]
    # Best individuals and context vectors related to the previous subpopulations
    current_best = {
        0: {
            "individual": np.array([1, 0, 1, 1, 0]),
            "context_vector": np.array([1, 0, 1, 1, 0, 1, 1, 0, 0, 1]),
            "fitness": 0.86
        },
        1: {
            "individual": np.array([1, 0]),
            "context_vector": np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1]),
            "fitness": 0.31
        },
        2: {
            "individual": np.array([0, 0, 1]),
            "context_vector": np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 1]),
            "fitness": 0.81
        }
    }
    return current_subpops, fitness, current_best, previous_subpops
