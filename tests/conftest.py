import pytest
import numpy as np
from unittest.mock import MagicMock
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


@pytest.fixture
def dummy_classification_evaluator():
    """Fixture for a dummy classification evaluator."""
    from pyccea.evaluation.wrapper import WrapperEvaluation
    evaluator = MagicMock(spec=WrapperEvaluation)
    evaluator.task = "classification"
    evaluator.eval_function = "accuracy"
    return evaluator


@pytest.fixture
def dummy_regression_evaluator():
    """Fixture for a dummy regression evaluator."""
    from pyccea.evaluation.wrapper import WrapperEvaluation
    evaluator = MagicMock(spec=WrapperEvaluation)
    evaluator.task = "regression"
    evaluator.eval_function = "mse"
    return evaluator


@pytest.fixture
def dummy_dataloader():
    """Fixture for a dummy DataLoader."""
    from pyccea.utils.datasets import DataLoader
    data_loader = MagicMock(spec=DataLoader)
    data_loader.n_features = 10
    return data_loader


@pytest.fixture
def dummy_collaborator():
    """Fixture for a dummy collaborator."""
    from pyccea.cooperation import Collaboration
    collaborator = MagicMock(spec=Collaboration)
    return collaborator


@pytest.fixture
def dummy_fitness_function():
    """Fixture for a dummy fitness function."""
    from pyccea.fitness import WrapperFitnessFunction
    fitness_function = MagicMock(spec=WrapperFitnessFunction)
    return fitness_function
