import pytest
from sklearn.datasets import make_classification


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
