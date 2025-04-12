import pytest


@pytest.fixture
def complete_data_conf():
    """Fixture for complete data configuration."""
    return {
        "general": {"splitter_type": "k_fold", "verbose": True},
        "splitter": {"preset": True, "kfolds": 5, "stratified": True},
        "normalization": {"normalize": True, "method": "min_max"}
    }