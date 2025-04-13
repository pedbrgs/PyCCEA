import pytest
import logging
from pyccea.utils.datasets import DataLoader


# Disable logging output during tests
logging.disable(logging.CRITICAL)


def test_init_valid_data_conf(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with a valid configuration."""
    DataLoader("dummy_dataset", complete_data_conf)


def test_init_asserts_missing_data_conf_sections(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with configurations missing required sections."""
    for primary_key in complete_data_conf.keys():
        with pytest.raises(AssertionError):
            incomplete_data_conf = complete_data_conf.copy()
            incomplete_data_conf.pop(primary_key)
            DataLoader("dummy_dataset", incomplete_data_conf)


def test_parse_general_parameters_missing_splitter_type(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with missing splitter_type."""
    complete_data_conf["general"].pop("splitter_type")
    with pytest.raises(AssertionError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_parse_general_parameters_invalid_splitter_type(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with invalid splitter_type."""
    complete_data_conf["general"]["splitter_type"] = "invalid_type"
    with pytest.raises(NotImplementedError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_parse_splitter_parameters_missing_kfolds(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with missing kfolds."""
    complete_data_conf["splitter_type"] = "k_fold"
    del complete_data_conf["splitter"]["kfolds"]
    with pytest.raises(AssertionError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_parse_splitter_parameters_missing_test_ratio_and_preset_false(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with missing test ratio when preset is false."""
    complete_data_conf["splitter"]["preset"] = False
    with pytest.raises(ValueError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_parse_splitter_parameters_test_ratio_out_of_bounds(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with test ratio out of bounds."""
    complete_data_conf["splitter"]["preset"] = False
    for test_ratio in [-0.5, 1.5]:
        complete_data_conf["splitter"]["test_ratio"] = test_ratio
        with pytest.raises(ValueError):
            DataLoader("dummy_dataset", complete_data_conf)


def test_parse_normalization_parameters_missing_method(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with missing normalization method."""
    complete_data_conf["normalization"]["normalize"] = True
    complete_data_conf["normalization"].pop("method")
    with pytest.raises(AssertionError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_parse_normalization_parameters_invalid_method(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with invalid normalization method."""
    complete_data_conf["normalization"]["normalize"] = True
    complete_data_conf["normalization"]["method"] = "invalid_method"
    with pytest.raises(NotImplementedError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_parse_normalization_parameters_method_and_normalize_false(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with normalization method when normalize is false."""
    complete_data_conf["normalization"]["normalize"] = False
    complete_data_conf["normalization"]["method"] = "min_max"
    with pytest.raises(ValueError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_load_data_invalid_dataset_name(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with an invalid dataset name."""
    with pytest.raises(ValueError):
        dl = DataLoader("invalid_dataset", complete_data_conf)
        dl._load()