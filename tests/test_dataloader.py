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
        with pytest.raises(
            AssertionError,
            match=f"The '{primary_key}' section should be specified in the data configuration file."
        ):
            incomplete_data_conf = complete_data_conf.copy()
            del incomplete_data_conf[primary_key]
            DataLoader("dummy_dataset", incomplete_data_conf)


def test_parse_general_parameters_missing_splitter_type(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with missing splitter_type."""
    complete_data_conf["general"].pop("splitter_type")
    with pytest.raises(
        AssertionError,
        match="The 'splitter_type' parameter should be specified in the general section of the "
        "data configuration file."
    ):
        DataLoader("dummy_dataset", complete_data_conf)


def test_parse_general_parameters_invalid_splitter_type(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with invalid splitter_type."""
    complete_data_conf["general"]["splitter_type"] = "invalid_type"
    with pytest.raises(
        NotImplementedError,
        match="The splitter type 'invalid_type' is not implemented."
    ):
        DataLoader("dummy_dataset", complete_data_conf)
