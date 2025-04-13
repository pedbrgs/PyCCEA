import toml
import pytest
from pyccea.utils.config import load_params


def test_load_params(tmp_path, complete_data_conf: dict) -> None:
    """Test loading parameters from a TOML configuration file."""
    conf_file = tmp_path / "test_config.toml"
    conf_file.write_text(toml.dumps(complete_data_conf))

    # Load parameters using the function
    loaded_conf = load_params(str(conf_file))

    # Assertions to check if loaded data matches expected
    assert isinstance(loaded_conf, dict)
    assert loaded_conf == complete_data_conf


def test_load_params_invalid_toml(tmp_path) -> None:
    """Test that loading a malformed TOML file raises TomlDecodeError."""
    invalid_conf_file = tmp_path / "invalid_config.toml"
    invalid_conf_file.write_text("invalid = ??? this is not toml")

    with pytest.raises(toml.TomlDecodeError):
        load_params(str(invalid_conf_file))
