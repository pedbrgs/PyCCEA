import pytest
import numpy as np
from pyccea.utils.mapping import angle_modulation_function, shifted_heaviside_function


def test_angle_modulation_function_consistency() -> None:
    """Test angle_modulation_function with a known set of coefficients and features."""
    coeffs = np.array([0.0, 1.0, 1.0, 0.0])
    n_features = 5
    result = angle_modulation_function(coeffs, n_features)
    expected_result = np.array([1.0, 1.0, 0, 0, 0])

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_features,)
    assert set(np.unique(result)).issubset({0, 1})
    assert np.allclose(result, expected_result, atol=1e-2)


def test_angle_modulation_function_invalid_coeffs() -> None:
    """Test angle_modulation_function with invalid coefficients."""
    coeffs = np.array([0.0, 1.0, 1.0])
    n_features = 5

    with pytest.raises(ValueError):
        angle_modulation_function(coeffs, n_features)


def test_shifted_heaviside_function_consistency() -> None:
    """Test shifted_heaviside_function with a known set of continuous values."""
    real_solution = np.array([0.6, 0.4, 0.8, 0.2])
    expected_result = np.array([1, 0, 1, 0])

    result = shifted_heaviside_function(real_solution)

    assert isinstance(result, np.ndarray)
    assert result.shape == real_solution.shape
    assert set(np.unique(result)).issubset({0, 1})
    assert np.array_equal(result, expected_result)
