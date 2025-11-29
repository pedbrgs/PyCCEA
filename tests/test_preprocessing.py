import pytest
import numpy as np
from pyccea.utils.preprocessing import Winsoriser


def test_winsoriser() -> None:
    """Test the quantile-based truncation method."""
    a = np.array([
        [10, 100], 
        [1, 110], 
        [100, 10], 
        [12, 120], 
        [200, 1000]
    ])
    lower, upper = [0.10, 0.90]
    expected_result = np.array([
        [10, 100],
        [4.6, 110],
        [100, 46],
        [12, 120],
        [160, 648],
    ])
    expected_lower_bounds = np.array([4.6, 46])
    expected_upper_bounds = np.array([160, 648])
    winsor = Winsoriser(lower, upper)
    result = winsor.fit_transform(a)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == a.shape[0]
    assert result.shape[1] == a.shape[1]
    assert np.allclose(result, expected_result, atol=1e-2)
    assert np.allclose(winsor._lower_bounds, expected_lower_bounds)
    assert np.allclose(winsor._upper_bounds, expected_upper_bounds)
