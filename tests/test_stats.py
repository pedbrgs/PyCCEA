import pytest
import numpy as np
from pyccea.utils.stats import statistical_comparison_between_independent_samples


def test_ttest_used_for_normal_data() -> None:
    """Test if t-test is used for normally distributed data."""
    np.random.seed(42)
    x = np.random.normal(loc=80, scale=5, size=100)
    y = np.random.normal(loc=90, scale=5, size=100)
    test_name, _, p_comparison, reject_null = statistical_comparison_between_independent_samples(x, y, alpha=0.05)
    assert test_name == "t-test"
    assert reject_null is True
    assert p_comparison < 0.05


def test_mann_whitney_u_used_for_non_normal_data() -> None:
    """Test if Mann-Whitney U test is used for non-normally distributed data."""
    np.random.seed(42)
    x = np.random.exponential(scale=1.00, size=30)
    y = np.random.exponential(scale=1.05, size=30)
    test_name, _, p_comparison, reject_null = statistical_comparison_between_independent_samples(x, y, alpha=0.05)
    assert test_name == "mann-whitney-u"
    assert reject_null is False
    assert p_comparison > 0.05


def test_small_sample_size() -> None:
    """Test that a ValueError is raised for small sample sizes (lower than 3)."""
    np.random.seed(42)
    x = np.random.normal(loc=80, scale=10, size=2)
    y = np.random.normal(loc=90, scale=5, size=2)
    with pytest.raises(ValueError, match="1st sample has only 2 observation\\(s\\)\\.|2nd sample has only 2 observation\\(s\\)\\."):
        statistical_comparison_between_independent_samples(x, y, alpha=0.05)
    x = np.random.normal(loc=80, scale=10, size=30)
    with pytest.raises(ValueError, match="1st sample has only 2 observation\\(s\\)\\.|2nd sample has only 2 observation\\(s\\)\\."):
        statistical_comparison_between_independent_samples(x, y, alpha=0.05)


def test_invalid_alternative() -> None:
    """Test that a ValueError is raised for invalid alternative hypothesis."""
    np.random.seed(42)
    x = np.random.normal(loc=80, scale=10, size=30)
    y = np.random.normal(loc=90, scale=5, size=30)
    with pytest.raises(ValueError, match="Alternative hypothesis must be 'two-sided', 'greater', or 'less'."):
        statistical_comparison_between_independent_samples(x, y, alpha=0.05, alternative="invalid")
