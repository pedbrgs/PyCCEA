import pytest
import numpy as np
from unittest.mock import MagicMock
from pyccea.initialization import (
    SubpopulationInitialization,
    RandomBinaryInitialization,
    RandomContinuousInitialization,
)


def test_get_subpop_binary_initialization(
        dummy_collaborator,
        dummy_fitness_function,
        dummy_dataloader
    ) -> None:
    """Test that the _get_subpop method returns a binary array of the correct shape."""
    n_subcomps = 4
    initializer = RandomBinaryInitialization(
        data=dummy_dataloader,
        subcomp_sizes=[5] * n_subcomps,
        subpop_sizes=[10] * n_subcomps,
        collaborator=dummy_collaborator,
        fitness_function=dummy_fitness_function,
    )
    subpop = initializer._get_subpop(subcomp_size=5, subpop_size=10)
    assert subpop.shape == (10, 5)
    assert np.all(np.isin(subpop, [0, 1]))  # only binary values


def test_get_subpop_continuous_initialization(
        dummy_collaborator,
        dummy_fitness_function,
        dummy_dataloader
    ) -> None:
    """Test that the _get_subpop method returns a continuous array of the correct shape."""
    n_subcomps = 4
    bounds = (-10, 10)
    initializer = RandomContinuousInitialization(
        data=dummy_dataloader,
        subcomp_sizes=[5] * n_subcomps,
        subpop_sizes=[10] * n_subcomps,
        collaborator=dummy_collaborator,
        fitness_function=dummy_fitness_function,
        bounds=bounds
    )
    subpop = initializer._get_subpop(subcomp_size=5, subpop_size=10)
    assert subpop.shape == (10, 5)
    assert np.all(subpop >= bounds[0])
    assert np.all(subpop <= bounds[1])

