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


def test_build_subpopulations(
        dummy_collaborator,
        dummy_fitness_function,
        dummy_dataloader
    ) -> None:
    """Test that the build_subpopulations method initializes subpopulations correctly."""

    # Patch dummy_collaborator methods
    dummy_collaborator.get_collaborators = lambda subpop_idx, indiv_idx, previous_subpops, current_subpops: [
        subpop[0] for subpop in current_subpops
    ]
    dummy_collaborator.build_context_vector = lambda collaborators: np.concatenate(collaborators)

    # Patch dummy_fitness_function
    dummy_fitness_function.evaluate = lambda x, data: 0  # dummy return

    # Define a minimal concrete subclass just for testing
    class ConcreteSubpopInit(SubpopulationInitialization):
        def _get_subpop(self, subcomp_size, subpop_size):
            # Return all-zero binary arrays for simplicity
            return np.zeros((subpop_size, subcomp_size), dtype=int)

        def _build_context_vector(self, subpop_idx, indiv_idx, subpops):
            # Return a dummy vector for testing
            return np.concatenate([subpop[0] for subpop in subpops])

    instance = ConcreteSubpopInit(
        data=dummy_dataloader,
        subcomp_sizes=[5, 10, 15],
        subpop_sizes=[2, 3, 4],
        collaborator=dummy_collaborator,
        fitness_function=dummy_fitness_function,
    )

    # Call the method under test
    instance.build_subpopulations()

    # Check that the subpops list has the correct number of subpopulations
    assert len(instance.subpops) == len(instance.subcomp_sizes)
    for i, (expected_size, expected_subpop_size) in enumerate(zip(instance.subcomp_sizes, instance.subpop_sizes)):
        # Check that each subpopulation has the correct size
        assert len(instance.subpops[i]) == expected_subpop_size
        for individual in instance.subpops[i]:
            # Check that each individual has the correct size
            assert len(individual) == expected_size


def test_evaluate_individuals(
        dummy_collaborator,
        dummy_fitness_function,
        dummy_dataloader
    ) -> None:
    """Test the evaluate_individuals method."""

    # Patch dummy collaborator methods (if used by _build_context_vector)
    dummy_collaborator.get_collaborators = lambda *args, **kwargs: []
    dummy_collaborator.build_context_vector = lambda collaborators: np.array([1, 2, 3])

    # Patch dummy fitness function to return a fixed fitness
    fitness_value = 0.5
    dummy_fitness_function.evaluate = lambda x, data: fitness_value

    # Concrete subclass with dummy abstract methods
    class ConcreteSubpopInit(SubpopulationInitialization):
        def _get_subpop(self, subcomp_size, subpop_size):
            # Return subpopulations filled with dummy individuals (arrays of zeros)
            return np.zeros((subpop_size, subcomp_size), dtype=int)

        def _build_context_vector(self, subpop_idx, indiv_idx, subpops):
            # Return a simple vector representing a complete solution (e.g., fixed vector)
            return np.array([indiv_idx, subpop_idx])

    # Instantiate the concrete class with dummy data
    instance = ConcreteSubpopInit(
        data=dummy_dataloader,
        subcomp_sizes=[3, 2],   # small subpop sizes for test
        subpop_sizes=[2, 3],
        collaborator=dummy_collaborator,
        fitness_function=dummy_fitness_function,
    )

    # Initialize subpopulations (required before evaluation)
    instance.build_subpopulations()

    # Run method under test
    instance.evaluate_individuals()

    # Assertions:
    # context_vectors and fitness should be lists with length == number of subpopulations
    assert len(instance.context_vectors) == len(instance.subpop_sizes)
    assert len(instance.fitness) == len(instance.subpop_sizes)

    # Each entry in context_vectors should be a numpy array with shape matching subpop size
    for i, subpop in enumerate(instance.subpops):
        assert instance.context_vectors[i].shape[0] == len(subpop)
        # Each fitness list should have fitness values equal to subpop size
        assert len(instance.fitness[i]) == len(subpop)
        # Fitness values should be what dummy_fitness_function.evaluate returns (42.0)
        assert all(f == fitness_value for f in instance.fitness[i])


def test_subpopulation_initialization_abstract_methods(
        dummy_collaborator,
        dummy_fitness_function,
        dummy_dataloader
    ) -> None:

    class DummySubpopInit(SubpopulationInitialization):
        # Override abstract methods but call super() to test base implementation
        def _get_subpop(self, subcomp_size, subpop_size):
            return super()._get_subpop(subcomp_size, subpop_size)

        def _build_context_vector(self, subpop_idx, indiv_idx, subpops):
            return super()._build_context_vector(subpop_idx, indiv_idx, subpops)

    instance = DummySubpopInit(
        data=dummy_dataloader,
        subcomp_sizes=[1],
        subpop_sizes=[1],
        collaborator=dummy_collaborator,
        fitness_function=dummy_fitness_function
    )

    instance._get_subpop(5, 10)
    instance._build_context_vector(0, 0, [])


def test_build_context_vector_binary_initialization(
        dummy_collaborator,
        dummy_fitness_function,
        dummy_dataloader
    ) -> None:
    """Test the _build_context_vector method for binary initialization."""
    subpop_idx = 1
    indiv_idx = 4
    subcomp_sizes = [7, 4]
    subpop_sizes = [2, 5]

    # Patch dummy collaborator methods
    dummy_collaborator.get_collaborators = lambda *args, **kwargs: []
    dummy_collaborator.build_context_vector = lambda collaborators: np.ones(sum(subcomp_sizes))

    # Patch dummy fitness function to return a fixed fitness
    fitness_value = 0.5
    dummy_fitness_function.evaluate = lambda x, data: fitness_value

    # Instantiate the concrete class with dummy data
    instance = RandomBinaryInitialization(
        data=dummy_dataloader,
        subcomp_sizes=subcomp_sizes,
        subpop_sizes=subpop_sizes,
        collaborator=dummy_collaborator,
        fitness_function=dummy_fitness_function,
    )

    # Initialize subpopulations (required before evaluation)
    instance.build_subpopulations()

    # Run method under test
    context_vector = instance._build_context_vector(
        subpop_idx=subpop_idx,
        indiv_idx=indiv_idx,
        subpops=instance.subpops
    )

    # Assertion
    assert context_vector.shape == (sum(subcomp_sizes),)


def test_build_context_vector_continuous_initialization(
        dummy_collaborator,
        dummy_fitness_function,
        dummy_dataloader
    ) -> None:
    """Test the _build_context_vector method for continuous initialization."""
    subpop_idx = 1
    indiv_idx = 4
    subcomp_sizes = [7, 4]
    subpop_sizes = [2, 5]
    bounds = (-10, 10)

    # Patch dummy collaborator methods
    dummy_collaborator.get_collaborators = lambda *args, **kwargs: []
    dummy_collaborator.build_context_vector = lambda collaborators: np.ones(sum(subcomp_sizes))

    # Patch dummy fitness function to return a fixed fitness
    fitness_value = 0.5
    dummy_fitness_function.evaluate = lambda x, data: fitness_value

    # Instantiate the concrete class with dummy data
    instance = RandomContinuousInitialization(
        data=dummy_dataloader,
        subcomp_sizes=subcomp_sizes,
        subpop_sizes=subpop_sizes,
        collaborator=dummy_collaborator,
        fitness_function=dummy_fitness_function,
        bounds=bounds
    )

    # Initialize subpopulations (required before evaluation)
    instance.build_subpopulations()

    # Run method under test
    context_vector = instance._build_context_vector(
        subpop_idx=subpop_idx,
        indiv_idx=indiv_idx,
        subpops=instance.subpops
    )

    # Assertion
    assert context_vector.shape == (sum(subcomp_sizes),)
