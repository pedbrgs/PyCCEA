import pytest
import numpy as np
from pyccea.optimizers import (
    BinaryGeneticAlgorithm,
    DifferentialEvolution
)


def test_binary_genetic_algorithm_initialization(optimizer_config, dummy_dataloader) -> None:
    """Test the initialization of the BinaryGeneticAlgorithm class."""
    # Create an instance of BinaryGeneticAlgorithm
    optimizer = BinaryGeneticAlgorithm(
        subpop_size=optimizer_config["coevolution"]["subpop_sizes"][0],
        n_features=dummy_dataloader.n_features,
        conf=optimizer_config
    )
    # Check that the attributes are set correctly
    assert optimizer.n_features == dummy_dataloader.n_features
    assert optimizer.subpop_size == optimizer_config["coevolution"]["subpop_sizes"][0]
    assert optimizer.mutation_rate == optimizer_config["optimizer"]["mutation_rate"]
    assert optimizer.crossover_rate == optimizer_config["optimizer"]["crossover_rate"]
    assert optimizer.selection_method == optimizer_config["optimizer"]["selection_method"]


def test_binary_genetic_algorithm_invalid_initialization(optimizer_config, dummy_dataloader) -> None:
    """Test the invalid initialization of the BinaryGeneticAlgorithm class."""
    # Test with invalid selection method
    with pytest.raises(AssertionError):
        BinaryGeneticAlgorithm(
            subpop_size=optimizer_config["coevolution"]["subpop_sizes"][0],
            n_features=dummy_dataloader.n_features,
            conf={"optimizer": {**optimizer_config["optimizer"], "selection_method": "invalid_method"}}
        )
    # Test with invalid mutation rate
    with pytest.raises(ValueError):
        BinaryGeneticAlgorithm(
            subpop_size=optimizer_config["coevolution"]["subpop_sizes"][0],
            n_features=dummy_dataloader.n_features,
            conf={"optimizer": {**optimizer_config["optimizer"], "mutation_rate": -0.1}}
        )
    # Test with invalid crossover rate
    with pytest.raises(ValueError):
        BinaryGeneticAlgorithm(
            subpop_size=optimizer_config["coevolution"]["subpop_sizes"][0],
            n_features=dummy_dataloader.n_features,
            conf={"optimizer": {**optimizer_config["optimizer"], "crossover_rate": 1.5}}
        )


def test_single_point_crossover(monkeypatch, optimizer_config) -> None:
    """Test _single_point_crossover with mocked crossover point and probability."""
    optimizer = BinaryGeneticAlgorithm(
        subpop_size=10,
        n_features=5,
        conf=optimizer_config
    )
    crossover_point = 3
    optimizer_config["optimizer"]["crossover_rate"] = 0.10
    parent_a = np.array([1, 0, 1, 0, 1])
    parent_b = np.array([0, 1, 0, 1, 0])

    # Mock np.random.uniform to always return 0.1 (less than crossover_rate)
    monkeypatch.setattr("numpy.random.uniform", lambda *a, **k: optimizer_config["optimizer"]["crossover_rate"])
    # Mock np.random.randint to always return 3 as crossover point
    monkeypatch.setattr("numpy.random.randint", lambda *a, **k: crossover_point)

    offspring = optimizer._single_point_crossover(parent_a, parent_b)
    # Should take first 3 from parent_b, last 2 from parent_a
    expected = np.concatenate([parent_b[:crossover_point], parent_a[crossover_point:]])
    assert np.array_equal(offspring, expected)


def test_single_point_crossover_no_crossover(monkeypatch, optimizer_config) -> None:
    """Test _single_point_crossover when crossover does not occur (prob > crossover_rate)."""
    optimizer_config["optimizer"]["crossover_rate"] = 0.98
    optimizer = BinaryGeneticAlgorithm(
        subpop_size=10,
        n_features=5,
        conf=optimizer_config
    )
    parent_a = np.array([1, 0, 1, 0, 1])
    parent_b = np.array([0, 1, 0, 1, 0])

    # Mock np.random.uniform to always return 0.99 (greater than crossover_rate)
    monkeypatch.setattr("numpy.random.uniform", lambda *a, **k: 0.99)
    # Mock np.random.choice to always return 0 (so parent_a is chosen)
    monkeypatch.setattr("numpy.random.choice", lambda *a, **k: 0)

    offspring = optimizer._single_point_crossover(parent_a, parent_b)
    assert np.array_equal(offspring, parent_a)

    # Now test with parent_b chosen
    monkeypatch.setattr("numpy.random.choice", lambda *a, **k: 1)
    offspring = optimizer._single_point_crossover(parent_a, parent_b)
    assert np.array_equal(offspring, parent_b)


def test_mutation_binary_genetic_algorithm(monkeypatch, optimizer_config) -> None:
    """Test _mutation where only some bits are flipped (custom probs) of the BinaryGeneticAlgorithm class."""
    optimizer_config["optimizer"]["mutation_rate"] = 0.5
    optimizer = BinaryGeneticAlgorithm(
        subpop_size=10,
        n_features=5,
        conf=optimizer_config
    )
    parent = np.array([1, 0, 1, 0, 1])
    # Custom probabilities: flip bits at positions 0, 2, 4
    probs = np.array([0.1, 0.6, 0.2, 0.7, 0.3])
    monkeypatch.setattr("numpy.random.uniform", lambda *a, **k: probs)
    offspring = optimizer._mutation(parent)
    expected = np.array([0, 0, 0, 0, 0])
    assert np.array_equal(offspring, expected)


def test_survivor_selection(optimizer_config) -> None:
    """Test the survivor selection method of the BinaryGeneticAlgorithm class."""
    optimizer = BinaryGeneticAlgorithm(
        subpop_size=10,
        n_features=5,
        conf=optimizer_config
    )
    # Create a population with 10 individuals
    population = np.array([
        [1, 0, 1, 0, 1],  # 0.50
        [0, 1, 0, 1, 0],  # 0.20 (second worst)
        [1, 1, 0, 0, 1],  # 0.80
        [0, 0, 1, 1, 0],  # 0.10 (worst)
        [1, 0, 0, 1, 1],  # 0.60
        [0, 1, 1, 0, 0],  # 0.40
        [1, 1, 1, 0, 0],  # 0.90
        [0, 0, 0, 1, 1],  # 0.30
        [1, 0, 1, 1, 0],  # 0.70
        [0, 1, 0, 0, 1]   # 0.30
    ])
    # Create a fitness array with random values
    fitness = np.array([0.5, 0.2, 0.8, 0.1, 0.6, 0.4, 0.9, 0.3, 0.7, 0.3])
    # Select survivors
    survivors = optimizer._survivor_selection(population, fitness)
    # Expected survivors should be the best individuals based on fitness
    expected_survivors = np.array([
        [1, 0, 1, 0, 1],
        # [0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1],
        # [0, 0, 1, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 1]
    ])
    # Assert that the survivors are the best individuals based on fitness
    assert np.array_equal(survivors, expected_survivors)


def test_tournament_selection(monkeypatch, optimizer_config) -> None:
    """Test _tournament_selection with fixed random choices."""
    optimizer_config["optimizer"]["tournament_sample_size"] = 2
    optimizer = BinaryGeneticAlgorithm(
        subpop_size=5,
        n_features=3,
        conf=optimizer_config
    )
    # Create a subpopulation and fitness
    subpop = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    fitness = [0.5, 0.2, 0.8, 0.1, 0.6]

    # Prepare fixed samples for np.random.choice in the tournament
    # First call returns [2, 4], second call returns [0, 1]
    samples = [
        np.array([2, 4]),  # For first parent
        np.array([0, 1])   # For second parent
    ]
    def fake_choice(a, size, replace):
        return samples.pop(0)
    monkeypatch.setattr("numpy.random.choice", fake_choice)

    # The best in [2,4] is index 2 (fitness 0.8), in [0,1] is index 0 (fitness 0.5)
    parent_a, parent_b = optimizer._tournament_selection(subpop, fitness)
    assert np.array_equal(parent_a, subpop[2])
    assert np.array_equal(parent_b, subpop[0])


def test_binary_genetic_algorithm_generational_evolve(monkeypatch, optimizer_config) -> None:
    """Test evolve method of BinaryGeneticAlgorithm with generational selection."""
    optimizer_config["optimizer"]["tournament_sample_size"] = 2
    subpop_size = 4
    elite_size = 1
    optimizer = BinaryGeneticAlgorithm(
        subpop_size=subpop_size,
        n_features=3,
        conf=optimizer_config
    )
    subpop = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])
    fitness = [0.5, 0.2, 0.8, 0.1]

    # Call counters
    choice_calls = {"count": 0}
    randint_calls = {"count": 0}
    uniform_calls = {"count": 0}

    # Provide enough samples for all tournament selections (2 per offspring, 3 offspring)
    samples = [np.array([2, 0]), np.array([2, 0]), np.array([2, 0]), np.array([2, 0]),
               np.array([2, 0]), np.array([2, 0]), np.array([2, 0]), np.array([2, 0])]
    def fake_choice(a, size, replace):
        choice_calls["count"] += 1
        return samples.pop(0)
    monkeypatch.setattr("numpy.random.choice", fake_choice)

    def fake_randint(*a, **k):
        randint_calls["count"] += 1
        return 1
    monkeypatch.setattr("numpy.random.randint", fake_randint)

    def fake_uniform(low=0, high=1, size=None):
        uniform_calls["count"] += 1
        if size is not None:
            arr = np.ones(size)
            arr[0] = 0.0  # Only first bit < mutation_rate
            return arr
        return 0.0
    monkeypatch.setattr("numpy.random.uniform", fake_uniform)

    # Run evolve
    new_subpop = optimizer.evolve(subpop, fitness)

    # Check shape and type
    assert isinstance(new_subpop, np.ndarray)
    assert new_subpop.shape == subpop.shape

    # Assert the number of calls
    assert choice_calls["count"] == 2 * (subpop_size - elite_size)  # 2 per offspring, 3 offspring
    assert randint_calls["count"] == (subpop_size - elite_size)  # 1 per offspring, 3 offspring
    # uniform_calls will be called for crossover, mutation, and possibly more, so check >= expected
    assert uniform_calls["count"] == 2 * (subpop_size - elite_size)  # 2 per offspring (crossover + mutation), 3 offspring


def test_binary_genetic_algorithm_steady_state_evolve(monkeypatch, optimizer_config) -> None:
    """Test evolve method of BinaryGeneticAlgorithm with steady-state selection."""
    optimizer_config["optimizer"]["selection_method"] = "steady-state"
    subpop_size = 4
    elite_size = 1
    optimizer = BinaryGeneticAlgorithm(
        subpop_size=subpop_size,
        n_features=3,
        conf=optimizer_config
    )
    subpop = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])
    fitness = [0.5, 0.2, 0.8, 0.1]

    # Call counters
    choice_calls = {"count": 0}
    randint_calls = {"count": 0}
    uniform_calls = {"count": 0}

    # Provide enough samples for all tournament selections (2 per offspring, 3 offspring)
    samples = [np.array([2, 0]), np.array([2, 0]), np.array([2, 0]), np.array([2, 0]),
               np.array([2, 0]), np.array([2, 0]), np.array([2, 0]), np.array([2, 0])]
    def fake_choice(a, size, replace):
        choice_calls["count"] += 1
        return samples.pop(0)
    monkeypatch.setattr("numpy.random.choice", fake_choice)

    def fake_randint(*a, **k):
        randint_calls["count"] += 1
        return 1
    monkeypatch.setattr("numpy.random.randint", fake_randint)

    def fake_uniform(low=0, high=1, size=None):
        uniform_calls["count"] += 1
        if size is not None:
            arr = np.ones(size)
            arr[0] = 0.0  # Only first bit < mutation_rate
            return arr
        return 0.0
    monkeypatch.setattr("numpy.random.uniform", fake_uniform)

    # Run evolve
    new_subpop = optimizer.evolve(subpop, fitness)

    # Check shape and type
    assert isinstance(new_subpop, np.ndarray)
    assert new_subpop.shape == subpop.shape

    # Assert the number of calls
    assert choice_calls["count"] == 2  # 2 per offspring, 1 offspring
    assert randint_calls["count"] == 2  # 1 per offspring, 2 offspring
    # uniform_calls will be called for crossover, mutation, and possibly more, so check >= expected
    assert uniform_calls["count"] == 4  # 2 per offspring (crossover + mutation), 2 offspring


