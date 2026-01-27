import pytest
import numpy as np
from unittest.mock import MagicMock
from pyccea.fitness import (
    DistanceBasedFitness,
    SubsetSizePenalty,
    WrapperFitnessFunction
)


def test_evaluate_predictive_performance(dummy_dataloader, dummy_classification_evaluator) -> None:
    """Test the evaluate predictive performance method."""
    fitness = WrapperFitnessFunction(evaluator=dummy_classification_evaluator)
    context_vector = np.array([1, 0, 1, 0, 1])
    _ = fitness._evaluate_predictive_performance(
        context_vector=context_vector,
        data=dummy_dataloader
    )
    dummy_classification_evaluator.evaluate.assert_called_once()


def test_subset_size_penalty_weights_sum(dummy_classification_evaluator) -> None:
    """Test that an AssertionError is raised if the sum of weights is not 1 for penalty-based fitness."""
    with pytest.raises(AssertionError, match="must be 1"):
        SubsetSizePenalty(dummy_classification_evaluator, [0.7, 0.4])


def test_subset_size_penalty_weights_length(dummy_classification_evaluator) -> None:
    """Test that an AssertionError is raised if the number of weights is not 2 for penalty-based fitness."""
    with pytest.raises(AssertionError, match="only two components"):
        SubsetSizePenalty(dummy_classification_evaluator, [0.7, 0.3, 0.2])


def test_subset_size_penalty_classification_evaluation(
    dummy_classification_evaluator,
    dummy_dataloader
) -> None:
    """Test the evaluate method of penalty-based fitness in classification."""
    weights = [0.7, 0.3]
    fitness = SubsetSizePenalty(dummy_classification_evaluator, weights)
    
    # Mock the _evaluate_predictive_performance method
    fitness._evaluate_predictive_performance = MagicMock(
        return_value={dummy_classification_evaluator.eval_function: 0.8}
    )

    context_vector = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    fitness_value = fitness.evaluate(context_vector, dummy_dataloader)

    expected_penalty = context_vector.sum() / dummy_dataloader.n_features
    expected_evaluation = 0.8
    expected_fitness = 1 * weights[0] * expected_evaluation - weights[1] * expected_penalty

    assert pytest.approx(fitness_value) == expected_fitness
    fitness._evaluate_predictive_performance.assert_called_once_with(context_vector, dummy_dataloader)


def test_subset_size_penalty_regression_evaluation(
    dummy_regression_evaluator,
    dummy_dataloader
) -> None:
    """Test the evaluate method of penalty-based fitness in regression."""
    weights = [0.7, 0.3]
    fitness = SubsetSizePenalty(dummy_regression_evaluator, weights)
    
    # Mock the _evaluate_predictive_performance method
    fitness._evaluate_predictive_performance = MagicMock(
        return_value={dummy_regression_evaluator.eval_function: 0.8}
    )

    context_vector = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    fitness_value = fitness.evaluate(context_vector, dummy_dataloader)

    expected_penalty = context_vector.sum() / dummy_dataloader.n_features
    expected_evaluation = 0.8
    expected_fitness = -1 * weights[0] * expected_evaluation - weights[1] * expected_penalty

    assert pytest.approx(fitness_value) == expected_fitness
    fitness._evaluate_predictive_performance.assert_called_once_with(context_vector, dummy_dataloader)


def test_distance_based_weights_sum(dummy_classification_evaluator) -> None:
    """Test that an AssertionError is raised if the sum of weights is not 1 for distance-based fitness."""
    with pytest.raises(AssertionError, match="must be 1"):
        DistanceBasedFitness(dummy_classification_evaluator, [0.7, 0.4, 0.2])


def test_distance_based_weights_length(dummy_classification_evaluator) -> None:
    """Test that an AssertionError is raised if the number of weights is not 3 for distance-based fitness."""
    with pytest.raises(AssertionError, match="only three components"):
        DistanceBasedFitness(dummy_classification_evaluator, [0.7, 0.3])


def test_compute_distances_no_estimators(
        dummy_regression_evaluator,
        dummy_dataloader
    ) -> None:
    """Test the distance computation in the distance-based fitness when no estimators are present."""
    dummy_regression_evaluator.estimators = []
    fitness = DistanceBasedFitness(dummy_regression_evaluator, [0.5, 0.25, 0.25])

    dist_same, dist_diff = fitness._compute_distances(dummy_dataloader)

    assert dist_same == 0
    assert dist_diff == 0


def test_compute_distances_with_estimators(
    dummy_regression_evaluator,
    dummy_dataloader
) -> None:
    """Test distance computation when estimators are present."""
    # Create mock training data
    dummy_dataloader.X_train = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
    dummy_dataloader.y_train = np.array([1.0, 2.0, 1.0, 2.0])
    dummy_dataloader.train_size = 4
    dummy_dataloader.n_features = 2

    # Mock the estimator and its kneighbors method
    mock_estimator = MagicMock()
    mock_estimator.kneighbors.return_value = (
        np.array([
            [0.0, 0.1, 0.2],  # distances for sample 0: itself(0.0), neighbor1(0.1), neighbor2(0.2)
            [0.0, 0.3, 0.4],  # sample 1
            [0.0, 0.2, 0.3],  # sample 2
            [0.0, 0.4, 0.5]   # sample 3
        ]),
        np.array([
            [0, 1, 2],        # indices for sample 0 neighbors
            [1, 0, 3],        # sample 1 neighbors
            [2, 0, 3],        # sample 2 neighbors
            [3, 1, 2]         # sample 3 neighbors
        ])
    )

    dummy_regression_evaluator.estimators = [mock_estimator]

    # Initialize fitness
    fitness = DistanceBasedFitness(dummy_regression_evaluator, [0.5, 0.25, 0.25])

    # Run the method
    dist_same, dist_diff = fitness._compute_distances(dummy_dataloader)

    # Assert the expected average distances
    assert pytest.approx(dist_same) == 0.30
    assert pytest.approx(dist_diff) == 0.30


def test_distance_based_classification_evaluation(
    dummy_classification_evaluator,
    dummy_dataloader
) -> None:
    """Test the evaluate method of distance-based fitness in classification."""
    weights = [0.5, 0.25, 0.25]
    dummy_classification_evaluator.estimators = [MagicMock()]
    fitness = DistanceBasedFitness(dummy_classification_evaluator, weights)

    # Mock _evaluate_predictive_performance
    fitness._evaluate_predictive_performance = MagicMock(
        return_value={"accuracy": 0.5}
    )
    # Mock the _compute_distances method
    fitness._compute_distances = MagicMock(
        return_value=(0.5, 0.2)
    )

    context_vector = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    fitness_value = fitness.evaluate(context_vector, dummy_dataloader)

    expected_distance_same = 0.5
    expected_distance_diff = 0.2
    expected_evaluation = 0.5
    expected_fitness = (
        1 * weights[0] * expected_evaluation +
        weights[1] * (expected_distance_diff / np.sqrt(context_vector.sum())) +
        weights[2] * (1 - (expected_distance_same / np.sqrt(context_vector.sum())))
    )

    assert fitness_value == pytest.approx(expected_fitness)
    fitness._compute_distances.assert_called_once_with(dummy_dataloader)


def test_distance_based_regression_evaluation(
    dummy_regression_evaluator,
    dummy_dataloader
) -> None:
    """Test the evaluate method of distance-based fitness in regression."""
    weights = [0.5, 0.25, 0.25]
    dummy_regression_evaluator.estimators = [MagicMock()]
    fitness = DistanceBasedFitness(dummy_regression_evaluator, weights)

    # Mock _evaluate_predictive_performance
    fitness._evaluate_predictive_performance = MagicMock(
        return_value={"mse": 0.5}
    )
    # Mock the _compute_distances method
    fitness._compute_distances = MagicMock(
        return_value=(0.5, 0.2)
    )

    context_vector = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    fitness_value = fitness.evaluate(context_vector, dummy_dataloader)

    expected_distance_same = 0.5
    expected_distance_diff = 0.2
    expected_evaluation = 0.5
    expected_fitness = (
        -1 * weights[0] * expected_evaluation +
        weights[1] * (expected_distance_diff / np.sqrt(context_vector.sum())) +
        weights[2] * (1 - (expected_distance_same / np.sqrt(context_vector.sum())))
    )

    assert fitness_value == pytest.approx(expected_fitness)
    fitness._compute_distances.assert_called_once_with(dummy_dataloader)


def test_subset_size_penalty_clone(dummy_classification_evaluator) -> None:
    """Test clone returns a new penalty fitness with a cloned evaluator."""
    evaluator_clone = MagicMock()
    dummy_classification_evaluator.clone = MagicMock(return_value=evaluator_clone)
    fitness = SubsetSizePenalty(dummy_classification_evaluator, [0.7, 0.3])

    fitness_clone = fitness.clone()

    assert fitness_clone is not fitness
    assert isinstance(fitness_clone, SubsetSizePenalty)
    assert fitness_clone.w1 == fitness.w1
    assert fitness_clone.w2 == fitness.w2
    assert fitness_clone.evaluator is evaluator_clone
    dummy_classification_evaluator.clone.assert_called_once()


def test_distance_based_clone(dummy_regression_evaluator) -> None:
    """Test clone returns a new distance-based fitness with a cloned evaluator."""
    evaluator_clone = MagicMock()
    dummy_regression_evaluator.clone = MagicMock(return_value=evaluator_clone)
    fitness = DistanceBasedFitness(dummy_regression_evaluator, [0.5, 0.25, 0.25])

    fitness_clone = fitness.clone()

    assert fitness_clone is not fitness
    assert isinstance(fitness_clone, DistanceBasedFitness)
    assert fitness_clone.w1 == fitness.w1
    assert fitness_clone.w2 == fitness.w2
    assert fitness_clone.w3 == fitness.w3
    assert fitness_clone.evaluator is evaluator_clone
    dummy_regression_evaluator.clone.assert_called_once()