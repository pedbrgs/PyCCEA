import numpy as np
from unittest.mock import patch
from pyccea.cooperation.collaboration import Collaboration
from pyccea.cooperation.best import SingleBestCollaboration
from pyccea.cooperation.elite import SingleEliteCollaboration
from pyccea.cooperation.random import SingleRandomCollaboration


def test_build_context_vector_concatenates_correctly() -> None:
    """Test if the context vector is built correctly by concatenating the collaborators."""
    collaborator = Collaboration()

    collaborators = [
        np.array([1, 2]),
        np.array([3, 4]),
        np.array([5, 6])
    ]

    context_vector = collaborator.build_context_vector(collaborators)

    expected_context_vector = np.array([1, 2, 3, 4, 5, 6])
    assert isinstance(context_vector, np.ndarray)
    assert np.array_equal(context_vector, expected_context_vector)


def test_get_best_collaborators(generation) -> None:
    """Test if the best collaborators are selected correctly."""
    current_subpops, fitness, current_best, _ = generation
    # Test with a single subpopulation
    subpop_idx = 0
    indiv_idx = 0
    collaborator = SingleBestCollaboration()
    collaborators = collaborator.get_collaborators(
        subpop_idx=subpop_idx,
        indiv_idx=indiv_idx,
        current_subpops=current_subpops,
        current_best=current_best
    )

    expected_collaborators = [
        np.array([1, 1, 0, 0, 1]),
        np.array([1, 0]),
        np.array([0, 0, 1])
    ]

    assert len(collaborators) == len(expected_collaborators)
    for i in range(len(collaborators)):
        assert np.array_equal(collaborators[i], expected_collaborators[i])


def test_get_elite_collaborators(generation) -> None:
    """Test if the elite collaborators are selected correctly."""
    current_subpops, fitness, _, previous_subpops = generation

    # Define mocked return values for random.choices per subpopulation
    # Assume we return the first individual from subpop[0] and the last one from subpop[2]
    mock_returns = [
        previous_subpops[0][0],
        previous_subpops[2][-1],
    ]

    # Get collaborators for the 1st (index = 0) individual of the 2nd (index = 1) subpopulation
    subpop_idx = 1
    indiv_idx = 0
    sample_size = 3

    collaborator = SingleEliteCollaboration(sample_size=sample_size)
    with patch("random.choices", side_effect=[[mock_returns[0]], [mock_returns[1]]]):
        collaborators = collaborator.get_collaborators(
            subpop_idx=subpop_idx,
            indiv_idx=indiv_idx,
            previous_subpops=previous_subpops,
            current_subpops=current_subpops,
            fitness=fitness
        )

    expected_collaborators = [
        np.array([0, 1, 0, 1, 1]),  # mock_returns[0]
        np.array([1, 1]),  # current_subpops[subpop_idx][indiv_idx]
        np.array([0, 0, 1])  # mockl_returns[1]
    ]

    assert len(collaborators) == len(expected_collaborators)
    for i in range(len(collaborators)):
        assert np.array_equal(collaborators[i], expected_collaborators[i])


def test_get_random_collaborators(generation) -> None:
    """Test if the random collaborators are selected correctly."""
    current_subpops, _, _, previous_subpops = generation

    # Define mocked return values for random.choices per subpopulation
    # Assume we return the last individual from subpop[0] and the first one from subpop[1]
    mock_returns = [
        previous_subpops[0][-1],
        previous_subpops[1][0],
        np.zeros(3)  # Placeholder for the current subpopulation
    ]

    # Get collaborators for the 2nd (index = 1) individual of the 3rd (index = 2) subpopulation
    subpop_idx = 2
    indiv_idx = 1

    collaborator = SingleRandomCollaboration()
    with patch("random.choices", side_effect=[[mock_returns[0]], [mock_returns[1]], [mock_returns[2]]]):
        collaborators = collaborator.get_collaborators(
            subpop_idx=subpop_idx,
            indiv_idx=indiv_idx,
            previous_subpops=previous_subpops,
            current_subpops=current_subpops,
        )

    expected_collaborators = [
        np.array([1, 0, 1, 1, 0]),  # mock_returns[0]
        np.array([1, 0]),  # mock_returns[1]
        np.array([1, 0, 1])  # current_subpops[subpop_idx][indiv_idx]
    ]

    assert len(collaborators) == len(expected_collaborators)
    for i in range(len(collaborators)):
        assert np.array_equal(collaborators[i], expected_collaborators[i])
