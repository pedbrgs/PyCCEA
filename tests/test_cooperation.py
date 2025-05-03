import numpy as np
from pyccea.cooperation import (
    Collaboration,
    SingleBestCollaboration,
    SingleEliteCollaboration,
    SingleRandomCollaboration
)


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
    current_subpops, fitness, current_best, previous_subpops = generation
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

    # Automatically compute expected collaborators
    expected_collaborators = []
    for i, subpop in enumerate(previous_subpops):
        if i == subpop_idx:
            expected_collaborators.append(current_subpops[i][indiv_idx])
        else:
            # Get index of individual with max fitness in subpop `i`
            best_indiv_idx = np.argmax(fitness[i])
            expected_collaborators.append(previous_subpops[i][best_indiv_idx])

    assert len(collaborators) == len(expected_collaborators)
    for i in range(len(collaborators)):
        assert np.array_equal(collaborators[i], expected_collaborators[i])


def test_get_elite_collaborators(generation) -> None:
    """Test if the elite collaborators are selected correctly."""
    current_subpops, fitness, _, previous_subpops = generation

    # Get collaborators for the 1st (index = 0) individual of the 2nd (index = 1) subpopulation
    subpop_idx = 1
    indiv_idx = 0
    sample_size = 2

    collaborator = SingleEliteCollaboration(sample_size=sample_size)
    collaborators = collaborator.get_collaborators(
        subpop_idx=subpop_idx,
        indiv_idx=indiv_idx,
        previous_subpops=previous_subpops,
        current_subpops=current_subpops,
        fitness=fitness
    )

    # Validate each collaborator is among the elite of the corresponding subpopulation
    collaborator_idx = 0
    for i in range(len(previous_subpops)):
        if i == subpop_idx:
            # Current subpop — ensure it is the same individual (indiv_idx)
            assert np.array_equal(collaborators[collaborator_idx], current_subpops[subpop_idx][indiv_idx])
        else:
            # Other subpops — get indices of top-k (elite) based on fitness
            subpop_fitness = fitness[i]
            top_k_indices = np.argsort(subpop_fitness)[-sample_size:]
            top_k_individuals = [previous_subpops[i][j] for j in top_k_indices]

            # Collaborator must be one of these elite individuals
            assert any(np.array_equal(collaborators[collaborator_idx], elite) for elite in top_k_individuals)

        collaborator_idx += 1


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
    collaborators = collaborator.get_collaborators(
        subpop_idx=subpop_idx,
        indiv_idx=indiv_idx,
        previous_subpops=previous_subpops,
        current_subpops=current_subpops,
    )

    assert len(collaborators) == len(previous_subpops)

    for i in range(len(previous_subpops)):
        if i == subpop_idx:
            # Should be the current individual from current_subpops
            assert np.array_equal(collaborators[i], current_subpops[subpop_idx][indiv_idx])
        else:
            # Should be one of the individuals in previous_subpops[i]
            match_found = any(np.array_equal(collaborators[i], ind) for ind in previous_subpops[i])
            assert match_found, f"Collaborator at index {i} is not in previous_subpops[{i}]"
