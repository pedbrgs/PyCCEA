import numpy as np
from pyccea.cooperation.collaboration import Collaboration


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