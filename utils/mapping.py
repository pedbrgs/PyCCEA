import numpy as np


def angle_modulation_function(coeffs: np.ndarray, n_features: int) -> np.ndarray:
    """Homomorphous mapping between binary-valued and continuous-valued space used by the Angle
    Modulated Differential Evolution (AMDE) algorithm.

    Parameters
    ----------
    coeffs: np.ndarray (4,)
        The AMDE evolves values for the four coefficients a, b, c, and d. The first coefficient
        represents the horizontal shift of the function, the second coefficient represents the
        maximum frequency of the sine function, the third coefficient represents the frequency of
        the cosine function, and the fourth coefficient represents the vertical shift of the
        function.
    n_features: int
        Number of variables in the original space (e.g., features in a subcomponent).

    Returns
    -------
    binary_solution: np.ndarray (n_features,)
        Binary solution in the original space.
    """
    a, b, c, d = coeffs
    x = np.linspace(0, 1, n_features)
    trig_function = np.sin(2 * np.pi * (x - a) * b * np.cos(2 * np.pi * (x - a) * c)) + d
    binary_solution = (trig_function > 0).astype(int)
    return binary_solution