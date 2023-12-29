import numpy as np


class AngleModulatedDifferentialEvolution():
    """Angle Modulated Differential Evolution (AMDE) algorithm (rand/1/exp).

    Perform a homomorphous mapping to abstract a problem (defined in binary-valued space) into a
    simpler problem (defined in continuos-valued space).

    Engelbrecht, Andries P., and Gary Pampara. "Binary differential evolution strategies."
    2007 IEEE congress on evolutionary computation. IEEE, (2007).

    Attributes
    ----------
    scaling_factor: float
        The mutation constant. In the literature this is also known as differential weight,
        being denoted by F. It should be in the range [0, 2].
    crossover_probability: float
        The recombination constant, should be in the range [0, 1]. In the literature this is
        also known as the crossover probability. Increasing this value allows a larger number
        of mutants to progress into the next generation, but at the risk of population
        stability.
    bounds: tuple[float, float], default (-1, 1)
        Bounds for coefficients (min, max).
    n_coeffs: int, default 4
        Number of coefficients in the angle modulation function. The AMDE evolves values for
        the four coefficients, a, b, c, and d. The first coefficient represents the horizontal
        shift of the function, the second coefficient represents the maximum frequency of the
        sine function, the third coefficient represents the frequency of the cosine function,
        and the fourth coefficient represents the vertical shift of the function.
    """

    def __init__(self,
                 subpop_size: int,
                 n_features: int,
                 conf: dict):
        """
        Parameters
        ----------
        subpop_size: int
            Number of individuals in the subpopulation.
        n_features: int
            Number of features, that is, decision variables.
        conf: dict
            Configuration parameters of the cooperative coevolutionary algorithm.
        """
        # Configuration parameters
        self.conf = conf
        # Subpopulation size
        if subpop_size < 4:
            raise ValueError(
                "To perform the crossover, it is expected to have at least 3 solutions that are "
                "different from each other and from the target vector. However, the 'subpop_size'"
                f" was set to {subpop_size}."
            )
        self.subpop_size = subpop_size
        # Scaling factor
        self.scaling_factor = self.conf["optimizer"]["scaling_factor"]
        if self.scaling_factor < 0 or self.scaling_factor > 2:
            raise ValueError("Scaling factor should be in the range [0, 2].")
        # Crossover probability
        self.crossover_probability = self.conf["optimizer"]["crossover_probability"]
        if self.crossover_probability < 0 or self.crossover_probability > 1:
            raise ValueError("Crossover probability should be in the range [0, 1].")
        # Number of features
        self.n_features = n_features
        # Number of coefficients in the angle modulation function
        self.n_coeffs = 4
        # Bounds for coefficients
        self.bounds = (-1, 1)

    def _select_solutions(self, pop: np.ndarray, target_vector_idx: float):
        """Select 3 solutions randomly from the population.

        These solutions must be different from each other and different from the target vector.
        """
        possible_solution_idxs = [idx for idx in range(self.subpop_size) if idx != target_vector_idx]
        selected_solution_idxs = np.random.choice(possible_solution_idxs, size=3, replace=False)
        selected_solutions = pop[selected_solution_idxs].copy()
        return selected_solutions

    def _mutation(self, pop: np.ndarray, target_vector_idx: float):
        """Perform difference-vector based mutation."""
        indiv_1, indiv_2, indiv_3 = self._select_solutions(pop, target_vector_idx)
        donor_vector = indiv_1 + self.scaling_factor*(indiv_2 - indiv_3)
        return donor_vector

    def _exponential_crossover(self, target_vector: np.ndarray, donor_vector: np.ndarray):
        """Perform exponential crossover."""
        n = np.random.choice(range(self.n_coeffs))
        trial_vector = np.zeros(self.n_coeffs)
        trial_vector[n] = donor_vector[n].copy()
        indices = [i if i < self.n_coeffs else i-self.n_coeffs for i in range(n+1, n+self.n_coeffs)]
        remaining_idxs = indices.copy()
        for i in indices:
            r = np.random.uniform()
            if r <= self.crossover_probability:
                trial_vector[i] = donor_vector[i].copy()
                remaining_idxs.remove(i)
            else:
                trial_vector[remaining_idxs] = target_vector[remaining_idxs].copy()
                break
        return trial_vector

    def _apply_boundary_constraints(self, trial_vectors: np.ndarray):
        """Apply boundary constraints on trial vectors."""
        trial_vectors = np.clip(trial_vectors, a_min=self.bounds[0], a_max=self.bounds[1])
        return trial_vectors

    def evolve(self, subpop: np.ndarray, fitness: list):
        """Evolve a subpopulation for a single generation.

        Parameters
        ----------
        subpop: np.ndarray
            Individuals of the subpopulation, where each individual is an array of size equal to
            the number of features.
        fitness: list
            Evaluation of all individuals in the subpopulation.

        Returns
        -------
        next_subpop: np.ndarray
            Individuals in the subpopulation of the next generation.
        """

        next_subpop = list()
        # Evolve the subpopulation for a single generation
        for i in range(self.subpop_size):
            donor_vector = self._mutation(pop=subpop, target_vector_idx=i)
            trial_vector = self._exponential_crossover(
                target_vector=subpop[i].copy(),
                donor_vector=donor_vector.copy()
            )
            next_subpop.append(trial_vector)

        next_subpop = self._apply_boundary_constraints(next_subpop)

        return next_subpop
