import numpy as np
from ..utils.datasets import DataLoader
from ..evaluation.wrapper import WrapperEvaluation
from ..fitness.function import WrapperFitnessFunction


class DistanceBasedFitness(WrapperFitnessFunction):
    """
    Objective function that maximizes balanced accuracy based on a k-nearest neighbors classifier.

    The fitness function is designed as a three-objective optimization, aimed at achieving a
    balance between maximizing balanced accuracy while simultaneously minimizing the average
    distance between instances sharing the same label and maximizing the average distance between
    instances with different labels.

    Firouznia, Marjan, Pietro Ruiu, and Giuseppe A. Trunfio. "Adaptive cooperative coevolutionary
    differential evolution for parallel feature selection in high-dimensional datasets."
    The Journal of Supercomputing (2023): 1-30.

    Attributes
    ----------
    w1: float
        Predictive performance weight.
    w2: float
        Weight of the complement of the average distance between instances and their neighbors
        with the same class.
    w3: float
        Weight of the average distance between instances and their neighbors of different classes.
    """

    def __init__(self, evaluator: WrapperEvaluation, weights: list):
        super().__init__(evaluator)
        # Check the number of weights
        if len(weights) != 3:
            raise AssertionError(
                f"'{DistanceBasedFitness.__name__}' fitness function has only three components. "
                "Therefore, it requires only three weights."
            )
        # Check the sum of the weights
        if sum(weights) != 1:
            raise AssertionError(
                f"The sum of weights is {sum(weights)} but must be 1."
            )
        self.w1 = weights[0]
        self.w2 = weights[1]
        self.w3 = weights[2]

    def _compute_distances(self, data: DataLoader):
        """
        Calculate the average distances between instances and their neighbors with the same class
        and their neighbors with different class.
        """
        avg_distances_same_label = list()
        avg_distances_diff_label = list()

        # If no feature is selected, no estimator has been trained
        if self.evaluator.estimators:
            return 0, 0

        # For all estimator
        for estimator in self.evaluator.estimators:
            # Get indices and distances between neighbors
            distances, indices = estimator.kneighbors(
                X=data.X_train,
                n_neighbors=estimator.n_neighbors+1, # Neighbors other than the i-th instance itself
                return_distance=True
            )
            sum_distances_same_label = list()
            sum_distances_diff_label = list()
            n_neighbors_same_label = 0
            n_neighbors_diff_label = 0
            # For all training instance
            for i in range(data.train_size):
                # Neighbors with the same label
                same_label_indices = np.where(data.y_train[indices[i]] == data.y_train[i])[0]
                sum_ith_distance_same_label = np.sum(distances[i][same_label_indices])
                n_neighbors_same_label += len(same_label_indices) - 1 # Neighbors other than the i-th instance itself
                sum_distances_same_label.append(sum_ith_distance_same_label)
                # Neighbors with different labels
                diff_label_indices = np.where(data.y_train[indices[i]] != data.y_train[i])[0]
                sum_ith_distance_diff_label = np.sum(distances[i][diff_label_indices])
                n_neighbors_diff_label += len(diff_label_indices) - 1
                sum_distances_diff_label.append(sum_ith_distance_diff_label)
            # Average distances for each estimator
            avg_distances_same_label.append(np.sum(sum_distances_same_label)/n_neighbors_same_label)
            avg_distances_diff_label.append(np.sum(sum_distances_diff_label)/n_neighbors_diff_label)
        # Mean average distances calculated across all estimators
        mean_avg_distance_same_label = np.mean(avg_distances_same_label)
        mean_avg_distance_diff_label = np.mean(avg_distances_diff_label)

        return mean_avg_distance_same_label, mean_avg_distance_diff_label

    def evaluate(self, context_vector: np.ndarray, data: DataLoader):
        """
        Evaluate the given context vector using the fitness function.

        Parameters
        ----------
        context_vector: np.ndarray
            Solution of the complete problem.
        data: DataLoader
            Container with process data and training and test sets.

        Returns
        -------
        fitness: float
            Quality of the context vector.
        """
        sqrt_n_selected_features = np.sqrt(context_vector.sum())
        evaluations = self._evaluate_predictive_performance(context_vector, data)
        evaluation = evaluations[self.evaluator.eval_function]
        mean_avg_distance_same_label, mean_avg_distance_diff_label = self._compute_distances(data)
        fitness = (
            self.w1*evaluation +
            self.w2*(mean_avg_distance_diff_label/sqrt_n_selected_features) +
            self.w3*(1-(mean_avg_distance_same_label/sqrt_n_selected_features))
        )
        return fitness
