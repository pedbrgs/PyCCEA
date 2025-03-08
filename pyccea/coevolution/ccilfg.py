import gc
import copy
import logging
import numpy as np
from math import log
from tqdm import tqdm
from scipy.stats import entropy
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from ..coevolution.ccga import CCGA
from ..decomposition.dummy import DummyFeatureGrouping


class CCILFG(CCGA):
    """Cooperative Co-Evolutionary Algorithm with Interaction Learning Feature Grouping (CCILFG).

    Decomposition strategy proposed on:
    Hou, Yaqing, et al. "A correlation-guided cooperative coevolutionary method for feature
    selection via interaction learning-based space division." Swarm and Evolutionary
    Computation 93 (2025): 101846.
    """

    def _symmetric_uncertainty(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the symmetric uncertainty (SU).

        It is considered a scale-insensitive correlation measure as it normalizes mutual
        information. The SU value between two random variables, X and Y, can be defined as
        follows:
            SU(X, Y) = 2*IG(X|Y)/(H(X)+H(Y)),
        where H(X) is the entropy of X, IG(X|Y) refers to the information gain, and H(X|Y) is the
        conditional entropy.

        Parameters
        ----------
        x : np.ndarray (n_examples,)
            First random variable.
        y : np.ndarray (n_examples,)
            Second random variable.

        Returns
        -------
        su : float
            Symmetric uncertainty between the two inputs.
        """
        # Compute mutual information
        mi = mutual_info_score(x, y)
        # Compute entropy of each variable
        hx = entropy(np.bincount(x))  # Entropy of x
        hy = entropy(np.bincount(y))  # Entropy of y
        # Symmetric Uncertainty formula
        su = 2.0 * mi / (hx + hy) if hx + hy > 0 else 0.0
        return su

    def _compute_symmetric_uncertainty_class(
            self,
            X: np.ndarray,
            y: np.ndarray,
            n_bins: int = 10
        ) -> np.ndarray:
        """Compute symmetric uncertainty between each feature in X and the class labels y.
        
        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)
            Input matrix (features).
        y (np.ndarray (n_samples,)
            Target vector (class labels).
        n_bins : int
            Number of bins for discretization of features.
        
        Returns
        -------
        su_values : np.ndarray (n_features,)
            Array of SU values between each feature and the class labels.
        """
        # Discretize features
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
        X_discretized = discretizer.fit_transform(X).astype(int)

        # Discretize class labels if necessary
        if not np.issubdtype(y.dtype, np.integer):
            y = np.digitize(y, bins=np.histogram_bin_edges(y, bins=n_bins))

        # Compute SU for each feature with the class labels
        su_values = np.array(
            [
                self._symmetric_uncertainty(X_discretized[:, i], y)
                for i in range(X.shape[1])
            ]
        )
        return su_values

    def _compute_symmetric_uncertainty_features_parallel(
            self,
            X: np.ndarray,
            n_bins: int = 10
        ) -> np.ndarray:
        """Optimized computation of symmetric uncertainty matrix."""
        # Discretize features
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
        X_discretized = discretizer.fit_transform(X).astype(int)

        n_features = X.shape[1]
        su_matrix = np.zeros((n_features, n_features))

        # Fill diagonal with 1s (SU of a feature with itself)
        np.fill_diagonal(su_matrix, 1.0)

        def compute_su(i, j):
            su = self._symmetric_uncertainty(X_discretized[:, i], X_discretized[:, j])
            return (i, j, su)

        # Compute SU for upper triangular matrix only (i <= j)
        task_count = (n_features * (n_features - 1)) // 2  # Number of unique pairs (i < j)
        with tqdm_joblib(tqdm(desc="Computing SU between features", total=task_count)) as progress_bar:
            results = Parallel(n_jobs=-1)(
                delayed(compute_su)(i, j)
                for i in range(n_features) for j in range(i + 1, n_features)
            )

        # Populate the symmetric matrix
        for i, j, su in results:
            su_matrix[i, j] = su
            su_matrix[j, i] = su  # Exploit symmetry

        return su_matrix

    def _compute_symmetric_uncertainty_features(
            self,
            X: np.ndarray,
            n_bins: int = 10
        ) -> np.ndarray:
        """Compute a symmetric uncertainty matrix for all features in dataset X.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)
            Input matrix (features).
        n_bins : int
            Number of bins for discretization of features.

        Returns
        -------
        su_matrix : np.ndarray (n_features, n_features)
            Symmetric uncertainty matrix between all features in X.
        """
        # Discretize features
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
        X_discretized = discretizer.fit_transform(X).astype(int)

        n_features = X.shape[1]
        su_matrix = np.zeros((n_features, n_features))
        # Fill diagonal with 1s (SU of a feature with itself)
        np.fill_diagonal(su_matrix, 1.0)

        for i in tqdm(range(n_features), desc="Computing SU between features"):
            for j in range(i + 1, n_features):
                su = self._symmetric_uncertainty(X_discretized[:, i], X_discretized[:, j])
                su_matrix[i, j] = su
                su_matrix[j, i] = su  # Symmetry: SU(i, j) == SU(j, i)
        return su_matrix

    def _remove_unimportant_features(self, importances: np.ndarray) -> np.ndarray:
        """Remove irrelevant or weaken features from folds and subsets.

        Parameters
        ----------
        importances : np.ndarray (n_features,)
            Importance of each feature based on symmetric uncertainty.

        Returns
        -------
        importances : np.ndarray
            Importance of the remaining features.
        """
        logging.info(f"Removing features with SU less than {round(self.su_threshold, 4)}...")
        features_to_keep = importances >= self.su_threshold
        self.removed_features = np.where(features_to_keep == False)[0]
        logging.info(f"{len(self.removed_features)} features were removed.")

        # Removing features from subsets and folds
        self.data.X_train = self.data.X_train[:, features_to_keep].copy()
        self.data.X_test = self.data.X_test[:, features_to_keep].copy()
        for k in range(self.data.kfolds):
            self.data.train_folds[k][0] = self.data.train_folds[k][0][:, features_to_keep].copy()
            self.data.val_folds[k][0] = self.data.val_folds[k][0][:, features_to_keep].copy()

        # Importance of the remaining features
        importances = importances[features_to_keep].copy()

        return importances

    def _find_knee_point(self, x: np.ndarray, y: np.ndarray) -> float:
        """Find the knee point in a curve.

        The knee point is the point where the curve has the maximum curvature.

        Parameters
        ----------
        x : np.ndarray
            x-axis values.
        y : np.ndarray
            y-axis values.

        Returns
        -------
        knee_point : float
            Knee point in the curve.
        """
        # Define the line connecting the first and last points
        x0, y0 = x[0], y[0]
        xn, yn = x[-1], y[-1]
        ang_coef = (yn - y0) / (xn - x0)  # Slope
        lin_coef = y0 - ang_coef* x0           # Intercept

        # Calculate distances from the curve points to the line
        distances = np.abs(ang_coef * x - y + lin_coef) / np.sqrt(ang_coef**2 + 1)
        # Find the index of the maximum distance (knee point)
        knee_index = np.argmax(distances)
        # Return the knee point coordinates and gamma
        gamma = y[knee_index]

        return gamma

    def _space_division_based_on_interaction_learning(
            self,
            su_matrix: np.ndarray,
            promising_features: np.ndarray,
            K: int = 3
        ) -> np.ndarray:
        """Space division based on interaction learning.

        Parameters
        ----------
        su_matrix : np.ndarray (n_features, n_features)
            Symmetric uncertainty matrix between all features in X.
        promising_features : np.ndarray (n_features,)
            Indices of promising features.
        K : int
            Number of neighbors to consider for each feature.

        Returns
        -------
        ranking : np.ndarray (n_features,)
            Ranking of features based on the space division.
        """
        # Initialize subspaces
        self.boundary_subspace = []  # B
        self.high_correlation_subspace = []  # H
        self.low_correlation_subspace = []  # L

        # Classify each feature based on neighbors
        for i, feature_idx in tqdm(enumerate(range(su_matrix.shape[0])), desc="Classifying features based on neighbors"):
            # Get SU_F values for this feature with all others
            su_values = su_matrix[i]

            # Get indices of top K neighbors (excluding itself)
            neighbor_indices = np.argsort(su_values)[-K - 1:-1][::-1]

            # Determine the group of the feature and its neighbors
            feature_group = "Fp" if feature_idx in promising_features else "Fr"
            neighbor_groups = [
                "Fp" if neighbor in promising_features else "Fr"
                for neighbor in neighbor_indices
            ]

            # Check if all neighbors belong to the same group
            if all(group == feature_group for group in neighbor_groups):
                if feature_group == "Fp":
                    self.high_correlation_subspace.append(feature_idx)  # Add to H
                else:
                    self.low_correlation_subspace.append(feature_idx)  # Add to L
            else:
                self.boundary_subspace.append(feature_idx)  # Add to B
        logging.info(f"Number of features in B: {len(self.boundary_subspace)}")
        logging.info(f"Number of features in H: {len(self.high_correlation_subspace)}")
        logging.info(f"Number of features in L: {len(self.low_correlation_subspace)}")
        ranking = np.array(
            self.high_correlation_subspace +
            self.low_correlation_subspace +
            self.boundary_subspace
        )
        return ranking

    def _init_decomposer(self) -> None:
        """Instantiate feature grouping method."""
        # Compute feature importances
        SU_c = self._compute_symmetric_uncertainty_class(self.data.X_train, self.data.y_train)
        # Compute symmetric uncertainty threshold
        # The threshold is 5% of the maximal feature importance of the current dataset
        self.su_threshold = 0.05*np.max(SU_c)
        # Remove irrelevant or weaken relevant features
        SU_c = self._remove_unimportant_features(SU_c)
        self.importances = SU_c.copy()

        # Find knee point
        sorted_indices = np.argsort(SU_c)[::-1]
        sorted_SU_c = np.array(SU_c)[sorted_indices]
        ranking = np.arange(1, len(sorted_SU_c) + 1)
        gamma = self._find_knee_point(ranking, sorted_SU_c)

        # Select promising and remaining features
        promising_features = np.where(SU_c >= gamma)[0]
        remaining_features = np.where(SU_c < gamma)[0]

        # Compute symmetric uncertainty matrix for all features
        if self.data.n_features > 100000:
            logging.info("Computing SU matrix in parallel...")
            SU_f = self._compute_symmetric_uncertainty_features_parallel(self.data.X_train)
        else:
            logging.info("Computing SU matrix sequentially...")
            SU_f = self._compute_symmetric_uncertainty_features(self.data.X_train)
        # Space division based on interaction learning
        ranking = self._space_division_based_on_interaction_learning(SU_f, promising_features)
    
        # Set the number of subcomponents and their sizes
        self.n_subcomps = 3
        self.subcomp_sizes = [
            len(self.high_correlation_subspace),
            len(self.low_correlation_subspace),
            len(self.boundary_subspace)
        ]

        # Ranking feature grouping using variable importances as scores
        self.decomposer = DummyFeatureGrouping(
            subcomp_sizes=self.subcomp_sizes,
            feature_idxs=ranking
        )

    def optimize(self) -> None:
        """Solve the feature selection problem through optimization."""
        # Decompose problem
        self._problem_decomposition()
        # Initialize subpopulations
        self._init_subpopulations()
        # Instantiate optimizers
        self._init_optimizers()

        # Get the best individual and context vector from each subpopulation
        self.current_best = self._get_best_individuals(
            subpops=self.subpops,
            fitness=self.fitness,
            context_vectors=self.context_vectors
        )
        # Select the globally best context vector
        self.best_context_vector, self.best_fitness = self._get_global_best()
        self.best_context_vectors.append(self.best_context_vector.copy())
        # Save the order of features considered in the random feature grouping
        self.best_feature_idxs = self.feature_idxs.copy()

        # Set the number of generations counter
        n_gen = 0
        # Number of generations that the best fitness has not improved
        stagnation_counter = 0
        # Initialize the optimization progress bar
        progress_bar = tqdm(total=self.conf["coevolution"]["max_gen"],
                            desc="Generations",
                            leave=False)

        # Iterate up to the maximum number of generations
        while n_gen <= self.conf["coevolution"]["max_gen"]:
            # Append current best fitness
            self.convergence_curve.append(self.best_fitness)

            # Evolve each subpopulation using a genetic algorithm
            current_subpops = list()
            for i in range(self.n_subcomps):
                current_subpop = self.optimizers[i].evolve(
                    subpop=self.subpops[i],
                    fitness=self.fitness[i]
                )
                current_subpops.append(current_subpop)

            # Evaluate each individual of the evolved subpopulations
            current_fitness = list()
            current_context_vectors = list()
            for i in range(self.n_subcomps):
                current_fitness.append(list())
                current_context_vectors.append(list())
                # Use best individuals from the previous generation (`self.current_best`) as
                # collaborators for each individual in the current generation after evolve
                # (`current_subpops`)
                for j in range(self.subpop_sizes[i]):
                    collaborators = self.best_collaborator.get_collaborators(
                        subpop_idx=i,
                        indiv_idx=j,
                        current_subpops=current_subpops,
                        current_best=self.current_best
                    )
                    context_vector = self.best_collaborator.build_context_vector(collaborators)
                    # Update the context vector
                    current_context_vectors[i].append(context_vector.copy())
                    # Update fitness
                    current_fitness[i].append(self.fitness_function.evaluate(context_vector, self.data))
            # Update subpopulations, context vectors and evaluations
            self.subpops = copy.deepcopy(current_subpops)
            self.fitness = copy.deepcopy(current_fitness)
            self.context_vectors = copy.deepcopy(current_context_vectors)
            del current_subpops, current_fitness, current_context_vectors
            gc.collect()

            # Get the best individual and context vector from each subpopulation
            self.current_best = self._get_best_individuals(
                subpops=self.subpops,
                fitness=self.fitness,
                context_vectors=self.context_vectors
            )
            # Select the globally best context vector
            best_context_vector, best_fitness = self._get_global_best()
            # Update best context vector
            if self.best_fitness < best_fitness:
                # Reset stagnation counter because best fitness has improved
                stagnation_counter = 0
                # Enable logger if specified
                logging.getLogger().disabled = False if self.verbose else True
                # Current fitness
                current_best_fitness = round(self.best_fitness, 4)
                # New fitness
                new_best_fitness = round(best_fitness, 4)
                # Show improvement
                logging.info(
                    f"\nUpdate fitness from {current_best_fitness} to {new_best_fitness}.\n"
                )
                # Update best context vector
                self.best_context_vector = best_context_vector.copy()
                self.best_context_vectors.append(self.best_context_vector.copy())
                # Update best fitness
                self.best_fitness = best_fitness
            else:
                # Increase stagnation counter because best fitness has not improved
                stagnation_counter += 1
                # Checks whether the optimization has been stagnant for a long time
                if stagnation_counter >= self.conf["coevolution"]["max_gen_without_improvement"]:
                    # Enable logger
                    logging.getLogger().disabled = False
                    logging.info(
                        "\nEarly stopping because fitness has been stagnant for "
                        f"{stagnation_counter} generations in a row."
                    )
                    break
            # Increase number of generations
            n_gen += 1
            # Update progress bar
            progress_bar.update(1)
        # Close progress bar after optimization
        progress_bar.close()
