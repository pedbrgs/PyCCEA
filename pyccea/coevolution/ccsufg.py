import gc
import copy
import logging
import numpy as np
from math import log
from tqdm import tqdm
from coevolution.ccga import CCGA
from decomposition.ranking import RankingFeatureGrouping


class CCSUFG(CCGA):
    """Cooperative Co-Evolutionary Algorithm with Symmetric Uncertainty Feature Grouping (CCSUFG).

    Symmetric Uncertainty computation based on:
    Li, Jundong, et al. "Feature selection: A data perspective." ACM Computing Surveys (CSUR) 50.6
    (2017): 1-45.

    Decomposition strategy proposed on:
    Song, Xian-Fang, et al. "Variable-size cooperative coevolutionary particle swarm optimization
    for feature selection on high-dimensional data." IEEE Transactions on Evolutionary Computation 
    24.5 (2020): 882-895.
    """

    def _hist(self, samples: list):
        """Build a histogram from a list of samples.

        Parameters
        ----------
        samples : list
            Samples to build a histogram.

        Returns
        -------
        histogram : iterator
            Histogram made from the samples.
        """
        d = dict()
        for s in samples:
            d[s] = d.get(s, 0) + 1
        histogram = map(lambda z: float(z)/len(samples), d.values())

        return histogram

    def _elog(self, x: float) -> float:
        """Compute the product of the logarithm of a value with the value itself.

        For entropy, 0 * log(0) is 0. However, log(0) raises an error. This function handles this
        specific case.

        Parameters
        ----------
        x : float
            Target value.

        Returns
        -------
         : float
            Product of the logarithm of the input value with the value itself.
        """
        if x <= 0. or x >= 1.:
            return 0
        else:
            return x*log(x)

    def _entropy_from_probs(self, probs: list, base: int = 2) -> float:
        """Compute entropy from a normalized list of probabilities of discrete outcomes.

        Parameters
        ----------
        probs : list
            Probabilities of discrete outcomes.
        base : int, default 2
            Logarithm base.

        Returns
        -------
         : float
            Entropy.
        """
        return -sum(map(self._elog, probs))/log(base)

    def _compute_discrete_entropy(self, samples: list, base: int = 2) -> float:
        """Compute discrete entropy given a list of samples (any hashable object).

        Parameters
        ----------
        samples : list
            Samples to calculate discrete entropy.
        base : int, default 2
            Logarithm base.

        Returns
        -------
         : float
            Discrete entropy.
        """
        return self._entropy_from_probs(self._hist(samples), base=base)

    def _midd(self, X: list, Y: list) -> float:
        """Compute discrete mutual information given a list of samples (any hashable object).

        Parameters
        ----------
        X : np.ndarray (n_examples,)
            First random variable.
        Y : np.ndarray (n_examples,)
            Second random variable.

        Returns
        -------
         : float
            Discrete mutual information.
        """
        return (
            -self._compute_discrete_entropy(list(zip(X, Y)))+
            self._compute_discrete_entropy(X)+self._compute_discrete_entropy(Y)
        )

    def _compute_conditional_entropy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute the conditional entropy (CE).

        The CE value between two variables, X and Y, can be defined as follows:
            CE(X,Y) = H(X) - I(X|Y),
        which measures the uncertainty of X when Y is given.

        Parameters
        ----------
        X : np.ndarray (n_examples,)
            First random variable.
        Y : np.ndarray (n_examples,)
            Second random variable.

        Returns
        -------
        conditional_entropy : float
            Conditional entropy of X and Y.
        """
        conditional_entropy = self._compute_discrete_entropy(X) - self._midd(X, Y)
        return conditional_entropy

    def _compute_information_gain(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute the information gain (IG).

        The IG value between two random variables, X and Y, can be defined as follows:
            IG(X, Y) = H(X) - H(X|Y),
        which represents the decrease degree of uncertainty of X when Y is known.

        Parameters
        ----------
        X : np.ndarray (n_examples,)
            First random variable.
        Y : np.ndarray (n_examples,)
            Second random variable.

        Returns
        -------
        information_gain : float
            Information gain between the two inputs.
        """
        information_gain = (
            self._compute_discrete_entropy(X) -
            self._compute_conditional_entropy(X, Y)
        )

        return information_gain

    def _compute_symmetric_uncertainty(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute the symmetric uncertainty (SU).

        It is considered a scale-insensitive correlation measure as it normalizes mutual
        information. The SU value between two random variables, X and Y, can be defined as
        follows:
            SU(X, Y) = 2*IG(X|Y)/(H(X)+H(Y)),
        where H(X) is the entropy of X, IG(X|Y) refers to the information gain, and H(X|Y) is the
        conditional entropy.

        Parameters
        ----------
        X : np.ndarray (n_examples,)
            First random variable.
        Y : np.ndarray (n_examples,)
            Second random variable.

        Returns
        -------
        symmetric_uncertainty : float
            Symmetric uncertainty between the two inputs.
        """
        # Calculate information gain of X and Y
        t1 = self._compute_information_gain(X, Y)
        # Calculate entropy of X
        t2 = self._compute_discrete_entropy(X)
        # Calculate entropy of Y
        t3 = self._compute_discrete_entropy(Y)
        # Compute symmetric uncertainty
        symmetric_uncertainty = 2.0*t1/(t2+t3)

        return symmetric_uncertainty

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

    def _compute_variable_importances(self) -> list:
        """Compute symmetric uncertainty between each feature and class labels.

        Returns
        -------
        importances : np.ndarray
            Importance of each feature based on symmetric uncertainty.
        """
        importances = list()

        for i in range(self.data.n_features):
            importances.append(
                self._compute_symmetric_uncertainty(
                    self.data.X_train[:, i],
                    self.data.y_train
                )
            )

        return np.array(importances)

    def _init_decomposer(self) -> None:
        """Instantiate feature grouping method."""
        # Compute feature importances
        importances = self._compute_variable_importances()
        # Compute symmetric uncertainty threshold
        # The threshold is 10% of the maximal feature importance of the current dataset
        self.su_threshold = 0.10*np.max(importances)
        # Remove irrelevant or weaken relevant features
        importances = self._remove_unimportant_features(importances)

        # Ranking feature grouping using variable importances as scores
        self.decomposer = RankingFeatureGrouping(
            n_subcomps=self.n_subcomps,
            subcomp_sizes=self.subcomp_sizes,
            scores=importances,
            method="elitist",
            ascending=False
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
