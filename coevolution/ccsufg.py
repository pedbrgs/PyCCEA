import logging
import numpy as np
from math import log
from coevolution.basic import CCBCGA
from decomposition.ranking import RankingFeatureGrouping


class CCSUFG(CCBCGA):
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
        logging.info(f"Removing features with SU less than {self.su_threshold}...")
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
        """Compute symmetric uncertainty between each feature and class labels."""
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
