import logging
import numpy as np
from ..decomposition.grouping import FeatureGrouping


class RankingFeatureGrouping(FeatureGrouping):
    """
    Decompose the problem (a collection of features) according to a score-based method.
    """

    methods = ["distributed", "elitist"]

    def __init__(self,
                 n_subcomps: int = None,
                 subcomp_sizes: list = list(),
                 scores: np.ndarray = np.empty(0),
                 method: str = None,
                 ascending: bool = True):
        super().__init__(n_subcomps, subcomp_sizes)
        """
        Parameters
        ----------
        n_subcomps: int
            Number of subcomponents, where each subcomponent is a subset of features.
        subcomp_sizes: list
            Number of features in each subcomponent.
        scores: np.ndarray
            Scores relative to the features and that allows sorting them by priority.
        method: str
            Grouping method used to decompose the problem according to scores.
        ascending: bool, default True
            If True, sort in ascending order. Otherwise, sort in descending order.
        """
        # Check if the chosen method is available
        if method not in RankingFeatureGrouping.methods:
            raise AssertionError(
                f"Method {method} was not found. "
                f"The available methods are {', '.join(RankingFeatureGrouping.methods)}."
            )
        self.scores = scores.copy()
        self.method = method
        self.ascending = ascending

    def decompose(self, X: np.ndarray, feature_idxs: np.ndarray = None):
        """
        Divide an n-dimensional problem into m subproblems.

        Parameters
        ----------
        X: np.ndarray
            n-dimensional input data.
        feature_idxs: np.ndarray, default None
            Indexes of features sorted according to the score. It is passed as a parameter if it
            has been previously calculated.

        Returns
        -------
        subcomponents: list
            Subcomponents, where each subcomponent is an array that can be accessed by indexing
            the list.
        feature_idxs: np.ndarray, default None
            Indexes of features sorted according to the score.
        """
        if feature_idxs is None:
            logging.info("Generating feature indexes according to the scores.")
            ranking = np.argsort(self.scores, axis=-1)
            # If lower scores should be ranked better.
            if not self.ascending:
                logging.info("Descending order of scores was chosen.")
                ranking = ranking[::-1].copy()

            if self.method == "elitist":
                # The order of features for decomposition is the ranking itself
                feature_idxs = ranking.copy()
            elif self.method == "distributed":
                # Distributes the top-ranked features evenly among the groups
                self.n_subcomps = self.n_subcomps if self.n_subcomps else len(self.subcomp_sizes)
                feature_idxs = [list() for _ in range(self.n_subcomps)]
                for i, value in enumerate(ranking):
                    feature_idxs[i % self.n_subcomps].append(value)
                feature_idxs = np.concatenate(feature_idxs, axis=0)
        # Shuffle the data features according to the indexes
        X = X[:, feature_idxs].copy()
        # Decompose the problem
        subcomponents = self._get_subcomponents(X=X)

        return subcomponents, feature_idxs
