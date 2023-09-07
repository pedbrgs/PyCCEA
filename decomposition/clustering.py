import numpy as np
from decomposition.grouping import FeatureGrouping


class ClusteringFeatureGrouping(FeatureGrouping):
    """
    Decompose the problem (a collection of features) according to a clustering.
    """

    def __init__(self, clusters: np.ndarray = np.empty(0),):
        """
        Parameters
        ----------
        clusters: np.ndarray
            Index of the cluster each feature belongs to.
        """
        self.clusters = clusters.copy()
        self.n_subcomps = len(np.unique(clusters))

    def decompose(self, X: np.ndarray, feature_idxs: np.ndarray = None):
        """
        Divide an n-dimensional problem into m subproblems.

        Parameters
        ----------
        X: np.ndarray
            n-dimensional input data.
        feature_idxs: np.ndarray, default None
            Feature indexes sorted according to clustering. It is passed as a parameter if it has
            been previously generated.

        Returns
        -------
        subcomponents: list
            Subcomponents, where each subcomponent is an array that can be accessed by indexing
            the list.
        feature_idxs: np.ndarray, default None
            Feature indexes sorted according to clustering. For example, if the first
            subpopulation has size x, the first x elements of this list will be the features of
            the first subcomponent and so on.
        """
        if feature_idxs is None:
            feature_idxs = list()
            self.subcomp_sizes = list()
            for cluster_id in range(self.n_subcomps):
                cluster_features = np.where(self.clusters == cluster_id)[0]
                self.subcomp_sizes.append(len(cluster_features))
                feature_idxs.extend(cluster_features)
            feature_idxs = np.array(feature_idxs)

        # Shuffle the data features according to the indexes
        X = X[:, feature_idxs].copy()
        # Decompose the problem
        subcomponents = self._get_subcomponents(X=X)

        return subcomponents, feature_idxs
