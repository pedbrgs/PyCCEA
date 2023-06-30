import copy
import numpy as np
from sklearn.utils import check_array
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from sklearn.utils.validation import FLOAT_DTYPES


class CIPLS(BaseEstimator):
    """Covariance-free Partial Least Squares (CIPLS).

    Jordao, Artur, et al. "Covariance-free partial least squares: An incremental dimensionality
    reduction method." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer
    Vision (2021).
    Source: https://github.com/arturjordao/IncrementalDimensionalityReduction

    Attributes
    ----------
    n: int
        Number of iterations. It starts at 0 and incrementally goes up to the number of samples
        (n_samples).
    n_features: int
        Number of variables.
    x_weights_: np.ndarray (n_features, n_components)
        Projection matrix.
    x_scores_: np.ndarray (n_samples, n_components)
        The transformed training samples (latent components).
    x_loadings_: np.ndarray (n_features, n_components)
        The loadings of X.
    y_loadings_: np.ndarray (n_targets, n_components)
        The loadings of Y, where n_targets is the number of response variables.
    x_rotations_: np.ndarray (n_components, n_features)
        Transposed and non-normalized projection matrix.
    sum_x: np.ndarray (n_features,)
        The sum of each feature individually across all training samples.
    sum_y: np.ndarray (1,)
        The sum of targets across all training samples.
    """

    def __init__(self, n_components=10, copy=True):
        """
        Parameters
        ----------
        n_components: int or None, default 10
            Number of components to keep. If 'n_components' is None, then its value is set to
            min(n_samples, n_features).
        copy: bool, default True
            If False, X will be overwritten. 'copy=False' can be used to save memory but is unsafe for
            general use.
        """
        self.__name__ = 'Covariance-free Partial Least Squares'
        self.n_components = n_components
        self.n = 0
        self.copy = copy
        self.sum_x = None
        self.sum_y = None
        self.n_features = None
        self.x_rotations_ = None
        self.x_loadings_ = None
        self.y_loadings_ = None
        self.x_scores_ = None
        self.x_weights_ = None

    def normalize(self, x):
        """Scale input vectors individually to unit norm (vector length)."""
        # This function takes a one-dimensional vector x, adds a new dimension to it (because the
        # normalize function from scikit-learn expects a two-dimensional array as input),
        # normalizes it along axis 0, and then returns the resulting normalized one-dimensional
        # vector.
        return normalize(x[:, np.newaxis], axis=0).ravel()

    def fit(self, X, Y):
        """Fit model to data

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Training data.
        Y: np.ndarray (n_samples,) or (n_samples, n_targets)
            Target data.
        """
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        Y = check_array(Y, dtype=FLOAT_DTYPES, copy=self.copy, ensure_2d=False)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if np.unique(Y).shape[0] == 2:
            Y[np.where(Y == 0)[0]] = -1

        n_samples, n_features = X.shape

        if self.n == 0:
            self.x_rotations_ = np.zeros((self.n_components, n_features))
            self.x_loadings_ = np.zeros((n_features, self.n_components))
            self.y_loadings_ = np.zeros((Y.shape[1], self.n_components))
            self.n_features = n_features

        for j in range(0, n_samples):
            self.n = self.n + 1
            u = X[j]
            l = Y[j]

            if self.n == 1:
                self.sum_x = u
                self.sum_y = l
            else:
                # Compute the incremental mean
                old_mean = 1 / (self.n - 1) * self.sum_x
                self.sum_x = self.sum_x + u
                mean_x = 1 / self.n * self.sum_x
                u = u - mean_x
                delta_x = mean_x - old_mean

                # Deflation process
                self.x_rotations_[0] = self.x_rotations_[0] - delta_x * self.sum_y
                self.x_rotations_[0] = self.x_rotations_[0] + (u * l)
                self.sum_y = self.sum_y + l
                t = np.dot(u, self.normalize(self.x_rotations_[0].T))

                self.x_loadings_[:, 0] = self.x_loadings_[:, 0] + (u * t)
                self.y_loadings_[:, 0] = self.y_loadings_[:, 0] + (l * t)

                # Compute the i-th component of the c-dimensional space
                for c in range(1, self.n_components):
                    u -= np.dot(t, self.x_loadings_[:, c - 1])
                    l -= np.dot(t, self.y_loadings_[:, c - 1])
                    # Deflation process
                    self.x_rotations_[c] = self.x_rotations_[c] + (u * l)
                    self.x_loadings_[:, c] = self.x_loadings_[:, c] + (u * t)
                    self.y_loadings_[:, c] = self.y_loadings_[:, c] + (l * t)
                    t = np.dot(u, self.normalize(self.x_rotations_[c].T))

        # Apply the dimension reduction
        self.transform(X=X)

        return self

    def transform(self, X, Y=None):
        """Apply the dimension reduction learned on the training data.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Training data.
        Y: np.ndarray (n_samples,) or (n_samples, n_targets), default None
            Target data.
        """
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)
        # Centralize the data
        mean = 1 / self.n * self.sum_x
        X -= mean
        # Scale each component of the projection matrix
        w_rotation = np.zeros(self.x_rotations_.shape)
        for c in range(0, self.n_components):
            w_rotation[c] = self.normalize(self.x_rotations_[c])

        self.x_weights_ = w_rotation.T.copy()
        self.x_scores_ = np.dot(X, w_rotation.T)
