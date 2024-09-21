import numpy as np


class VIP:
    """ Variable Importance in Projection (VIP).

    Mehmood, Tahir, et al. "A review of variable selection methods in partial least squares
    regression." Chemometrics and intelligent laboratory systems 118 (2012): 62-69.
    Source: https://github.com/scikit-learn/scikit-learn/issues/7050

    Attributes
    ----------
    n_features : int
        Number of variables.
    n_components : int
        Number of components.
    x_rotations_ : np.ndarray (n_features, n_components)
        Projection matrix used to transform X.
    x_scores_ : np.ndarray (n_samples, n_components)
        The transformed training samples (latent components).
    y_loadings_ : np.ndarray (n_targets, n_components)
        The loadings of Y.
    importances : np.ndarray (n_features,)
        Importance of each feature based on its contribution to yield the latent space.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : sklearn model object
            Partial Least Squares regression model. It can be the traditional version (PLS) or the
            Covariance-free version (CIPLS).
        """
        # Projection matrix
        self.x_rotations_ = model.x_rotations_.copy()
        # Latent components
        self.x_scores_ = model.x_scores_.copy()
        # Loadings of Y
        self.y_loadings_ = model.y_loadings_.copy()
        # Number of features and number of components, respectively
        self.n_features, self.n_components = self.x_rotations_.shape

    def compute(self):
        """Calculate feature importances."""
        # Sum of squares explained by each component (n_components,)
        sum_of_squares = np.diag(self.x_scores_.T @ self.x_scores_ @ self.y_loadings_.T @ self.y_loadings_)
        # Reshape array (n_components, 1)
        sum_of_squares = sum_of_squares.reshape(self.n_components, -1)
        # Cumulative sum of squares
        cum_sum_of_squares = np.sum(sum_of_squares)
        # Projection matrix norm
        weight_norm = np.linalg.norm(self.x_rotations_, axis=0)
        # Normalized weights
        weights = (self.x_rotations_ / np.expand_dims(weight_norm, axis=0)) ** 2
        # Variable Importances in Projection (VIP)
        squared_importances = self.n_features * (weights @ sum_of_squares).ravel() / cum_sum_of_squares
        # To avoid "RuntimeWarning: invalid value encountered in sqrt"
        squared_importances[squared_importances < 0] = np.nan
        self.importances = np.sqrt(squared_importances)
        self.importances[np.where(np.isnan(self.importances))[0]] = -999