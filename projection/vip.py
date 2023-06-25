import numpy as np


class VIP:
    """ Variable Importance in Projection (VIP).

    Mehmood, Tahir, et al. "A review of variable selection methods in partial least squares
    regression." Chemometrics and intelligent laboratory systems 118 (2012): 62-69.
    Source: https://github.com/scikit-learn/scikit-learn/issues/7050

    Attributes
    ----------
    p: int
        Number of variables.
    c: int
        Number of components.
    n: int
        Number of observations.
    W: np.ndarray (p x c)
        Projection matrix.
    T: np.ndarray (n x c)
        The transformed training samples (latent components).
    Q: np.ndarray (k x c)
        The loadings of Y.
    importances: np.ndarray (p x 1)
        Importance of each feature based on its contribution to yield the latent space.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model: sklearn model object
            Partial Least Squares regression model. It can be the traditional version (PLS) or the
            Covariance-free version (CIPLS).
        """
        # Projection matrix
        self.W = model.x_weights_.copy()
        # Latent components
        self.T = model.x_scores_.copy()
        # Loadings of Y
        self.Q = model.y_loadings_.copy()

        # Number of features and number of components, respectively
        self.p, self.c = self.W.shape
        # Number of observations
        self.n, _ = self.T.shape
        # Variable Importance in Projection (VIP)
        self.importances = np.zeros((self.p,))

    def compute(self):
        """Calculate feature importances."""
        S = np.diag(self.T.T @ self.T @ self.Q.T @ self.Q).reshape(self.c, -1)
        S_cum = np.sum(S)
        for i in range(self.p):
            w = np.array([(self.W[i,j] / np.linalg.norm(self.W[:,j]))**2 for j in range(self.c)])
            self.importances[i] = np.sqrt(self.p*(S.T @ w)/S_cum)
