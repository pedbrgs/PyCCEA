import numpy as np
from sklearn.base import BaseEstimator


class KernelPLS(BaseEstimator):
    """Kernel Partial Least Squares (KPLS).

    This class implements a kernelized version of Partial Least Squares (PLS), allowing the use of
    nonlinear mappings through kernel functions such as RBF, polynomial, or linear. The algorithm
    extracts latent components in the kernel space that maximize covariance between the predictor
    variables and response(s).
    """

    def __init__(
            self,
            n_components: int = 2,
            kernel: str = "rbf",
            gamma: float = None,
            degree: int = 3,
            coef0: float = 1.0
        ) -> None:
        """
        Initialize the KernelPLS model.

        Parameters
        ----------
        n_components : int, default=2
            Number of latent components to extract.
        kernel : {"rbf", "linear", "poly"} or callable, default="rbf"
            Kernel type to use. Can be one of:
            - "rbf": radial basis function (Gaussian) kernel.
            - "linear": standard dot product kernel.
            - "poly": polynomial kernel.
            - callable: user-defined kernel function with signature (X, Y) -> ndarray.
        gamma : float or None, default=None
            Kernel coefficient for RBF. If None, uses 1 / n_features.
        degree : int, default=3
            Degree of the polynomial kernel.
        coef0 : float, default=1.0
            Independent term in polynomial kernel.
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _rbf(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the RBF (Gaussian) kernel matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            First input data.
        Y : ndarray of shape (n_samples_Y, n_features)
            Second input data.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            RBF kernel matrix.
        """
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        X_norm = np.sum(X**2, axis=1)[:, None]
        Y_norm = np.sum(Y**2, axis=1)[None, :]
        return np.exp(-self.gamma * (X_norm + Y_norm - 2 * X.dot(Y.T)))

    def _linear(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the linear kernel matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            First input data.
        Y : ndarray of shape (n_samples_Y, n_features)
            Second input data.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Linear kernel matrix.
        """
        return X.dot(Y.T)

    def _poly(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the polynomial kernel matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            First input data.
        Y : ndarray of shape (n_samples_Y, n_features)
            Second input data.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Polynomial kernel matrix.
        """
        return (X.dot(Y.T) + self.coef0) ** self.degree

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """Compute the kernel matrix between X and Y using the selected kernel.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            First input data.
        Y : ndarray of shape (n_samples_Y, n_features), optional
            Second input data. If None, uses X.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel matrix.
        """
        if Y is None:
            Y = X
        if self.kernel == "rbf":
            return self._rbf(X, Y)
        elif self.kernel == "linear":
            return self._linear(X, Y)
        elif self.kernel == "poly":
            return self._poly(X, Y)
        elif callable(self.kernel):
            return self.kernel(X, Y)
        else:
            raise ValueError("Invalid kernel. Choose 'rbf', 'linear', 'poly', or provide a callable.")

    def _center_kernel(self, K: np.ndarray) -> np.ndarray:
        """Center a kernel matrix.

        Parameters
        ----------
        K : ndarray of shape (n_samples, n_samples)
            Kernel matrix.

        Returns
        -------
        Kc : ndarray of shape (n_samples, n_samples)
            Centered kernel matrix.
        """
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n

    def _center_cross_kernel(self, Kx: np.ndarray) -> np.ndarray:
        """Center a cross-kernel matrix with respect to training data.

        Parameters
        ----------
        Kx : ndarray of shape (n_samples_new, n_samples_train)
            Cross-kernel matrix between new samples and training samples.

        Returns
        -------
        Kxc : ndarray of shape (n_samples_new, n_samples_train)
            Centered cross-kernel matrix.
        """
        return Kx - np.ones((Kx.shape[0], 1)) @ self.K_mean_cols_[None, :] \
                   - Kx.mean(axis=1)[:, None] + self.K_mean_total_

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit the KernelPLS model to the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input data.
        Y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Training target data.

        Returns
        -------
        self : object
            Fitted model.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n, m = Y.shape
        self.X_fit_ = X.copy()
        self.Y_mean_ = Y.mean(axis=0)
        Y_res = Y - self.Y_mean_

        K = self._compute_kernel(X)
        Kc = self._center_kernel(K)

        self.K_fit_ = K
        self.Kc_ = Kc
        self.K_mean_cols_ = K.mean(axis=0)
        self.K_mean_total_ = K.mean()

        W = np.zeros((n, self.n_components))
        Q = np.zeros((m, self.n_components))

        for h in range(self.n_components):
            w = Kc @ Y_res[:, 0]
            denom = np.sqrt(w @ (Kc @ w))
            if denom < 1e-12:
                break
            w = w / denom
            t = Kc @ w
            q = (Y_res.T @ t) / (t @ t)
            Y_res = Y_res - np.outer(t, q)
            W[:, h] = w
            Q[:, h] = q

        self.W_ = W
        self.Q_ = Q
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict using the KernelPLS model.

        Parameters
        ----------
        X_new : ndarray of shape (n_samples_new, n_features)
            Input samples.

        Returns
        -------
        Y_pred : ndarray of shape (n_samples_new, n_targets)
            Predicted target values.
        """
        Kx = self._compute_kernel(np.asarray(X_new), self.X_fit_)
        Kx_c = self._center_cross_kernel(Kx)
        T_new = Kx_c @ self.W_
        return T_new @ self.Q_.T + self.Y_mean_
