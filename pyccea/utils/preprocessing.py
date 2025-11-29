import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Winsoriser(BaseEstimator, TransformerMixin):
    """Winsorization (quantile-based truncation) class.

    The 'fit' method calculates the lower and upper quantiles on the training data. The
    'transform' method applies the capping, replacing values below the lower bound with the bound
    itself, and values above the upper bound with the upper bound itself.

    Attributes
    ----------
    _lower_bounds : ndarray, shape (n_features,)
        The lower quantile values calculated for each column of X during `fit`.
    _upper_bounds : ndarray, shape (n_features,)
        The upper quantile values calculated for each column of X during `fit`.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        """Init method.

        Parameters
        ----------
        lower : float, default=0.01
            The lower quantile (percentile) to use for the truncation limit.
            E.g., 0.01 (1st percentile) truncates the lowest 1% of values.
        upper : float, default=0.99
            The upper quantile (percentile) to use for the truncation limit.
            E.g., 0.99 (99th percentile) truncates the highest 1% of values.
        """
        self.lower = lower
        self.upper = upper

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """Calculates the truncation limits (quantiles) per column (feature) in X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The training dataset for which the quantiles will be calculated.
        y : ndarray, default=None
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        self : object
            Returns the instance of the transformer.
        """
        X = np.asarray(X)
        self._lower_bounds = np.quantile(a=X, q=self.lower, axis=0)
        self._upper_bounds = np.quantile(a=X, q=self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applies Winsorization (truncation) to X using the limits calculated in `fit`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The dataset to be transformed (can be training or test data).

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_features)
            The transformed array with truncated outliers.
        """
        X = np.asarray(X).copy()
        X_transformed = np.clip(a=X, a_min=self._lower_bounds, a_max=self._upper_bounds)
        return X_transformed
