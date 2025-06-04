import logging
import warnings
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet


class RegressionModel():
    """Load a regression model, adjust its hyperparameters and get the best model.

    Attributes
    ----------
    estimator : sklearn model object
        Trained model. In case optimize is True, it is the model that was chosen by the Randomized
        Search, i.e., estimator which gave the best result on the validation data.
    hyperparams : dict
        Hyperparameters of the model. In case optimize is True, it is the best hyperparameters
        used to fit the machine learning model.
    """

    models = {
        "linear": (LinearRegression, {}),
        "ridge": (Ridge, {}),
        "lasso": (Lasso, {}),
        "elastic_net": (ElasticNet, {}),
        "random_forest": (RandomForestRegressor, {}),
        "support_vector_machine": (SVR, {}),
    }

    def __init__(self, model_type: str):
        """Initialize the regression model.

        Parameters
        ----------
        model_type : str
            Name of the machine learning model that will be fitted for a regression task.
        """
        # Check if the chosen regression model is available
        if model_type not in RegressionModel.models.keys():
            raise AssertionError(
                f"Model type '{model_type}' was not found. "
                f"The available models are {', '.join(RegressionModel.models.keys())}."
            )
        # Initialize regression model
        self.model_type = model_type
        model_class, params = RegressionModel.models[self.model_type]
        self.estimator = model_class(**params)
        # Initialize logger with info level
        logging.basicConfig(encoding="utf-8", level=logging.INFO)
        # Suppress warnings
        warnings.filterwarnings(action="ignore", category=FutureWarning)


    def _model_selection(self, 
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         n_iter: int = 100,
                         seed: int = None,
                         kfolds: int = 5):
        """Optimize the hyperparameters of the model and return the best model found.

        Parameters
        ----------
        X_train: np.ndarray
            Train input data.
        y_train: np.ndarray
            Train output data.
        n_iter: int, default 100
            Number of hyperparameter settings that are sampled. It trades off runtime and quality
            of the solution.
        seed: int, default None
            Controls the shuffling applied for subsampling the data.
        kfolds: int, default 5
            Number of folds in the k-fold cross validation.

        Returns
        -------
        estimator: sklearn model object
            Trained model. It is the model that was chosen by the Randomized Search, i.e., estimator
            which gave the best result on the validation data.
        hyperparams: dict
            Hyperparameters of the model. It is the best hyperparameters used to fit the machine
            learning model.
        """
        # Grid of possible hyperparameter values
        if self.model_type == "ridge":
            self.grid = {
                "alpha": np.logspace(-6, 6, num=13)
            }
        elif self.model_type == "lasso":
            self.grid = {
                "alpha": np.logspace(-6, 6, num=13)
            }
        elif self.model_type == "random_forest":
            self.grid = {
                "n_estimators": np.arange(10, 201, 10),
                "max_features": ["sqrt", "log2", None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False]
            }
        elif self.model_type == "support_vector_machine":
            self.grid = {
                "C": np.logspace(-6, 6, num=13),
                "epsilon": [0.01, 0.1, 0.5, 1.0],
                "kernel": ["linear", "poly", "rbf", "sigmoid"]
            }
        elif self.model_type == "elastic_net":
            self.grid = {
                "alpha": np.logspace(-6, 6, num=13),
                "l1_ratio": np.linspace(0, 1, 11)
            }
        elif self.model_type == "linear":
            self.grid = {
                "fit_intercept": [True, False]
            }

        # Scoring metric used for regression tasks
        scoring = "neg_mean_squared_error"

        # Tuning hyperparameters with Randomized Search with Cross Validation
        search = RandomizedSearchCV(estimator=self.estimator,
                                    param_distributions=self.grid,
                                    n_iter=n_iter,
                                    cv=kfolds,
                                    random_state=seed,
                                    scoring=scoring,
                                    refit=True)
        logging.info("Tuning hyperparameters ...")
        with warnings.catch_warnings():
            # NOTE: Suppress DeprecationWarning from scikit-learn Ridge regression
            # when calling scipy.linalg.solve() with deprecated 'sym_pos' argument.
            warnings.filterwarnings("ignore", message="The 'sym_pos' keyword is deprecated")
            search.fit(X_train, y_train)
        # Get best model and its respective hyperparameters
        estimator = search.best_estimator_
        hyperparams = search.best_params_

        return estimator, hyperparams 

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              seed: int = None,
              kfolds: int = 10,
              n_iter: int = 100,
              optimize: bool = False,
              verbose: bool = False):
        """Build and train a regression model.

        Parameters
        ----------
        X_train: np.ndarray
            Train input data.
        y_train: np.ndarray
            Train output data.   
        seed: int, default None
            Controls the shuffling applied for subsampling the data.
        kfolds: int, default 10
            Number of folds in the k-fold cross validation.
        n_iter: int, default 100
            Number of hyperparameter settings that are sampled. It trades off runtime and quality
            of the solution.
        optimize: bool, default False
            If True, optimize the hyperparameters of the model and return the best model.
        verbose: bool, default False
            If True, show logs.
        """
        logging.getLogger().disabled = False if verbose else True

        if optimize:
            # Best results before tuning
            self.estimator, self.hyperparams = self._model_selection(X_train=X_train,
                                                                     y_train=y_train,
                                                                     seed=seed,
                                                                     kfolds=kfolds,
                                                                     n_iter=n_iter)
        else:
            logging.info("Fitting model ...")
            self.estimator.fit(X_train, y_train)
            self.hyperparams = self.estimator.get_params()
        logging.info(f"Hyperparameters: {self.hyperparams}")


class ClassificationModel():
    """Load a classification model, adjust its hyperparameters and get the best model.

    Attributes
    ----------
    estimator : sklearn model object
        Trained model. In case optimize is True, it is the model that was chosen by the Randomized
        Search, i.e., estimator which gave the best result on the validation data.
    hyperparams : dict
        Hyperparameters of the model. In case optimize is True, it is the best hyperparameters
        used to fit the machine learning model.
    """

    models = {
        "complement_naive_bayes": (ComplementNB, {}),
        "gaussian_naive_bayes": (GaussianNB, {}),
        "k_nearest_neighbors": (KNeighborsClassifier, {"n_neighbors": 1}),
        "logistic_regression": (LogisticRegression, {}),
        "multinomial_naive_bayes": (MultinomialNB, {}),
        "random_forest": (RandomForestClassifier, {"n_jobs": -1}),
        "support_vector_machine": (SVC, {}),
    }

    def __init__(self, model_type: str):
        """Initialize the classification model.

        Parameters
        ----------
        model_type : str
            Name of the machine learning model that will be fitted for a classification task.
        """
        # Check if the chosen classification model is available
        if not model_type in ClassificationModel.models.keys():
            raise AssertionError(
                f"Model type '{model_type}' was not found. "
                f"The available models are {', '.join(ClassificationModel.models.keys())}."
            )
        # Initialize classification model
        self.model_type = model_type
        model_class, params = ClassificationModel.models[self.model_type]
        self.estimator = model_class(**params)
        # Initialize logger with info level
        logging.basicConfig(encoding="utf-8", level=logging.INFO)
        # Supress divide-by-zero warnings
        warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
        # Supress future warnings
        warnings.filterwarnings(action="ignore", category=FutureWarning)

    def _model_selection(self, 
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         n_iter: int = 100,
                         seed: int = None,
                         kfolds: int = 5):
        """Optimize the hyperparameters of the model and return the best model found.

        Parameters
        ----------
        X_train: np.ndarray
            Train input data.
        y_train: np.ndarray
            Train output data.
        n_iter: int, default 100
            Number of hyperparameter settings that are sampled. It trades off runtime and quality
            of the solution.
        seed: int, default None
            Controls the shuffling applied for subsampling the data.
        kfolds: int, default 5
            Number of folds in the k-fold cross validation.

        Returns
        -------
        estimator: sklearn model object
            Trained model. It is the model that was chosen by the Randomized Search, i.e., estimator
            which gave the best result on the validation data.
        hyperparams: dict
            Hyperparameters of the model. It is the best hyperparameters used to fit the machine
            learning model.
        """
        # Grid of possible hyperparameter values
        if self.model_type == "support_vector_machine":
            self.grid = {
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "degree": np.arange(1, 3),
                "gamma": ["scale", "auto"],
                "class_weight": [None, "balanced"]
            }
        elif self.model_type == "random_forest":
            self.grid = {
                "n_estimators": np.arange(1, 250, 50),
                "criterion": ["gini", "entropy"],
                "min_samples_split": [2, 0.1, 0.2, 0.3, 0.4],
                "min_samples_leaf": [1, 0.1, 0.2, 0.3, 0.4, 0.5],
                "max_features": ["sqrt", "log2", None],
                "class_weight": ["balanced_subsample", None],
                "ccp_alpha": np.arange(0, 0.6, 0.1),
                "max_samples": np.arange(0.1, 1.0, 0.1)
            }
        elif self.model_type in ["complement_naive_bayes", "multinomial_naive_bayes"]:
            self.grid = {
                "alpha": [0.01, 0.1, 0.5, 1.0, 10.0],
                "fit_prior": [True, False]
            }
        elif self.model_type in ["gaussian_naive_bayes"]:
            self.grid = {
                "var_smoothing": np.logspace(0,-9, num=100)
            }
        elif self.model_type == "k_nearest_neighbors":
            self.grid = {
                "n_neighbors": np.arange(1, 31, 1)
            }

        # Scoring metric used according to the classification task
        scoring = "f1_macro" if np.unique(y_train).shape[0] > 2 else "f1"

        # Tuning hyperparameters with Randomized Search with Cross Validation
        search = RandomizedSearchCV(estimator=self.estimator,
                                    param_distributions=self.grid,
                                    n_iter=n_iter,
                                    cv=kfolds,
                                    random_state=seed,
                                    scoring=scoring,
                                    refit=True)
        logging.info("Tuning hyperparameters ...")
        search.fit(X_train, y_train)
        # Get best model and its respective hyperparameters
        estimator = search.best_estimator_
        hyperparams = search.best_params_

        return estimator, hyperparams 

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              seed: int = None,
              kfolds: int = 10,
              n_iter: int = 100,
              optimize: bool = False,
              verbose: bool = False):
        """Build and train a classification model.

        Parameters
        ----------
        X_train: np.ndarray
            Train input data.
        y_train: np.ndarray
            Train output data.   
        seed: int, default None
            Controls the shuffling applied for subsampling the data.
        kfolds: int, default 10
            Number of folds in the k-fold cross validation.
        n_iter: int, default 100
            Number of hyperparameter settings that are sampled. It trades off runtime and quality
            of the solution.
        optimize: bool, default False
            If True, optimize the hyperparameters of the model and return the best model.
        verbose: bool, default False
            If True, show logs.
        """
        logging.getLogger().disabled = False if verbose else True

        if optimize:
            # Best results before tuning
            self.estimator, self.hyperparams = self._model_selection(X_train=X_train,
                                                                     y_train=y_train,
                                                                     seed=seed,
                                                                     kfolds=kfolds,
                                                                     n_iter=n_iter)
        else:
            logging.info("Fitting model ...")
            self.estimator.fit(X_train, y_train)
            self.hyperparams = self.estimator.get_params()
        logging.info(f"Hyperparameters: {self.hyperparams}")
