import logging
import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score


class Model():
    """
    Loads a machine learning model, adjusts its hyperparameters and gets the best model.

    Attributes
    ----------
    estimator: estimator object
        Model that was chosen by the Halving Grid Search, i.e., estimator which gave the best
        result on the left out data.
    best_hyperparams: dict
        Best hyperparameters used to fit the machine learning model.
    """
    
    def __init__(self, model_name: str):
        """
        Parameters
        ----------
        model_name: str
            Name of the machine learning model that will be fitted.
        """

        self.model_name = model_name
        logging.basicConfig(encoding='utf-8', level=logging.INFO)
        logging.info(f"Model: {self.model_name}")
        warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

    def load(self, data, seed: int = 123456, kfolds: int = 5, optimize: bool = False):
        """
        Loads machine learning model and adjusts its hyperparameters according to model_name given
        as a parameter.
        
        Parameters
        ----------
        data: Dataset object
            Object containing the dataset and its post-processing subsets.
        seed: int, default 123456
            Controls the shuffling applied for subsampling the data.
        kfolds: int, default 5
            Number of folds in the k-fold cross validation.
        optimize: bool, default False
            If True, optimize the hyperparameters of the model and return the best model.
        """

        if self.model_name == 'svm':
            # Initializing Support Vector Classification
            model = SVC()
            # Grid of possible hyperparameter values
            self.grid = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': np.arange(1, 6),
                'gamma': ['scale', 'auto'],
                'class_weight': [None, 'balanced']
            }
            
        elif self.model_name == 'random_forest':
            # Initializing Random Forest Classifier
            model = RandomForestClassifier()
            # Grid of possible hyperparameter values
            self.grid = {
                'n_estimators': np.arange(1, 250, 50),
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 0.1, 0.2, 0.3, 0.4],
                'min_samples_leaf': [1, 0.1, 0.2, 0.3, 0.4, 0.5],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced_subsample', None],
                'ccp_alpha': np.arange(0, 0.6, 0.1),
                'max_samples': np.arange(0.1, 1.0, 0.1)
            }

        if optimize:
            # Scoring metric used according to the classification task
            scoring = 'f1_macro' if data.y_train.nunique() > 2 else 'f1'
            # Tuning hyperparameters with Halving Grid Search with cross validation
            search = HalvingGridSearchCV(model,
                                         self.grid,
                                         cv=kfolds,
                                         random_state=seed,
                                         scoring=scoring,
                                         refit=True)
            logging.info("Tuning hyperparameters ...")
            search.fit(data.X_train, data.y_train)
            # Best results before tuning
            self.estimator = search.best_estimator_
            self.best_hyperparams = search.best_params_
            logging.info(f"Best hyperparameters: {self.best_hyperparams}")

        else:
            logging.info("Fitting model ...")
            model.fit(data.X_train, data.y_train)
            self.estimator = model
            logging.info(f"Default hyperparameters: {self.estimator.get_params()}")
