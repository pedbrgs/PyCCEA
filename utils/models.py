import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

class Model():
    
    """
    Loads a machine learning model, adjusts its hyperparameters and gets     the best model.
    """
    
    def __init__(self, model_name):
        
        """
        Parameters
        ----------
        model_name: str
            Name of the machine learning model that will be fitted.
        """
        
        self.model_name = model_name

    def load(self, data, seed = 123456, kfolds = 5, n_iter = 80):
        
        """
        Loads machine learning model and adjusts its hyperparameters
        according to model_name given as a parameter.
        
        Parameters
        ----------
        data: Dataset object
            Object containing the dataset and its post-processing subsets.
        seed: int, default 123456
            Controls the shuffling applied for subsampling the data.
        kfolds: int, default 5
            Number of folds in the k-fold cross validation.
        n_iter: int, default 80
            Number of parameter settings that are sampled.

        Attributes
        ----------
        estimator: estimator object
            Model that was chosen by the Random Search, i.e., estimator
            which gave the best result on the left out data.
        best_hyperparams: dict
            Best hyperparameters used to fit the machine learning model.
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
        
        # Tuning hyperparameters with Random Search with
        # cross validation
        search = RandomizedSearchCV(model,
                                    self.grid,
                                    cv = kfolds,
                                    n_iter = n_iter,
                                    random_state = seed)
        print("Tuning hyperparameters ...")
        search.fit(data.X_train, data.y_train)
        # Best results before tuning
        self.estimator = search.best_estimator_
        self.best_hyperparams = search.best_params_
        print(f"Best hyperparameters: {self.best_hyperparams}")