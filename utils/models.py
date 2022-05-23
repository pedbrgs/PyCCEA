import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

class Model():
    
    """
    Loads a machine learning model, adjusts its hyperparameters and trains it.
    """
    
    def __init__(self, model_name):
        
        """
        Parameters
        ----------
        model_name: str
            Name of the machine learning model that will be fitted.
        """
        
        self.model_name = model_name
    
    def load(self, data, seed = 123456, kfolds = 5):
        
        """
        Loads machine learning model and adjusts its hyperparameters
        according to model_name given as a parameter.
        
        Parameters
        ----------
        seed: int, default 123456
            Controls the shuffling applied for subsampling the data.
        kfolds: int, default 5
            Number of folds in the k-fold cross validation.

        Attributes
        ----------
        model: estimator object
            Model that was chosen by the Halving Grid Search, i.e.,
            estimator which gave the best result on the left out data.
        """
        
        if self.model_name == 'svm':
            # Initializing Support Vector Classification
            model = SVC()
            # Grid of possible hyperparameter values
            self.grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                         'degree': np.arange(1, 6),
                         'gamma': ['scale', 'auto'],
                         'class_weight': [None, 'balanced']}
            # Tuning hyperparameters with Halving Grid Search with
            # cross validation
            search = HalvingGridSearchCV(model,
                                         self.grid,
                                         cv = kfolds,
                                         random_state = seed)
            print("Tuning hyperparameters ...")
            search.fit(data.X_train, data.y_train)
            # Best results before tuning
            self.model = search.best_estimator_
            self.best_hyperparams = search.best_params_
            print(f"Best hyperparameters: {self.best_hyperparams}")
            
        elif self.model_name == 'random_forest':
            pass