from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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