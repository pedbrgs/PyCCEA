import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


class DataLoader():
    """
    Load dataset and preprocess it to train machine learning algorithms.

    Attributes
    ----------
    data: pd.DataFrame
        Raw dataset.
    X: pd.DataFrame
        Raw input data.
    y: pd.Series
        Raw output data.
    X_train: np.array
        Train input data.
    X_test: np.array
        Test input data.
    y_train: np.array
        Train output data.
    y_test: np.array
        Test output data.
    """

    datasets = {
        'breast_cancer': './datasets/breast_cancer/wdbc.csv',
        'dermatology': './datasets/dermatology/dermatology.csv',
        'divorce': './datasets/divorce/divorce.csv',
        'qsar_toxicity': './datasets/qsar_toxicity/qsar_oral_toxicity.csv'
        }

    def __init__(self, dataset: str, encode: bool = False):
        """
        Parameters
        ----------
        dataset: str
            Name of the dataset that will be loaded and processed.
        encode: bool
            Converts categorical variables into numerical values with One Hot Encoding. 
        """
        
        self.dataset = dataset
        self.encode = encode
        # Initialize logger with info level
        logging.basicConfig(encoding='utf-8', level=logging.INFO)
        
    def load(self):
        """
        Load dataset according to dataset given as a parameter.
        """
        try:
            path = DataLoader.datasets[self.dataset]
        except:
            # Check if the chosen dataset is available
            raise AssertionError(
                f"The '{self.dataset}' dataset is not available. "
                f"The available datasets are {', '.join(DataLoader.datasets.keys())}."
            )
        # Load dataset
        logging.info(f"Dataset: {self.dataset}")
        self.data = pd.read_csv(path)
        
    def preprocess(self):
        """
        Preprocess dataset according to dataset given as a parameter, i.e., splits the data into
        input X and output Y, handles categorical variables and normalizes all variables.
        """
        # Setting a default representation for NaN values 
        self.data.replace(to_replace = '?', value=np.nan, inplace=True)
        # Removing rows with at least one NaN value
        self.data.dropna(inplace=True)
        
        # Binary classification problem (C = 2)
        if self.dataset == 'breast_cancer':
            self.X = self.data.iloc[:,2:].copy()
            self.y = self.data.iloc[:,1].copy()
            self.y.loc[self.y == 'M'] = 1
            self.y.loc[self.y == 'B'] = 0
        
        # Multiclass classification problem (C = 6)
        elif self.dataset == 'dermatology':
            self.X = self.data.iloc[:,:-1].copy()
            self.y = self.data.iloc[:,-1].copy()
            if self.encode:
                self.y = self.y.values.reshape((-1,1))
                self.y = OneHotEncoder().fit_transform(self.y).toarray()

        # Binary classification problem (C = 2)
        elif self.dataset == 'divorce':
            self.X = self.data.iloc[:,:-1].copy()
            self.y = self.data.iloc[:,-1].copy()
            
        # Binary classification problem (C = 2)
        elif self.dataset == 'qsar_toxicity':
            self.X = self.data.iloc[:,:-1].copy()
            self.y = self.data.iloc[:,-1].copy()
            self.y.loc[self.y == 'positive'] = 1
            self.y.loc[self.y == 'negative'] = 0
        
        # Labels as integer values
        self.y = self.y.astype('int')
            
    def split(self, val_size: float = 0.10, test_size: float = 0.10, seed: int = 123456):
        """
        Split dataset into training, validation and test sets.
        
        Parameters
        ----------
        val_size: float, default 0.10
            Proportion of the dataset to include in the validation set. It should be between 0 and
            1. It can be an integer too, but it refers to the number of observations in the
            validation set, in this case.
        test_size: float, default 0.10
            Proportion of the dataset to include in the test set. It should be between 0 and 1.
            It can be an integer too, but it refers to the number of observations in the test set,
            in this case.
        seed: int, default 123456
            Controls the shuffling applied to the data before applying the split.
        """
        # Split data into training and test sets
        subsets = train_test_split(self.X.to_numpy(),
                                   self.y.to_numpy(),
                                   test_size=test_size,
                                   random_state=seed)
        self.X_train, self.X_test, self.y_train, self.y_test = subsets
        # Split training set into training and validation sets
        subsets = train_test_split(self.X_train,
                                   self.y_train,
                                   test_size=val_size/(1-test_size),
                                   random_state=seed)
        self.X_train, self.X_val, self.y_train, self.y_val = subsets
