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
    
    def __init__(self, dataset: str, root: str = './datasets/', encode: bool = False):
        """
        Parameters
        ----------
        dataset: str
            Name of the dataset that will be loaded and processed.
        root: str
            Root directory of the datasets.
        encode: bool
            Converts categorical variables into numerical values with One Hot Encoding. 
        """
        
        self.dataset = dataset
        self.root = root
        self.encode = encode
        logging.basicConfig(encoding='utf-8', level=logging.INFO)
        logging.info(f"Dataset: {self.dataset}")
        
    def load(self):
        """
        Load dataset according to dataset given as a parameter.
        """
        try:
            path = {
                'breast_cancer': self.root + 'breast_cancer/wdbc.csv',
                'dermatology': self.root + 'dermatology/dermatology.csv',
                'divorce': self.root + 'divorce/divorce.csv',
                'qsar_toxicity': self.root + 'qsar_toxicity/qsar_oral_toxicity.csv'
            }[self.dataset]
            
        except:
            raise AssertionError(f"The {self.dataset} dataset is not available.")
            
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
            
    def split(self, test_size: float = 0.20, seed: int = 123456):
        """
        Split dataset into train and test sets.
        
        Parameters
        ----------
        test_size: float, default 0.20
            Proportion of the dataset to include in the test set. It should be between 0 and 1.
            It can be integer too, but it refers to the number of observations in the test set, in
            this case.
        seed: int, default 123456
            Controls the shuffling applied to the data before applying the split.
        """
        subsets = train_test_split(self.X.to_numpy(),
                                   self.y.to_numpy(),
                                   test_size=test_size,
                                   random_state=seed)
        self.X_train, self.X_test, self.y_train, self.y_test = subsets
