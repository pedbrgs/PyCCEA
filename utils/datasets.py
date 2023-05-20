import logging
import numpy as np
import pandas as pd
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
    X_train: np.ndarray
        Train input data.
    X_test: np.ndarray
        Test input data.
    y_train: np.ndarray
        Train output data.
    y_test: np.ndarray
        Test output data.
    """

    data_folder = './datasets/'

    datasets = {
        '11_tumor': f'{data_folder}11_tumor.csv',
        '9_tumor': f'{data_folder}9_tumor.csv',
        'brain_tumor_1': f'{data_folder}brain_tumor_1.csv',
        'brain_tumor_2': f'{data_folder}brain_tumor_2.csv',
        'breast_cancer': f'{data_folder}wdbc.csv',
        'dermatology': f'{data_folder}dermatology.csv',
        'divorce': f'{data_folder}divorce.csv',
        'dlbcl': f'{data_folder}dlbcl.csv',
        'leukemia_1': f'{data_folder}leukemia_1.csv',
        'leukemia_2': f'{data_folder}leukemia_2.csv',
        'leukemia_3': f'{data_folder}leukemia_3.csv',
        'lungc': f'{data_folder}lungc.csv',
        'prostate_tumor_1': f'{data_folder}prostate_tumor_1.csv',
        'qsar_toxicity': f'{data_folder}qsar_oral_toxicity.csv'
        }

    def __init__(self, dataset: str, encode: bool = False):
        """
        Parameters
        ----------
        dataset: str
            Name of the dataset that will be loaded and processed.
        """
        
        self.dataset = dataset
        # Initialize logger with info level
        logging.basicConfig(encoding='utf-8', level=logging.INFO)

    def _check_header(self, file: str):
        """
        Check if a CSV file has a header.

        Parameters
        ----------
        file: str
            Name of the CSV file.

        Returns
        -------
        has_header: bool
            True if file has a header.
        """
        data = pd.read_csv(file, header=None, nrows=1)
        has_header = data.dtypes.nunique() != 1
        return has_header

    def _get_input(self):
        """Get the input data X from the dataset when the output is the last column."""
        return self.data.iloc[:,:-1].copy()

    def _get_output(self):
        """Get the output data y from the dataset when the output is the last column."""
        return self.data.iloc[:,-1].copy()

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
        if self._check_header(path):
            self.data = pd.read_csv(path, header=None)
        else:
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

        # Standard preprocessing cases
        special_datasets = ['breast_cancer', 'qsar_toxicity']
        standard_datasets = list(set(DataLoader.datasets.keys()).difference(special_datasets))

        if self.dataset in standard_datasets:
            self.X = self._get_input()
            self.y = self._get_output()

        elif self.dataset == 'breast_cancer':
            self.X = self.data.iloc[:,2:].copy()
            self.y = self.data.iloc[:,1].copy()
            self.y.loc[self.y == 'M'] = 1
            self.y.loc[self.y == 'B'] = 0

        elif self.dataset == 'qsar_toxicity':
            self.X = self._get_input()
            self.y = self._get_output()
            self.y.loc[self.y == 'positive'] = 1
            self.y.loc[self.y == 'negative'] = 0
        
        # Labels as integer values
        self.y = self.y.astype(int)
            
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
        if test_size > 0:
            subsets = train_test_split(self.X.to_numpy(),
                                       self.y.to_numpy(),
                                       test_size=test_size,
                                       random_state=seed)
            self.X_train, self.X_test, self.y_train, self.y_test = subsets
            # Split training set into training and validation sets
            if val_size > 0:
                subsets = train_test_split(self.X_train,
                                           self.y_train,
                                           test_size=val_size/(1-test_size),
                                           random_state=seed)
                self.X_train, self.X_val, self.y_train, self.y_val = subsets
            # Use only training and test sets
            else:
                # There is no validation set
                self.X_val, self.y_val = None, None
        else:
            # There is no test set
            self.X_test, self.y_test = None, None
            # Split data into training and validation sets
            if val_size > 0:
                subsets = train_test_split(self.X.to_numpy(),
                                           self.y.to_numpy(),
                                           test_size=val_size,
                                           random_state=seed)
                self.X_train, self.X_val, self.y_train, self.y_val = subsets
            # Do not split the data. It can be a cross-validation with all data
            else:
                self.X_train, self.y_train = self.X.to_numpy().copy(), self.y.to_numpy().copy()
                # There is no validation set
                self.X_val, self.y_val = None, None
