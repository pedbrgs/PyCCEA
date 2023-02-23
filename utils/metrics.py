import logging
from utils.datasets import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


class Evaluator():
    """
    Evaluate machine learning model according to some classification metrics.

    Attributes
    ----------
    y_pred: np.array
        Output predicted by the machine learning model given as a parameter in the test set.
    conf_matrix: np.array
        Square matrix (CxC), where C is the number of classes, containing the number of true
        positives (tp), number of false positives (fp), number of true negatives (tn) and
        number of false negatives (fn).
    precision: float
        Ratio of the correctly identified positive cases to all the predicted positive cases,
        i.e., tp/(tp+fp). 
    recall: float
        Also known as sensitivity, is the ratio of the correctly identified positive cases to
        all the actual positive cases, i.e., tp/(tp+fn).
    f1_score: float
        Harmonic mean of precision and recall, i.e., 2.(precision.recall)/(precision+recall).
    accuracy: float
        Ratio of number of correct predictions to the total number of input samples, i.e.,
        (tp+tn)/(tp+fp+tn+fn).
    """
    
    def __init__(self):

        logging.basicConfig(encoding='utf-8', level=logging.INFO)

    def evaluate(self, model, data: DataLoader, verbose: bool=True):
        """
        Parameters
        ----------
        model: estimator object
            Model that was chosen by the Halving Grid Search, i.e., estimator which gave the best
            result on the left out data.
        data: Dataset object
            Object containing the dataset and its post-processing subsets.
        verbose: bool, default True
            If True, show evaluation metrics in the test set.
        """

        # Predict
        self.y_pred = model.estimator.predict(data.X_test)

        # Type of aggregation used in the evaluation metrics according to the classification task
        avg = 'macro' if data.y_train.nunique() > 2 else 'binary'
        
        # Confusion matrix
        self.conf_matrix = confusion_matrix(data.y_test, self.y_pred)
        # Precision
        self.precision = round(precision_score(data.y_test, self.y_pred, average=avg), 4)
        # Recall
        self.recall = round(recall_score(data.y_test, self.y_pred, average=avg), 4)
        # F1-score
        self.f1_score = round(f1_score(data.y_test, self.y_pred, average=avg), 4)
        # Accuracy
        self.accuracy = round(accuracy_score(data.y_test, self.y_pred), 4)
        # Specificity
        tn, fp, fn, tp = self.conf_matrix.ravel()
        self.specificity = round(tn / (tn + fp), 4)

        # Show evaluation metrics
        if verbose:
            logging.info(f"Precision: {self.precision}")
            logging.info(f"Recall/Sensitivity/TPR: {self.recall}")
            logging.info(f"F1-score: {self.f1_score}")
            logging.info(f"Accuracy: {self.accuracy}")
            logging.info(f"Specificity/TNR: {self.specificity}")
