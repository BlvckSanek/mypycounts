import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, \
    accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.base import BaseEstimator
from typing import Union, List, Dict


class Metrics:
    """
    A class for calculating and analyzing metrics of machine learning models.

    Attributes:
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    X : np.ndarray, optional
        Feature matrix (default is None).

    Methods:
    -------
    r2()
        Calculate the R-squared (coefficient of determination) metric.
    mse()
        Calculate the Mean Squared Error metric.
    mae()
        Calculate the Mean Absolute Error metric.
    rmse()
        Calculate the Root Mean Squared Error metric.
    accuracy()
        Calculate the accuracy metric for classification tasks.
    recall()
        Calculate the recall (sensitivity) metric for classification tasks.
    precision()
        Calculate the precision metric for classification tasks.
    f1()
        Calculate the F1-score metric for classification tasks.
    confusion_matrix_df()
        Generate the confusion matrix as a pandas DataFrame.
    classification_report_df()
        Generate the classification report as a pandas DataFrame.
    prediction_contributors()
        Calculate the contributors to the model's predictions.
    top_feature_contributors()
        Identify the top feature contributors to the target variable prediction.
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray = None):
        """
        Initialize Metrics instance.

        Parameters:
        -----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.
        X : np.ndarray, optional
            Feature matrix (default is None).

        """

        self.y_true = y_true
        self.y_pred = y_pred
        self.X = X

    def _get_feature_names(self) -> List[str]:
        """
        Get feature names from the feature matrix.

        Returns:
        --------
        List[str]:
            List of feature names.

        """

        if isinstance(self.y_true, (pd.Series, pd.DataFrame)):
            feature_names = self.X.columns.tolist()
        else:
            feature_names = [f"Feature_{i}" for i in range(self.X.shape[1])]
        return feature_names

    def r2(self) -> float:
        """
        Calculate the R-squared (coefficient of determination) metric.

        Returns:
        --------
        float:
            R-squared value.

        Examples:
        ---------
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> metrics = Metrics(y_true, y_pred)
        >>> metrics.r2()
        0.9486081370449679
        """
        return r2_score(self.y_true, self.y_pred)

    def mse(self) -> float:
        """
        Calculate the Mean Squared Error metric.

        Returns:
        --------
        float:
            Mean Squared Error value.

        Examples:
        ---------
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> metrics = Metrics(y_true, y_pred)
        >>> metrics.mse()
        0.375
        """
        return mean_squared_error(self.y_true, self.y_pred)

    def mae(self) -> float:
        """
        Calculate the Mean Absolute Error metric.

        Returns:
        --------
        float:
            Mean Absolute Error value.

        Examples:
        ---------
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> metrics = Metrics(y_true, y_pred)
        >>> metrics.mae()
        0.5
        """
        return mean_absolute_error(self.y_true, self.y_pred)

    def rmse(self) -> float:
        """
        Calculate the Root Mean Squared Error metric.

        Returns:
        --------
        float:
            Root Mean Squared Error value.

        Examples:
        ---------
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> metrics = Metrics(y_true, y_pred)
        >>> metrics.rmse()
        0.6123724356957945
        """
        return np.sqrt(self.mse())

    def accuracy(self) -> float:
        """
        Calculate the accuracy metric for classification tasks.

        Returns:
        --------
        float:
            Accuracy value.

        Examples:
        ---------
        >>> y_true = np.array([0, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 1, 1])
        >>> metrics = Metrics(y_true, y_pred)
        >>> metrics.accuracy()
        0.75
        """
        return accuracy_score(self.y_true, self.y_pred)

    def recall(self) -> float:
        """
        Calculate the recall (sensitivity) metric for classification tasks.

        Returns:
        --------
        float:
            Recall value.

        Examples:
        ---------
        >>> y_true = np.array([0, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 1, 1])
        >>> metrics = Metrics(y_true, y_pred)
        >>> metrics.recall()
        0.5
        """
        return recall_score(self.y_true, self.y_pred)

    def precision(self) -> float:
        """
        Calculate the precision metric for classification tasks.

        Returns:
        --------
        float:
            Precision value.

        Examples:
        ---------
        >>> y_true = np.array([0, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 1, 1])
        >>> metrics = Metrics(y_true, y_pred)
        >>> metrics.precision()
        0.6666666666666666
        """
        return precision_score(self.y_true, self.y_pred)

    def f1(self) -> float:
        """
        Calculate the F1-score metric for classification tasks.

        Returns:
        --------
        float:
            F1-score value.

        Examples:
        ---------
        >>> y_true = np.array([0, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 1, 1])
        >>> metrics = Metrics(y_true, y_pred)
        >>> metrics.f1()
        0.5714285714285715
        """
        return f1_score(self.y_true, self.y_pred)

    def confusion_matrix_df(self) -> pd.DataFrame:
        """
        Generate the confusion matrix as a pandas DataFrame.

        Returns:
        --------
        pd.DataFrame:
            Confusion matrix as a pandas DataFrame.

        Examples:
        ---------
        >>> y_true = np.array([0, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 1, 1])
        >>> metrics = Metrics(y_true, y_pred)
        >>> metrics.confusion_matrix_df()
           0  1
        0  1  1
        1  0  2
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        return pd.DataFrame(cm, index=np.unique(self.y_true), columns=np.unique(self.y_true))

    def classification_report_df(self) -> pd.DataFrame:
        """
        Generate the classification report as a pandas DataFrame.

        Returns:
        --------
        pd.DataFrame:
            Classification report as a pandas DataFrame.

        Examples:
        ---------
        >>> y_true = np.array([0, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 1, 1])
        >>> metrics = Metrics(y_true, y_pred)
        >>> metrics.classification_report_df()
                   precision    recall  f1-score  support
        0              1.00      0.50      0.67      2.0
        1              0.67      1.00      0.80      2.0
        """
        report_dict = classification_report(self.y_true, self.y_pred, output_dict=True)
        return pd.DataFrame(report_dict).transpose()

    def prediction_contributors(self) -> Dict:
        """
        Calculate the contributors to the model's predictions.

        Returns:
        --------
        Dict:
            Dictionary containing contributors to the model's predictions.

        Examples:
        ---------
        >>> y_true = np.array([0, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 1, 1])
        >>> metrics = Metrics(y_true, y_pred)
        >>> metrics.prediction_contributors()
        {0: {0: 1}, 1: {1: 2}}
        """
        contributors = {}
        for true_label, pred_label in zip(self.y_true, self.y_pred):
            if true_label not in contributors:
                contributors[true_label] = {}
            if pred_label not in contributors[true_label]:
                contributors[true_label][pred_label] = 0
            contributors[true_label][pred_label] += 1
        return contributors

    def top_feature_contributors(self, model: BaseEstimator, n: int = 3) -> List[str]:
        """
        Identify the top feature contributors to the target variable prediction.

        Parameters:
        -----------
        model : BaseEstimator
            Machine learning model.
        n : int, optional
            Number of top contributors to return (default is 3).

        Returns:
        --------
        List[str]:
            List of top feature contributors.

        Raises:
        -------
        ValueError:
            If feature matrix 'X' is not provided or if the model does not have 'coef_' or 'feature_importances_' attribute.

        Examples:
        ---------
        >>> from sklearn.linear_model import LinearRegression
        >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> y = np.array([1, 2, 3])
        >>> model = LinearRegression()
        >>> metrics = Metrics(y, y, X)
        >>> metrics.top_feature_contributors(model)
        ['Feature_2', 'Feature_1', 'Feature_0']
        """
        if self.X is None:
            raise ValueError("Feature matrix 'X' is required to calculate feature contributions.")

        model.fit(self.X, self.y_true)
        
        if not hasattr(model, "coef_") and not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not have 'coef_' or 'feature_importances_' attribute.")

        if hasattr(model, "coef_"):
            coef_abs = np.abs(model.coef_)
        elif hasattr(model, "feature_importances_"):
            coef_abs = model.feature_importances_
        
        top_indices = np.argsort(coef_abs)[::-1][:n]
        feature_names = self._get_feature_names()
        top_features = [feature_names[i] for i in top_indices]
        return top_features