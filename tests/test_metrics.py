import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression, make_classification
from metrics import Metrics


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
    return X, y


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=3, n_classes=2, random_state=42)
    return X, y


@pytest.fixture
def multi_class_data():
    X, y = make_classification(n_samples=100, n_features=3, n_classes=3, random_state=42)
    return X, y


def test_regression_metrics(regression_data):
    X, y = regression_data
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    metrics = Metrics(y, y_pred)
    assert metrics.r2() == pytest.approx(model.score(X, y))
    assert metrics.mse() == pytest.approx(np.mean((y - y_pred) ** 2))
    assert metrics.mae() == pytest.approx(np.mean(np.abs(y - y_pred)))
    assert metrics.rmse() == pytest.approx(np.sqrt(np.mean((y - y_pred) ** 2)))


def test_classification_metrics(classification_data):
    X, y = classification_data
    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    metrics = Metrics(y, y_pred)
    assert metrics.accuracy() == pytest.approx(model.score(X, y))
    assert metrics.recall() == pytest.approx(1.0)  # As we're using a regression model, recall should be 1.0
    assert metrics.precision() == pytest.approx(1.0)  # As we're using a regression model, precision should be 1.0
    assert metrics.f1() == pytest.approx(1.0)  # As we're using a regression model, F1-score should be 1.0


def test_multi_class_classification_metrics(multi_class_data):
    X, y = multi_class_data
    model = RandomForestClassifier()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    metrics = Metrics(y, y_pred)
    assert metrics.accuracy() == pytest.approx(model.score(X, y))


def test_confusion_matrix_df(classification_data):
    X, y = classification_data
    model = LogisticRegression()  # Not a classifier, just for testing
    model.fit(X, y)
    y_pred = model.predict(X)
    
    metrics = Metrics(y, y_pred)
    cm_df = metrics.confusion_matrix_df()
    assert isinstance(cm_df, pd.DataFrame)
    assert cm_df.shape == (2, 2)  # Assuming binary classification


def test_classification_report_df(classification_data):
    X, y = classification_data
    model = LogisticRegression()  # Not a classifier, just for testing
    model.fit(X, y)
    y_pred = model.predict(X)
    
    metrics = Metrics(y, y_pred)
    cr_df = metrics.classification_report_df()
    assert isinstance(cr_df, pd.DataFrame)
    assert cr_df.shape[0] == 2  # Assuming binary classification


def test_invalid_data_type():
    with pytest.raises(TypeError):
        Metrics("y_true", "y_pred")  # Both y_true and y_pred should be numpy arrays or pandas Series/DataFrames


def test_missing_feature_matrix():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 1, 0])
    with pytest.raises(ValueError):
        Metrics(y_true, y_pred)  # Feature matrix X is required for some methods


def test_classification_metrics_performance(benchmark, classification_data):
    X, y = classification_data
    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    metrics = Metrics(y, y_pred)
    
    # Measure the performance of the accuracy() method
    benchmark(metrics.accuracy)
