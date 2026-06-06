import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def test_imports():
    """Test that all key packages import correctly."""
    import xgboost
    import shap
    import imblearn
    assert True


def test_dataframe_operations():
    """Test basic dataframe operations used in the project."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'class': [0, 1, 0]})
    assert df.shape == (3, 3)
    assert df['class'].sum() == 1


def test_logistic_regression_trains():
    """Test that logistic regression trains without error."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 4