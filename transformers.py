"""
Module housing various transformers to be used in
Sklearn pipelines.
"""
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Select rows to be transformed.

    Follows sklearn design patterns containing fit() and
    transform() methods.
    """
    def __init__(self, attributes_names: list):
        """Initialize the selector with a list of attribut names."""
        self.attributes_names = attributes_names

    def fit(self, X: np.ndarray, y: np.ndarray=None):
        """Fit the data."""
        return self

    def transform(self, X) -> np.ndarray:
        """Return the selected columns as numpy arrays
        for use in sklearn."""
        return X[self.attributes_names].values


class ReturnDataFrame(BaseEstimator, TransformerMixin):
    """
    For additional exploration after going through a pipeline,
    transforms the array back into a pandas dataframe.
    """
    def __init__(self, attributes_names: list):
        """Initialize the selector with a list of attribut names."""
        self.attributes_names = attributes_names

    def fit(self, X: np.ndarray, y: np.ndarray=None):
        """Fit the data."""
        return self

    def transform(self, X: np.ndarray) -> pd.core.frame.DataFrame:
        """Return the numpy arrays
        for use in sklearn."""
        return pd.DataFrame(X, columns=self.attributes_names)
