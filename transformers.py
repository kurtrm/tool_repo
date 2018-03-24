"""
Module housing various transformers to be used in
Sklearn pipelines.
"""
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Select rows to be transformed.

    Follows sklearn design patterns containing fit() and
    transform() methods.
    """
    def __init__(self, attributes_names: list) -> np.ndarray:
        """Initialize the selector with a list of attribut names."""
        self.attributes_names = attributes_names

    def fit(self, X, y=None):
        """Fit the data."""
        return self

    def transform(self, X):
        """Return the selected columns as numpy arrays
        for use in sklearn."""
        return X[self.attributes_names].values
