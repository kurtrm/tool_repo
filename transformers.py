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


"""
This class below may become a common pattern.
Given a column with more than 3 or 4 categories, it may make sense
to aggregate categories into one.

In this case, there was 'Medium', 'Large', and 'Large / Medium' cateogries.
Looking at their means, mins, and maxs, they did not seem statistically
different, and the 'Large / Medium' category significantly outnumbered
the other two combined.
"""


class CategoricalAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregate categories that aren't dissimilar,
    meant for a specific column.
    """

    def fit(self, X, y=None):
        """Fit the data."""
        return self

    def transform(self, X):
        """Return the selected columns as numpy arrays
        for use in sklearn."""
        X[(X == 'Large') | (X == 'Medium')] = 'Large / Medium'
        return X

    def _to_large_medium(self, value):
        """
        Method to apply, aggregates 'Medium' and 'Large'
        categorized data into the 'Large / Medium' category.
        """
        if value == 'Medium' or value == 'Large':
            return 'Large / Medium'
        else:
            return value


class EquipmentAge(BaseEstimator, TransformerMixin):
    """
    Aggregate categories that aren't dissimilar,
    meant for a specific column.
    """

    def fit(self, X, y=None):
        """Fit the data."""
        return self

    def transform(self, X):
        """Return the selected columns as numpy arrays
        for use in sklearn."""
        X[(X == 'Large') | (X == 'Medium')] = 'Large / Medium'
        return X

    def _to_large_medium(self, value):
        """
        Method to apply, aggregates 'Medium' and 'Large'
        categorized data into the 'Large / Medium' category.
        """
        if value == 'Medium' or value == 'Large':
            return 'Large / Medium'
        else:
            return value


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Custom imputer for replacing NaNs in categorical data.
    """
    def fit(self, X, y=None):
        """Fit the data."""
        return self

    def transform(self, X):
        """Return the selected columns as numpy arrays
        for use in sklearn."""
        series = pd.Series(X)
        filled = series.fillna('Unknown')
        return filled.values
