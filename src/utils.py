"""Utility functions for saving artifacts and data preprocessing."""
import joblib
import os
from pathlib import Path
from typing import Union

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer


def save_file(
    data: Union[BaseEstimator, ColumnTransformer], file_name: str
) -> None:
    """Save the given data to a file using joblib.dumb.

    Args:
        data (Union[BaseEstimator, ColumnTransformer]): The data to be saved.
            It should be an instance of either BaseEstimator or
            ColumnTransformer from the scikit-learn library.
        file_name (str): The name of the file to be saved.
    """

    file_path = Path(__file__).parent / ".." / f"models/{file_name}"

    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file:
        joblib.dump(data, file)


def load_file(file_name: str) -> Union[BaseEstimator, ColumnTransformer]:
    """Load data from a file using joblib.load.

    Args:
        file_name (str): The name of the file to be loaded.

    Returns:
        Union[BaseEstimator, ColumnTransformer]: The loaded data. It is
        returned as an instance of either BaseEstimator or ColumnTransformer
        from the scikit-learn library.
    """

    file_path = Path(__file__).parent / ".." / f"models/{file_name}"

    with open(file_path, "rb") as file:
        return joblib.load(file)


class ColumnDropperTransformer:
    """Custom Transformer class that can be added to scikit-learn Pipeline
    This transformer drops specified columns from the input DataFrame.
    """

    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self


class ColumnNormalizedTransformer:
    """Custom Transformer class that normalizes selected features over other
    selected features.
    """

    def transform(self, X, y=None):
        X["population_per_household"] = X["population"] / X["households"]
        X["rooms_per_household"] = X["total_rooms"] / X["households"]
        X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"]
        return X

    def fit(self, X, y=None):
        return self
