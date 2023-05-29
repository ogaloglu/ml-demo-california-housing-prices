"""Utility functions for saving artifacts and data preprocessing."""
import joblib
import os
from itertools import product
from pathlib import Path
from typing import Union, Tuple

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


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


def prepare_data() -> None:
    """Prepare data by splitting data into partitions.
    """
    data_dir = Path(__file__).parent / ".." / "data/"
    df = pd.read_csv(os.path.join(data_dir, "housing.csv"))

    y = df.pop("median_house_value")
    X = df

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, shuffle=True
    )

    # save the data
    X_train.to_csv(os.path.join(data_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(data_dir, "y_train.csv"), index=False)

    X_val.to_csv(os.path.join(data_dir, "X_val.csv"), index=False)
    y_val.to_csv(os.path.join(data_dir, "y_val.csv"), index=False)

    X_test.to_csv(os.path.join(data_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(data_dir, "y_test.csv"), index=False)


def validate_data_existance() -> bool:
    """Validate that dataset is already split.

    Returns:
        bool: Either split data exists or not.
    """
    directory_path = Path(__file__).parent / ".." / "data/"
    return all(
        [
            os.path.isfile(
                os.path.join(directory_path, f"{data}_{partition}.csv")
            )
            for data, partition in product(
                ["X", "y"], ["train", "val", "test"]
            )
        ]
    )


def get_data() -> Tuple[pd.DataFrame]:
    """Get all partitions of the California Housing Prices dataset.

    Returns:
        Tuple[pd.DataFrame]: A tuple that includes all partitions of the
        dataset.
    """
    if not validate_data_existance():
        prepare_data()

    data_dir = Path(__file__).parent / ".." / "data/"

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))

    X_val = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(data_dir, "y_val.csv"))

    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))

    return X_train, y_train, X_val, y_val, X_test, y_test


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
