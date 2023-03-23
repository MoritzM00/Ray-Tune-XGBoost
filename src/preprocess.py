"""Preprocess step in the pipeline."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_wine_dataset(path: str):
    """Load the wine quality dataset.

    Parameters
    ----------
    path : str
        Path to the data.

    Returns
    -------
    data : pd.DataFrame
        Dataframe containing the data.
    """
    return pd.read_csv(path)


def save_data(X_train, X_test, y_train, y_test):
    """Save the data to disk.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : np.ndarray
        Test features.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    """
    # TODO: use params.yaml to specify the paths
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy", y_test)


def preprocess():
    """Preprocess the data."""
    wines = load_wine_dataset("data/raw/winequality-red.csv")

    target = "quality"
    X = wines.drop(columns=target)
    y = wines[target]
    y -= 3

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    # apply standard scaling
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    save_data(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    preprocess()
