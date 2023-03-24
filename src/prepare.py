"""Preprocess step in the pipeline."""
import os
from pathlib import Path

import pandas as pd
import yaml


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


def save_data(data: pd.DataFrame, path: str) -> None:
    """Save the data to disk.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data.
    """
    data.to_csv(path, index=False)


def preprocess(params):
    """Preprocess the data."""
    print(params)

    raw_path = Path(params["raw"], params["dataset_name"])
    processed_path = Path(params["processed"], params["dataset_name"])
    wines = load_wine_dataset(raw_path)

    target = "quality"  # params.yaml
    wines[target] -= 3

    os.mkdir(params["processed"])
    save_data(wines, processed_path)


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    preprocess(params)
