"""Test utils."""
from pathlib import Path

import pandas as pd


def get_raw_data_path(config: dict) -> str:
    """Get the path to the raw data.

    Parameters
    ----------
    config : Dict
        Configuration dictionary.

    Returns
    -------
    raw_data_path : pathlib.Path
        Path to the raw data.
    """
    return Path(config["raw"], config["dataset_name"])


def get_processed_data_path(config: dict):
    """Get the path to the processed data.

    Parameters
    ----------
    config : Dict
        Configuration dictionary.

    Returns
    -------
    processed_data_path : pathlib.Path
        Path to the processed data.
    """
    return Path(config["processed"], config["dataset_name"])


def load_data(raw: bool, config: dict, **kwargs):
    """Load the wine quality dataset.

    Parameters
    ----------
    raw : bool
        Whether to load the raw or processed data.
    config: dict
        Configuration dictionary.
    **kwargs
        Additional Keyword arguments to pass to `pd.read_csv`.

    Returns
    -------
    data : pd.DataFrame
        Dataframe containing the data.
    """
    if raw:
        path = get_raw_data_path(config)
    else:
        path = get_processed_data_path(config)
    return pd.read_csv(path, **kwargs)
