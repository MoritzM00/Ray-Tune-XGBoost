"""Utilities for the project."""
from pathlib import Path

import pandas as pd
import yaml


def load_config(stage: str) -> dict:
    """Load the configuration file.

    Parameters
    ----------
    stage : str
        Stage of the pipeline.

    Returns
    -------
    config : Dict
        Configuration dictionary.
    """
    return yaml.safe_load(open("params.yaml"))[stage]


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


def get_model_path(config: dict):
    """Get the path to the model.

    Parameters
    ----------
    config : Dict
        Configuration dictionary.

    Returns
    -------
    model_path : pathlib.Path
        Path to the model.
    """
    return Path(config["models"], config["model_name"])


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
