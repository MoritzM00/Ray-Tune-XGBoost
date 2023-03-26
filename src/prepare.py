"""Preprocess step in the pipeline."""
from pathlib import Path

import yaml

from src.utils import get_processed_data_path, load_data


def preprocess(config):
    """Preprocess the data."""
    wines = load_data(raw=True, config=config)

    target = "quality"
    wines[target] -= 3

    Path(config["processed"]).mkdir(parents=True, exist_ok=True)
    wines.to_csv(get_processed_data_path(config), index=False)


if __name__ == "__main__":
    config = yaml.safe_load(open("params.yaml"))["prepare"]
    preprocess(config)
