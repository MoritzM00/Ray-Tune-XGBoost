"""Preprocess step in the pipeline."""

from pathlib import Path

from src.utils import get_processed_data_path, load_config, load_data


def preprocess(config):
    """Preprocess the data."""
    wines = load_data(raw=True, config=config)

    target = "quality"
    wines[target] -= 3

    Path(config["processed"]).mkdir(parents=True, exist_ok=True)
    wines.to_csv(get_processed_data_path(config), index=False)


if __name__ == "__main__":
    config = load_config("prepare")
    preprocess(config)
