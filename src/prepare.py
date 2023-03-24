"""Preprocess step in the pipeline."""
import pandas as pd


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


def preprocess():
    """Preprocess the data."""
    wines = load_wine_dataset("data/raw/winequality-red.csv")

    target = "quality"  # params.yaml
    wines[target] -= 3

    # TODO: use params.yaml to specify the paths
    save_data(wines, "data/processed/wines.csv")


if __name__ == "__main__":
    preprocess()
