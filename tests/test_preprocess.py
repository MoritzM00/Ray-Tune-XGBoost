"""Tests for the preprocess module."""
import pandas as pd

from src.preprocess import load_wine_dataset


def test_load_wine_dataset():
    """Test the load_wine_dataset function."""
    wines = load_wine_dataset("data/raw/winequality-red.csv")
    assert isinstance(wines, pd.DataFrame)
