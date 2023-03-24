"""Tests for the preprocess module."""
import os

import pandas as pd

from src.prepare import save_data


def test_save_data():
    """Test the save_data function."""
    dummy = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    save_data(dummy, path="./dummy.csv")
    assert os.path.exists("./dummy.csv")
    os.remove("./dummy.csv")
