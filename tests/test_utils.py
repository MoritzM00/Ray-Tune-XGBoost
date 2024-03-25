"""Tests for the preprocess module."""

from pathlib import Path

import pytest

from src.utils import get_processed_data_path, get_raw_data_path


@pytest.fixture(scope="session")
def config():
    """Return the simulated configuration dict for prepare stage."""
    return {
        "raw": "data/raw",
        "processed": "data/processed",
        "dataset_name": "winequality.csv",
    }


def test_raw_path(config):
    """Test that the raw data path is correct."""
    assert get_raw_data_path(config) == Path("data/raw/winequality.csv")


def test_processed_path(config):
    """Test that the processed data path is correct."""
    assert get_processed_data_path(config) == Path("data/processed/winequality.csv")
