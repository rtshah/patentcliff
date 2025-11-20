"""Test model baselines."""
import pytest
import pandas as pd
import numpy as np
from src.model.train import baseline_class_mean, baseline_linear
from src.utils import load_config


def test_class_mean_baseline():
    """Test that class mean baseline runs without error."""
    # Create dummy data
    train = pd.DataFrame({
        "complexity_score": [1, 1, 2, 2],
        "price_drop_pct": [50, 60, 70, 80],
    })
    val = pd.DataFrame({
        "complexity_score": [1, 2],
        "price_drop_pct": [55, 75],
    })
    test = pd.DataFrame({
        "complexity_score": [1, 2],
        "price_drop_pct": [55, 75],
    })
    
    result = baseline_class_mean(train, val, test)
    assert "metrics" in result
    assert "test_rmse" in result["metrics"]


def test_linear_baseline():
    """Test that linear baseline runs without error."""
    train = pd.DataFrame({
        "entrants_by_6m": [3, 5, 7, 9],
        "complexity_score": [1, 1, 2, 2],
        "price_drop_pct": [50, 60, 70, 80],
    })
    val = pd.DataFrame({
        "entrants_by_6m": [4, 6],
        "complexity_score": [1, 2],
        "price_drop_pct": [55, 75],
    })
    test = pd.DataFrame({
        "entrants_by_6m": [4, 6],
        "complexity_score": [1, 2],
        "price_drop_pct": [55, 75],
    })
    
    result = baseline_linear(train, val, test)
    assert "metrics" in result
    assert "test_rmse" in result["metrics"]

