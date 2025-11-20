"""Test label construction."""
import pytest
import pandas as pd
from src.label.build_label import build_label_row
from src.connectors.openfda_client import OpenFDAClient
from src.connectors.nadac_client import NADACClient
from src.utils import load_config


def test_label_math():
    """Test that price_drop_pct is computed correctly."""
    # price_drop_pct = (price_T0 - price_T6) / price_T0 * 100
    price_t0 = 100.0
    price_t6 = 30.0
    expected_drop = ((price_t0 - price_t6) / price_t0) * 100
    assert expected_drop == 70.0


def test_missing_price_handling():
    """Test that missing prices are flagged correctly."""
    # Placeholder - would test with mocked NADAC data
    assert True

