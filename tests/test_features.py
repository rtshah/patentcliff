"""Test feature engineering."""
import pytest
from src.features.complexity import complexity_score
from src.utils import load_config


def test_complexity_score():
    """Test complexity score computation."""
    config = load_config()
    
    # Oral tablet should be low complexity
    score = complexity_score("ORAL", "TABLET", config)
    assert score == 1
    
    # Injectable should be high complexity
    score = complexity_score("INTRAVENOUS", "INJECTION", config)
    assert score == 5

