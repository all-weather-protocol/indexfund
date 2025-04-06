"""
Unit tests for the weighting module.
"""

import io
import math

# Add the parent directory to the path to make the package importable
import os
import sys
from unittest.mock import patch

import pytest

from index500.weighting import (
    calculate_index_weights,
    calculate_weight_market_cap,
    calculate_weight_sqrt_market_cap,
    display_initial_weights,
    normalize_weights,
    print_portfolio_weights,
)

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture
def sample_market_caps():
    """Sample market cap data for testing."""
    return {
        "btc": 1000000000000,  # 1 trillion
        "eth": 200000000000,  # 200 billion
        "sol": 50000000000,  # 50 billion
        "link": 5000000000,  # 5 billion
    }


@pytest.fixture
def sample_historical_data():
    """Sample historical data for testing display_initial_weights."""
    timestamp1 = 1609459200000  # 2021-01-01
    timestamp2 = 1609545600000  # 2021-01-02

    return {
        "btc": [
            (timestamp1, 30000, 600000000000),  # ts, price, mcap
            (timestamp2, 31000, 620000000000),
        ],
        "eth": [
            (timestamp1, 800, 100000000000),
            (timestamp2, 850, 106000000000),
        ],
        "sol": [
            (timestamp1, 1.5, 500000000),
            (timestamp2, 1.8, 600000000),
        ],
    }


def test_calculate_weight_market_cap(sample_market_caps):
    """Test that market cap weights are calculated correctly."""
    weights = calculate_weight_market_cap(sample_market_caps)

    # Should return exactly the same values as input
    assert weights == sample_market_caps
    assert weights["btc"] == 1000000000000
    assert weights["eth"] == 200000000000
    assert weights["sol"] == 50000000000
    assert weights["link"] == 5000000000


def test_calculate_weight_sqrt_market_cap(sample_market_caps):
    """Test that square root market cap weights are calculated correctly."""
    weights = calculate_weight_sqrt_market_cap(sample_market_caps)

    # Check square root calculations
    assert weights["btc"] == math.sqrt(1000000000000)
    assert weights["eth"] == math.sqrt(200000000000)
    assert weights["sol"] == math.sqrt(50000000000)
    assert weights["link"] == math.sqrt(5000000000)


def test_normalize_weights(sample_market_caps):
    """Test that weights are normalized correctly to sum to 1.0."""
    # Test with market cap weighting
    weights = calculate_weight_market_cap(sample_market_caps)
    normalized = normalize_weights(weights)

    # Sum should be 1.0
    assert sum(normalized.values()) == pytest.approx(1.0)

    # Check individual normalized weights
    total = sum(sample_market_caps.values())
    assert normalized["btc"] == pytest.approx(sample_market_caps["btc"] / total)
    assert normalized["eth"] == pytest.approx(sample_market_caps["eth"] / total)
    assert normalized["sol"] == pytest.approx(sample_market_caps["sol"] / total)
    assert normalized["link"] == pytest.approx(sample_market_caps["link"] / total)

    # Test with a simpler example
    simple_weights = {"a": 1, "b": 1, "c": 2}
    norm = normalize_weights(simple_weights)
    assert norm["a"] == 0.25
    assert norm["b"] == 0.25
    assert norm["c"] == 0.5


def test_calculate_index_weights_market_cap(sample_market_caps):
    """Test index weight calculation using market cap method."""
    weights = calculate_index_weights(sample_market_caps, "market_cap")

    # Should be normalized weights based on market cap
    total = sum(sample_market_caps.values())
    assert weights["btc"] == pytest.approx(sample_market_caps["btc"] / total)
    assert weights["eth"] == pytest.approx(sample_market_caps["eth"] / total)
    assert weights["sol"] == pytest.approx(sample_market_caps["sol"] / total)
    assert weights["link"] == pytest.approx(sample_market_caps["link"] / total)

    # Sum should be 1.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_calculate_index_weights_sqrt_market_cap(sample_market_caps):
    """Test index weight calculation using square root market cap method."""
    weights = calculate_index_weights(sample_market_caps, "sqrt_market_cap")

    # Calculate expected normalized weights
    sqrt_values = {asset: math.sqrt(mc) for asset, mc in sample_market_caps.items()}
    total_sqrt = sum(sqrt_values.values())
    expected = {asset: sqrt_val / total_sqrt for asset, sqrt_val in sqrt_values.items()}

    # Check weights match expected values
    for asset in sample_market_caps:
        assert weights[asset] == pytest.approx(expected[asset])

    # Sum should be 1.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_calculate_index_weights_invalid_method(sample_market_caps):
    """Test that an invalid method raises a ValueError."""
    with pytest.raises(ValueError):
        calculate_index_weights(sample_market_caps, "invalid_method")


def test_print_portfolio_weights():
    """Test that print_portfolio_weights correctly formats weights."""
    weights = {"btc": 0.6, "eth": 0.3, "sol": 0.1}

    # Capture stdout to check the output
    with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
        print_portfolio_weights(weights)
        output = fake_stdout.getvalue()

    # Check that output contains expected content
    assert "=== Portfolio Weights ===" in output
    assert "BTC: 60.00%" in output
    assert "ETH: 30.00%" in output
    assert "SOL: 10.00%" in output

    # Check correct sorting (highest weight first)
    btc_pos = output.find("BTC")
    eth_pos = output.find("ETH")
    sol_pos = output.find("SOL")
    assert btc_pos < eth_pos < sol_pos


def test_display_initial_weights(sample_historical_data):
    """Test display_initial_weights function."""
    methods = ["market_cap", "sqrt_market_cap"]

    # Capture stdout to check the output
    with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
        display_initial_weights(sample_historical_data, methods)
        output = fake_stdout.getvalue()

    # Check output contains expected content for both methods
    assert "Initial weights for Market Cap:" in output
    assert "Initial weights for Sqrt Market Cap:" in output

    # BTC should have largest weight in market cap method
    assert "BTC:" in output


def test_display_initial_weights_empty_data():
    """Test display_initial_weights with empty data."""
    empty_data = {}

    # Capture stdout to check the output
    with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
        display_initial_weights(empty_data, ["market_cap"])
        output = fake_stdout.getvalue()

    # Check error message
    assert "No data available to calculate weights" in output
