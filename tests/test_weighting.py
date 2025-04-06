"""
Unit tests for the weighting module.
"""

import math
from io import StringIO
from unittest.mock import patch

import pytest

from core.weighting import (
    calculate_index_weights,
    calculate_weight_market_cap,
    calculate_weight_sqrt_market_cap,
    display_initial_weights,
    normalize_weights,
    print_portfolio_weights,
)


@pytest.fixture
def sample_market_caps():
    """Sample market cap data for testing."""
    return {
        "btc": 600000000000,  # 600B
        "eth": 100000000000,  # 100B
        "sol": 1000000000,  # 1B
    }


@pytest.fixture
def sample_weights():
    """Sample weight data for testing."""
    return {"btc": 0.6, "eth": 0.3, "sol": 0.1}


@pytest.fixture
def sample_historical_data():
    """Sample historical data for testing."""
    timestamp1 = 1609459200000  # 2021-01-01
    timestamp2 = 1609545600000  # 2021-01-02

    return {
        "btc": [
            [timestamp1, 30000, 600000000000],  # ts, price, mcap
            [timestamp2, 31000, 620000000000],
        ],
        "eth": [
            [timestamp1, 800, 100000000000],
            [timestamp2, 850, 106000000000],
        ],
        "sol": [
            [timestamp1, 10, 1000000000],
            [timestamp2, 11, 1100000000],
        ],
    }


def test_calculate_weight_market_cap(sample_market_caps):
    """Test calculating weights based on market cap."""
    weights = calculate_weight_market_cap(sample_market_caps)

    # Verify the weights directly use market cap values
    assert weights["btc"] == 600000000000
    assert weights["eth"] == 100000000000
    assert weights["sol"] == 1000000000


def test_calculate_weight_sqrt_market_cap(sample_market_caps):
    """Test calculating weights based on square root of market cap."""
    weights = calculate_weight_sqrt_market_cap(sample_market_caps)

    # Verify the weights use square root of market cap values
    assert weights["btc"] == math.sqrt(600000000000)
    assert weights["eth"] == math.sqrt(100000000000)
    assert weights["sol"] == math.sqrt(1000000000)


def test_normalize_weights(sample_weights):
    """Test normalizing weights to sum to 1.0."""
    # Create unnormalized weights
    unnormalized = {"btc": 600, "eth": 300, "sol": 100}

    # Normalize
    normalized = normalize_weights(unnormalized)

    # Check that weights sum to 1.0
    assert sum(normalized.values()) == pytest.approx(1.0)

    # Check proportions are preserved
    assert normalized["btc"] == pytest.approx(0.6)
    assert normalized["eth"] == pytest.approx(0.3)
    assert normalized["sol"] == pytest.approx(0.1)

    # Test with a single asset
    single_asset = {"btc": 500}
    normalized_single = normalize_weights(single_asset)
    assert normalized_single["btc"] == 1.0


def test_calculate_index_weights_market_cap(sample_market_caps):
    """Test calculating index weights using market cap method."""
    weights = calculate_index_weights(sample_market_caps, "market_cap")

    total_mcap = sum(sample_market_caps.values())

    # Verify normalized weights
    assert weights["btc"] == pytest.approx(600000000000 / total_mcap)
    assert weights["eth"] == pytest.approx(100000000000 / total_mcap)
    assert weights["sol"] == pytest.approx(1000000000 / total_mcap)

    # Ensure weights sum to 1.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_calculate_index_weights_sqrt_market_cap(sample_market_caps):
    """Test calculating index weights using square root of market cap method."""
    weights = calculate_index_weights(sample_market_caps, "sqrt_market_cap")

    sqrt_btc = math.sqrt(sample_market_caps["btc"])
    sqrt_eth = math.sqrt(sample_market_caps["eth"])
    sqrt_sol = math.sqrt(sample_market_caps["sol"])
    total_sqrt = sqrt_btc + sqrt_eth + sqrt_sol

    # Verify normalized weights
    assert weights["btc"] == pytest.approx(sqrt_btc / total_sqrt)
    assert weights["eth"] == pytest.approx(sqrt_eth / total_sqrt)
    assert weights["sol"] == pytest.approx(sqrt_sol / total_sqrt)

    # Ensure weights sum to 1.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_calculate_index_weights_invalid_method(sample_market_caps):
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        calculate_index_weights(sample_market_caps, "invalid_method")

    assert "Unknown weighting method" in str(excinfo.value)


def test_print_portfolio_weights(sample_weights):
    """Test printing portfolio weights."""
    with patch("sys.stdout", new=StringIO()) as fake_output:
        print_portfolio_weights(sample_weights)
        output = fake_output.getvalue()

        # Check for header and all tokens
        assert "=== Portfolio Weights ===" in output
        assert "BTC: 60.00%" in output
        assert "ETH: 30.00%" in output
        assert "SOL: 10.00%" in output

        # Check order (should be descending by weight)
        btc_pos = output.find("BTC")
        eth_pos = output.find("ETH")
        sol_pos = output.find("SOL")

        assert btc_pos < eth_pos < sol_pos


def test_display_initial_weights(sample_historical_data):
    """Test displaying initial weights for different methods."""
    methods = ["market_cap", "sqrt_market_cap"]

    with patch("core.weighting.print_portfolio_weights") as mock_print:
        display_initial_weights(sample_historical_data, methods)

        # Should be called once for each method
        assert mock_print.call_count == 2

        # Check weights for market_cap method (first call)
        market_cap_weights = mock_print.call_args_list[0][0][0]
        assert (
            market_cap_weights["btc"]
            > market_cap_weights["eth"]
            > market_cap_weights["sol"]
        )
        assert sum(market_cap_weights.values()) == pytest.approx(1.0)

        # Check weights for sqrt_market_cap method (second call)
        sqrt_weights = mock_print.call_args_list[1][0][0]
        assert sqrt_weights["btc"] > sqrt_weights["eth"] > sqrt_weights["sol"]
        assert sum(sqrt_weights.values()) == pytest.approx(1.0)

        # The square root method should give more weight to smaller assets compared to direct market cap
        assert (
            sqrt_weights["sol"] / sqrt_weights["btc"]
            > market_cap_weights["sol"] / market_cap_weights["btc"]
        )


def test_display_initial_weights_empty_data():
    """Test displaying weights with empty data."""
    methods = ["market_cap"]

    with patch("builtins.print") as mock_print:
        display_initial_weights({}, methods)

        # Should print an error message
        mock_print.assert_called_with("No data available to calculate weights")


def test_display_initial_weights_with_timestamps():
    """Test that display_initial_weights uses the earliest timestamp."""
    # Create data with different timestamps
    timestamp1 = 1609459200000  # 2021-01-01
    timestamp2 = 1609545600000  # 2021-01-02

    data = {
        "btc": [
            [timestamp2, 31000, 620000000000],  # Later timestamp
            [timestamp1, 30000, 600000000000],  # Earlier timestamp
        ],
        "eth": [
            [timestamp1, 800, 100000000000],
            [timestamp2, 850, 106000000000],
        ],
    }

    methods = ["market_cap"]

    with patch("core.weighting.calculate_index_weights") as mock_calculate:
        mock_calculate.return_value = {"btc": 0.8, "eth": 0.2}

        with patch("core.weighting.print_portfolio_weights"):
            display_initial_weights(data, methods)

            # Should use market caps from the earlier timestamp (timestamp1)
            expected_market_caps = {"btc": 600000000000, "eth": 100000000000}

            # Check that calculate_index_weights was called with the right market caps
            mock_calculate.assert_called_with(expected_market_caps, "market_cap")
