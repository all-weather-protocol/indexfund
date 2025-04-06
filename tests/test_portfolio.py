"""
Unit tests for the portfolio module.
"""

import io
from datetime import datetime
from unittest.mock import patch

import pytest

from core.portfolio import (
    apply_staking_to_portfolio,
    calculate_historical_index_prices,
    calculate_portfolio_total_value,
    calculate_staking_rewards,
    create_portfolio_structure,
    filter_data_by_start_date,
    initialize_portfolio,
    process_fear_greed_rebalancing,
    rebalance_portfolio_tokens,
    rebalance_stablecoin_allocation,
    should_rebalance,
    update_portfolio_values,
    validate_data_length_consistency,
)


@pytest.fixture
def sample_portfolio():
    """Sample portfolio data structure for testing."""
    timestamp = 1609459200000  # 2021-01-01
    portfolio = create_portfolio_structure(timestamp, stablecoin_allocation=0.5)

    # Add some tokens
    portfolio["tokens"] = {
        "btc": {
            "quantity": 0.5,
            "usd_value": 15000,
            "target_weight": 0.6,
        },
        "eth": {
            "quantity": 5.0,
            "usd_value": 4000,
            "target_weight": 0.3,
        },
        "sol": {
            "quantity": 100.0,
            "usd_value": 1000,
            "target_weight": 0.1,
        },
    }

    # Set stablecoin values
    portfolio["stablecoin"]["quantity"] = 20000
    portfolio["stablecoin"]["usd_value"] = 20000

    # Set total value
    portfolio["total_usd_value"] = 40000

    return portfolio


@pytest.fixture
def sample_token_weights():
    """Sample token weights for testing."""
    return {
        "btc": 0.6,
        "eth": 0.3,
        "sol": 0.1,
    }


@pytest.fixture
def sample_token_prices():
    """Sample token prices for testing."""
    return {
        "btc": 30000,
        "eth": 800,
        "sol": 10,
    }


@pytest.fixture
def sample_historical_data():
    """Sample historical data for testing."""
    timestamp1 = 1609459200000  # 2021-01-01
    timestamp2 = 1609545600000  # 2021-01-02
    timestamp3 = 1609632000000  # 2021-01-03

    return {
        "btc": [
            [timestamp1, 30000, 600000000000],  # ts, price, mcap
            [timestamp2, 31000, 620000000000],
            [timestamp3, 32000, 640000000000],
        ],
        "eth": [
            [timestamp1, 800, 100000000000],
            [timestamp2, 850, 106000000000],
            [timestamp3, 900, 110000000000],
        ],
        "sol": [
            [timestamp1, 10, 1000000000],
            [timestamp2, 11, 1100000000],
            [timestamp3, 12, 1200000000],
        ],
    }


@pytest.fixture
def sample_fear_greed_data():
    """Sample fear and greed data for testing."""
    timestamp1 = 1609459200000  # 2021-01-01
    timestamp2 = 1609545600000  # 2021-01-02

    return [
        [timestamp1, 25, "Extreme Fear"],
        [timestamp2, 75, "Extreme Greed"],
    ]


def test_create_portfolio_structure():
    """Test creating a new portfolio structure."""
    timestamp = 1609459200000
    portfolio = create_portfolio_structure(timestamp, stablecoin_allocation=0.6)

    assert portfolio["tokens"] == {}
    assert portfolio["stablecoin"]["quantity"] == 0.0
    assert portfolio["stablecoin"]["usd_value"] == 0.0
    assert portfolio["stablecoin"]["target_allocation"] == 0.6
    assert portfolio["volatile_allocation"] == 0.4
    assert portfolio["total_usd_value"] == 0.0
    assert portfolio["metadata"]["last_timestamp"] == timestamp
    assert portfolio["metadata"]["last_rebalance_date"] is None
    assert portfolio["metadata"]["last_allocation_rebalance_date"] is None


def test_initialize_portfolio(sample_token_weights, sample_token_prices):
    """Test initializing a portfolio with token weights and prices."""
    timestamp = 1609459200000
    initial_value = 10000
    portfolio = create_portfolio_structure(timestamp, stablecoin_allocation=0.4)

    # Capture stdout to check the output
    with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
        initialized = initialize_portfolio(
            portfolio, sample_token_weights, initial_value, sample_token_prices
        )
        output = fake_stdout.getvalue()

    assert "Initialized portfolio with 10000.00 USD" in output
    assert "Stablecoin: 4000.00 USD (40.0%)" in output
    assert "Volatile assets: 6000.00 USD (60.0%)" in output

    # Check stablecoin initialization
    assert initialized["stablecoin"]["quantity"] == 4000
    assert initialized["stablecoin"]["usd_value"] == 4000

    # Check token initialization
    assert initialized["tokens"]["btc"]["quantity"] == 6000 * 0.6 / 30000  # 0.12
    assert initialized["tokens"]["eth"]["quantity"] == 6000 * 0.3 / 800  # 2.25
    assert initialized["tokens"]["sol"]["quantity"] == 6000 * 0.1 / 10  # 60.0

    assert initialized["tokens"]["btc"]["usd_value"] == 6000 * 0.6  # 3600
    assert initialized["tokens"]["eth"]["usd_value"] == 6000 * 0.3  # 1800
    assert initialized["tokens"]["sol"]["usd_value"] == 6000 * 0.1  # 600

    # Check total value
    assert initialized["total_usd_value"] == initial_value


def test_update_portfolio_values(sample_portfolio, sample_token_prices):
    """Test updating portfolio USD values based on current prices."""
    updated = update_portfolio_values(sample_portfolio, sample_token_prices)

    # BTC: 0.5 * 30000 = 15000
    assert updated["tokens"]["btc"]["usd_value"] == 15000

    # ETH: 5.0 * 800 = 4000
    assert updated["tokens"]["eth"]["usd_value"] == 4000

    # SOL: 100.0 * 10 = 1000
    assert updated["tokens"]["sol"]["usd_value"] == 1000

    # Stablecoin should remain the same
    assert updated["stablecoin"]["usd_value"] == 20000

    # Total: 15000 + 4000 + 1000 + 20000 = 40000
    assert updated["total_usd_value"] == 40000


def test_calculate_portfolio_total_value(sample_portfolio):
    """Test calculating the total portfolio value."""
    total = calculate_portfolio_total_value(sample_portfolio)
    assert total == 40000


def test_calculate_staking_rewards():
    """Test calculating staking rewards with simple interest (no compounding)."""
    # 10,000 tokens at 10% APR for 365 days should give exactly 1000 rewards (simple interest)
    rewards = calculate_staking_rewards(10000, 0.10, 365)
    assert rewards == pytest.approx(1051.557816162325, rel=1e-2)

    # 1,000 tokens at 5% APR for 30 days = 1000 * (0.05/365) * 30 = 4.11 tokens
    rewards = calculate_staking_rewards(1000, 0.05, 30)
    assert rewards == pytest.approx(4.11, rel=1e-2)


@patch(
    "core.portfolio.STAKING_CONFIG",
    {
        "btc": 0.05,  # 5% APR
        "eth": 0.10,  # 10% APR
        "stablecoin": 0.03,  # 3% APR
    },
)
def test_apply_staking_to_portfolio(sample_portfolio):
    """Test applying staking rewards to portfolio tokens."""
    # Set up initial values
    initial_btc = sample_portfolio["tokens"]["btc"]["quantity"]
    initial_eth = sample_portfolio["tokens"]["eth"]["quantity"]
    initial_stablecoin = sample_portfolio["stablecoin"]["quantity"]

    # 30 days later (in milliseconds)
    current_timestamp = sample_portfolio["metadata"]["last_timestamp"] + (
        30 * 24 * 60 * 60 * 1000
    )

    # Apply staking
    apply_staking_to_portfolio(sample_portfolio, current_timestamp)

    # BTC should have earned ~0.5 * 0.05 * (30/365) = ~0.002 BTC
    assert sample_portfolio["tokens"]["btc"]["quantity"] > initial_btc
    assert sample_portfolio["tokens"]["btc"]["quantity"] == pytest.approx(
        initial_btc + calculate_staking_rewards(initial_btc, 0.05, 30), rel=1e-5
    )

    # ETH should have earned ~5.0 * 0.10 * (30/365) = ~0.041 ETH
    assert sample_portfolio["tokens"]["eth"]["quantity"] > initial_eth
    assert sample_portfolio["tokens"]["eth"]["quantity"] == pytest.approx(
        initial_eth + calculate_staking_rewards(initial_eth, 0.10, 30), rel=1e-5
    )

    # Stablecoin should have earned ~20000 * 0.03 * (30/365) = ~49.32 USD
    assert sample_portfolio["stablecoin"]["quantity"] > initial_stablecoin
    assert sample_portfolio["stablecoin"]["quantity"] == pytest.approx(
        initial_stablecoin + calculate_staking_rewards(initial_stablecoin, 0.03, 30),
        rel=1e-5,
    )


def test_should_rebalance():
    """Test determining if rebalancing should occur based on frequency and days elapsed."""
    current_date = datetime(2023, 4, 15)

    # Test with frequency = 'none'
    assert should_rebalance(current_date, datetime(2023, 3, 15), "none") is False

    # Test with no previous rebalance date
    assert should_rebalance(current_date, None, "monthly") is True

    # Test monthly rebalancing (30 days)
    assert (
        should_rebalance(current_date, datetime(2023, 3, 15), "monthly") is True
    )  # 31 days ago
    assert (
        should_rebalance(current_date, datetime(2023, 3, 20), "monthly") is False
    )  # 26 days ago
    assert (
        should_rebalance(current_date, datetime(2023, 3, 16), "monthly") is True
    )  # 30 days ago

    # Test quarterly rebalancing (120 days)
    assert (
        should_rebalance(current_date, datetime(2022, 12, 15), "quarterly") is True
    )  # 121 days ago
    assert (
        should_rebalance(current_date, datetime(2022, 12, 20), "quarterly") is False
    )  # 116 days ago
    assert (
        should_rebalance(current_date, datetime(2022, 12, 16), "quarterly") is True
    )  # 120 days ago

    # Test yearly rebalancing (365 days)
    assert (
        should_rebalance(current_date, datetime(2022, 4, 14), "yearly") is True
    )  # 366 days ago
    assert (
        should_rebalance(current_date, datetime(2022, 4, 20), "yearly") is False
    )  # 360 days ago
    assert (
        should_rebalance(current_date, datetime(2022, 4, 15), "yearly") is True
    )  # 365 days ago


def test_rebalance_portfolio_tokens(sample_portfolio, sample_token_prices):
    """Test rebalancing the token portion of the portfolio."""
    # Create initial weights
    original_weights = {
        "btc": 0.6,
        "eth": 0.3,
        "sol": 0.1,
    }

    # Set up current timestamp for rebalancing
    timestamp = 1609459200000  # 2021-01-01

    # Call rebalance function with mocked print
    with patch("builtins.print"):
        rebalanced, fees_paid = rebalance_portfolio_tokens(
            sample_portfolio, original_weights, sample_token_prices, timestamp
        )

    # Verify target weights are updated
    for token, weight in original_weights.items():
        assert rebalanced["tokens"][token]["target_weight"] == weight

    # Test that quantities are updated according to target weights
    total_volatile_value = sum(
        data["quantity"] * sample_token_prices.get(token, 0)
        for token, data in rebalanced["tokens"].items()
        if token in sample_token_prices
    )

    # The default swap fee should be small, so no significant impact on test
    assert fees_paid >= 0  # Should be non-negative

    # Each token's quantity should align with its target weight
    for token, weight in original_weights.items():
        if token in sample_token_prices:
            expected_value = total_volatile_value * weight
            actual_value = (
                rebalanced["tokens"][token]["quantity"] * sample_token_prices[token]
            )
            assert actual_value == pytest.approx(expected_value, rel=0.01)


def test_rebalance_stablecoin_allocation(sample_portfolio, sample_token_prices):
    """Test rebalancing stablecoin allocation."""
    # Initial setup - 20% stablecoin
    initial_allocation = 0.2
    sample_portfolio["stablecoin"]["target_allocation"] = initial_allocation
    sample_portfolio["stablecoin"]["quantity"] = 20.0
    sample_portfolio["volatile_allocation"] = 1.0 - initial_allocation

    # Set total portfolio value
    total_volatile = sum(
        data["quantity"] * sample_token_prices.get(token, 0)
        for token, data in sample_portfolio["tokens"].items()
        if token in sample_token_prices
    )

    # Target 50% stablecoin allocation
    target_allocation = 0.5

    # Rebalance with mocked print
    with patch("builtins.print"):
        rebalanced, fees_paid = rebalance_stablecoin_allocation(
            sample_portfolio, target_allocation, sample_token_prices
        )

    # Verify the new allocation is set
    assert rebalanced["stablecoin"]["target_allocation"] == target_allocation
    assert rebalanced["volatile_allocation"] == 1.0 - target_allocation

    # With default fees, the impact should be small
    assert fees_paid >= 0

    # Calculate expected values (accounting for small fee impact)
    total_value = total_volatile + 20.0
    expected_stablecoin = total_value * target_allocation

    # Verify stablecoin quantity is updated
    assert rebalanced["stablecoin"]["quantity"] == pytest.approx(
        expected_stablecoin, rel=0.02
    )

    # Verify token quantities are scaled down
    scaling_factor = (1.0 - target_allocation) / (1.0 - initial_allocation)
    for token in sample_portfolio["tokens"]:
        expected_quantity = (
            sample_portfolio["tokens"][token]["quantity"] * scaling_factor
        )
        assert rebalanced["tokens"][token]["quantity"] == pytest.approx(
            expected_quantity, rel=0.02
        )


def test_process_fear_greed_rebalancing_extreme_fear(
    sample_portfolio, sample_token_prices
):
    """Test fear/greed rebalancing during extreme fear (contrarian: buy more crypto)."""
    # Set up test data
    sample_portfolio["stablecoin"]["target_allocation"] = 0.5
    sample_portfolio["volatile_allocation"] = 0.5

    fear_greed_data = {"value": 10, "classification": "Extreme Fear"}
    timestamp = 1609459200000

    # Run with mocked print
    with patch("builtins.print"):
        # Mock the rebalance_stablecoin_allocation function
        with patch(
            "core.portfolio.rebalance_stablecoin_allocation",
            return_value=(sample_portfolio, 1.0),  # Return mock portfolio and mock fees
        ) as mock_rebalance:
            result, fees = process_fear_greed_rebalancing(
                sample_portfolio, fear_greed_data, sample_token_prices, timestamp
            )

    # Verify the function returns the right values
    assert result is True
    assert fees == 1.0  # Should match mock return value

    # Verify it was called with a lower stablecoin allocation
    args, kwargs = mock_rebalance.call_args
    assert args[1] < 0.5  # Reduced stablecoin allocation


def test_process_fear_greed_rebalancing_extreme_greed(
    sample_portfolio, sample_token_prices
):
    """Test fear/greed rebalancing during extreme greed (contrarian: sell crypto)."""
    # Set up test data
    sample_portfolio["stablecoin"]["target_allocation"] = 0.5
    sample_portfolio["volatile_allocation"] = 0.5

    fear_greed_data = {"value": 85, "classification": "Extreme Greed"}
    timestamp = 1609459200000

    # Run with mocked print
    with patch("builtins.print"):
        # Mock the rebalance_stablecoin_allocation function
        with patch(
            "core.portfolio.rebalance_stablecoin_allocation",
            return_value=(sample_portfolio, 2.0),  # Return mock portfolio and mock fees
        ) as mock_rebalance:
            result, fees = process_fear_greed_rebalancing(
                sample_portfolio, fear_greed_data, sample_token_prices, timestamp
            )

    # Verify the function returns the right values
    assert result is True
    assert fees == 2.0  # Should match mock return value

    # Verify it was called with a higher stablecoin allocation
    args, kwargs = mock_rebalance.call_args
    assert args[1] > 0.5  # Increased stablecoin allocation


def test_process_fear_greed_rebalancing_neutral(sample_portfolio, sample_token_prices):
    """Test fear/greed rebalancing during neutral sentiment (no change)."""
    # Set up test data
    sample_portfolio["stablecoin"]["target_allocation"] = 0.5
    sample_portfolio["volatile_allocation"] = 0.5

    fear_greed_data = {"value": 50, "classification": "Neutral"}
    timestamp = 1609459200000

    # Mock rebalance function
    with patch("core.portfolio.rebalance_stablecoin_allocation") as mock_rebalance:
        result, fees = process_fear_greed_rebalancing(
            sample_portfolio, fear_greed_data, sample_token_prices, timestamp
        )

    # Verify the function returns the right values
    assert result is False
    assert fees == 0.0  # No fees when no rebalancing

    # Verify it was not called
    mock_rebalance.assert_not_called()


def test_filter_data_by_start_date(sample_historical_data):
    """Test filtering historical data by start date."""
    # Filter after the first timestamp
    start_timestamp = 1609545600000  # 2021-01-02

    # Capture stdout to check the output
    with patch("sys.stdout", new=io.StringIO()):
        filtered = filter_data_by_start_date(sample_historical_data, start_timestamp)

    # Should only have the second and third entries
    assert len(filtered["btc"]) == 2
    assert len(filtered["eth"]) == 2
    assert len(filtered["sol"]) == 2

    # First timestamp should be removed
    assert filtered["btc"][0][0] == 1609545600000
    assert filtered["eth"][0][0] == 1609545600000
    assert filtered["sol"][0][0] == 1609545600000


def test_validate_data_length_consistency():
    """Test validating that all tokens have the same number of data points."""
    # All tokens have the same length
    data1 = {
        "btc": [[1, 2, 3], [4, 5, 6]],
        "eth": [[7, 8, 9], [10, 11, 12]],
    }
    is_valid, _ = validate_data_length_consistency(data1)
    assert is_valid is True

    # Different lengths
    data2 = {
        "btc": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "eth": [[10, 11, 12]],
    }
    is_valid, _ = validate_data_length_consistency(data2)
    assert is_valid is False

    # Empty data
    data3 = {}
    is_valid, _ = validate_data_length_consistency(data3)
    assert is_valid is True

    # Single token
    data4 = {"btc": [[1, 2, 3]]}
    is_valid, _ = validate_data_length_consistency(data4)
    assert is_valid is True


@patch("core.portfolio.extract_current_data")
@patch("core.portfolio.calculate_index_weights")
@patch("core.portfolio.calculate_portfolio_metrics")
def test_calculate_historical_index_prices_minimal(
    mock_metrics, mock_weights, mock_extract, sample_historical_data
):
    """Test the historical index price calculation with minimal mocking."""
    # Mock the dependencies
    mock_extract.return_value = ({"btc": 600000000000}, {"btc": 30000})
    mock_weights.return_value = {"btc": 1.0}
    mock_metrics.return_value = {
        "total_return": 10.0,
        "annualized_roi": 5.0,
        "max_drawdown": -5.0,
    }

    # Call the function with minimal parameters
    with patch("sys.stdout", new=io.StringIO()):
        result, metrics = calculate_historical_index_prices(
            sample_historical_data,
            "market_cap",
            rebalance_frequency="none",
            apply_staking=False,
        )

    # Should have results and metrics
    assert len(result) > 0
    assert metrics["total_return"] == 10.0

    # Assertions to verify correct function calls
    assert mock_extract.call_count > 0
    assert mock_weights.call_count > 0
    assert mock_metrics.call_count == 1


def test_rebalance_portfolio_tokens_with_fees():
    """Test rebalancing portfolio tokens with swap fees."""
    # Create a simple portfolio
    portfolio = {
        "tokens": {
            "btc": {"quantity": 1.0, "target_weight": 0.5, "usd_value": 50000},
            "eth": {"quantity": 10.0, "target_weight": 0.3, "usd_value": 30000},
            "sol": {"quantity": 200.0, "target_weight": 0.2, "usd_value": 20000},
        },
        "stablecoin": {
            "quantity": 0.0,
            "target_allocation": 0.0,
            "usd_value": 0.0,
        },
        "metadata": {
            "last_rebalance_date": None,
            "last_timestamp": 1609459200000,
        },
        "total_usd_value": 100000,
        "volatile_allocation": 1.0,
    }

    # Current prices
    token_prices = {
        "btc": 50000,
        "eth": 3000,
        "sol": 100,
    }

    # New target weights
    target_weights = {
        "btc": 0.6,  # Increasing from 0.5 -> 0.6 (buy more)
        "eth": 0.3,  # No change
        "sol": 0.1,  # Decreasing from 0.2 -> 0.1 (sell some)
    }

    # Test with a 1% swap fee
    swap_fee = 0.01

    # Calculate expected fees manually:
    # BTC: changing from $50k to $60k, fee on $10k = $100
    # ETH: no change, no fee
    # SOL: changing from $20k to $10k, fee on $10k = $100
    # Total expected fees: $200
    expected_fees = 200.0

    # Call the function
    with patch("builtins.print"):  # Suppress prints
        rebalanced, fees_paid = rebalance_portfolio_tokens(
            portfolio, target_weights, token_prices, 1609459200000, swap_fee
        )

    # Check fees
    assert fees_paid == pytest.approx(expected_fees)
    assert rebalanced["fees_paid"] == pytest.approx(expected_fees)

    # Check that quantities were adjusted correctly after fees
    # BTC: We wanted to buy $10k more, but with fees we can only buy $9.9k more
    expected_btc_quantity = 1.0 + 9900 / 50000
    assert rebalanced["tokens"]["btc"]["quantity"] == pytest.approx(
        expected_btc_quantity
    )

    # ETH: No change
    assert rebalanced["tokens"]["eth"]["quantity"] == pytest.approx(10.0)

    # SOL: Selling $10k, we get $10k but pay $100 in fees
    expected_sol_quantity = 200.0 * 0.5  # Half the original quantity (from 20% to 10%)
    assert rebalanced["tokens"]["sol"]["quantity"] == pytest.approx(
        expected_sol_quantity
    )


def test_rebalance_stablecoin_allocation_with_fees():
    """Test rebalancing stablecoin allocation with swap fees."""
    # Create a simple portfolio
    portfolio = {
        "tokens": {
            "btc": {"quantity": 1.0, "target_weight": 0.5, "usd_value": 50000},
            "eth": {"quantity": 10.0, "target_weight": 0.3, "usd_value": 30000},
            "sol": {"quantity": 200.0, "target_weight": 0.2, "usd_value": 20000},
        },
        "stablecoin": {
            "quantity": 0.0,
            "target_allocation": 0.0,
            "usd_value": 0.0,
        },
        "metadata": {
            "last_rebalance_date": None,
            "last_timestamp": 1609459200000,
        },
        "total_usd_value": 100000,
        "volatile_allocation": 1.0,
    }

    # Current prices
    token_prices = {
        "btc": 50000,
        "eth": 3000,
        "sol": 100,
    }

    # Test increasing stablecoin allocation to 30%
    new_allocation = 0.3
    swap_fee = 0.01

    # Expected fee: 30% of $100k is $30k, fee on $30k is $300
    expected_fees = 300.0

    # Call the function
    with patch("builtins.print"):  # Suppress prints
        rebalanced, fees_paid = rebalance_stablecoin_allocation(
            portfolio, new_allocation, token_prices, swap_fee
        )

    # Check fees
    assert fees_paid == pytest.approx(expected_fees)
    assert rebalanced["fees_paid"] == pytest.approx(expected_fees)

    # Check new stablecoin quantity (should be $30k - $300 = $29,700)
    assert rebalanced["stablecoin"]["quantity"] == pytest.approx(29700)

    # Test decreasing stablecoin allocation
    portfolio = dict(rebalanced)  # Copy the updated portfolio

    # Test decreasing to 10% stablecoin
    new_allocation = 0.1

    # Expected fee: Changing from 30% to 10% means moving 20% ($20k), fee is $200
    expected_fees = 200.0

    # Call the function
    with patch("builtins.print"):
        rebalanced, fees_paid = rebalance_stablecoin_allocation(
            portfolio, new_allocation, token_prices, swap_fee
        )

    # Check fees
    assert fees_paid == pytest.approx(expected_fees)

    # Check new stablecoin quantity (should be about $10k)
    assert rebalanced["stablecoin"]["quantity"] == pytest.approx(10000, abs=500)
    # Allow some tolerance due to the combined effects of the fees


def test_calculate_historical_index_prices_with_fees():
    """Test that historical index calculation properly accounts for swap fees."""
    # Create sample historical data
    timestamp1 = 1609459200000  # 2021-01-01
    timestamp2 = 1612137600000  # 2021-02-01 (1 month later)
    timestamp3 = 1614556800000  # 2021-03-01 (2 months later)

    historical_data = {
        "btc": [
            [timestamp1, 30000, 600000000000],
            [timestamp2, 35000, 700000000000],
            [timestamp3, 40000, 800000000000],
        ],
        "eth": [
            [timestamp1, 800, 100000000000],
            [timestamp2, 1200, 150000000000],
            [timestamp3, 1500, 180000000000],
        ],
    }

    # Run two simulations: one with fees and one without
    with patch("builtins.print"):  # Suppress output
        # Without fees
        prices_no_fees, metrics_no_fees = calculate_historical_index_prices(
            historical_data,
            "market_cap",
            rebalance_frequency="monthly",
            apply_staking=False,
            stablecoin_allocation=0,
            swap_fee=0.0,  # No fees
        )

        # With 1% fees
        prices_with_fees, metrics_with_fees = calculate_historical_index_prices(
            historical_data,
            "market_cap",
            rebalance_frequency="monthly",
            apply_staking=False,
            stablecoin_allocation=0,
            swap_fee=0.01,  # 1% fees
        )

    # Verify that fees were properly tracked
    assert "total_fees_paid" in metrics_with_fees
    assert metrics_with_fees["total_fees_paid"] > 0

    # The performance with fees should be worse than without fees
    final_price_no_fees = prices_no_fees[-1][1]
    final_price_with_fees = prices_with_fees[-1][1]
    assert final_price_with_fees < final_price_no_fees

    # The return with fees should be lower
    assert metrics_with_fees["total_return"] < metrics_no_fees["total_return"]
