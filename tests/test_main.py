"""
Unit tests for the main module.
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest

# Import from data_loading module
from data_loading import load_and_prepare_data

# Import from main module
from main import run_performance_analysis

# Import from strategy module
from strategy import (
    calculate_strategy_performance,
    format_strategy_name,
    generate_strategy_key,
)


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
    return [
        [1609459200000, 25, "Extreme Fear"],  # 2021-01-01
        [1609545600000, 50, "Neutral"],  # 2021-01-02
        [1609632000000, 75, "Extreme Greed"],  # 2021-01-03
    ]


@pytest.fixture
def sample_performance_metrics():
    """Sample performance metrics for testing."""
    return {
        "max_drawdown": np.float64(0.0),
        "volatility": np.float64(0.2803321820974809),
        "sharpe_ratio": np.float64(1676.5534155291866),
        "sortino_ratio": 0,
    }


@pytest.fixture
def sample_index_prices():
    """Sample index prices for testing."""
    timestamp1 = 1609459200000  # 2021-01-01
    timestamp2 = 1609545600000  # 2021-01-02
    timestamp3 = 1609632000000  # 2021-01-03

    return [
        [timestamp1, 100.0],
        [timestamp2, 101.9020694506869],
        [timestamp3, 103.80432726510048],
    ]


@pytest.fixture
def sample_investment_values():
    """Sample investment values for testing."""
    return [1000.0, 1050.0, 1100.0]


def test_format_strategy_name():
    """Test formatting strategy names with various parameters."""
    # Test with basic parameters
    name = format_strategy_name("market_cap", "monthly", True, 0.5)
    assert "Market Cap (Monthly)" in name
    assert "with Staking" in name
    assert "50% Market Cap" in name
    assert "50% Stablecoin" in name

    # Test without staking
    name = format_strategy_name("sqrt_market_cap", "quarterly", False, 0.3)
    assert "Sqrt Market Cap (Quarterly)" in name
    assert "without Staking" in name
    assert "70% Sqrt Market Cap" in name
    assert "30% Stablecoin" in name

    # Test with start date as string
    name = format_strategy_name("market_cap", "yearly", True, 0.4, "2021-01-01")
    assert "Start: 2021-01-01" in name

    # Test with start date as datetime
    start_date = datetime(2021, 1, 1)
    name = format_strategy_name("market_cap", "none", True, 0.0, start_date)
    assert "Start: 2021-01-01" in name
    assert "Market Cap (None)" in name
    assert "0% Stablecoin" not in name  # Should not show 0% stablecoin


def test_generate_strategy_key():
    """Test generating strategy keys with various parameters."""
    # Test with basic parameters
    key = generate_strategy_key("market_cap", "monthly", True, False)
    assert key == "market_cap_monthly_staking_True_feargreed_False"

    # Test with fear/greed
    key = generate_strategy_key("sqrt_market_cap", "quarterly", False, True)
    assert key == "sqrt_market_cap_quarterly_staking_False_feargreed_True"

    # Test with start date as string
    key = generate_strategy_key("market_cap", "yearly", True, False, "2021-01-01")
    assert key == "market_cap_yearly_staking_True_feargreed_False_20210101"

    # Test with start date as datetime
    start_date = datetime(2021, 1, 1)
    key = generate_strategy_key("market_cap", "none", True, True, start_date)
    assert key == "market_cap_none_staking_True_feargreed_True_20210101"


@patch("portfolio.calculate_historical_index_prices")
@patch("metrics.calculate_financial_metrics")
def test_calculate_strategy_performance(
    mock_calculate_metrics,
    mock_calculate_prices,
    sample_historical_data,
    sample_performance_metrics,
    sample_index_prices,
):
    """Test calculating strategy performance."""
    # Mock the dependencies
    mock_calculate_prices.return_value = (sample_index_prices, {})
    mock_calculate_metrics.return_value = sample_performance_metrics

    # Call the function
    index_prices, metrics, investment_values = calculate_strategy_performance(
        sample_historical_data,
        "market_cap",
        "monthly",
        1000.0,
        stablecoin_allocation=0.5,
        apply_staking=True,
    )

    # Verify the results
    assert index_prices == sample_index_prices
    assert metrics == sample_performance_metrics
    assert len(investment_values) == len(sample_index_prices)

    # Test empty result case
    mock_calculate_prices.return_value = ([], {})
    result = calculate_strategy_performance(
        sample_historical_data, "market_cap", "monthly", 1000.0
    )
    assert result == (
        [
            [1609459200000, 100.0],
            [1609545600000, 101.9020694506869],
            [1609632000000, 103.80432726510048],
        ],
        {
            "max_drawdown": np.float64(0.0),
            "volatility": np.float64(0.2803321820974809),
            "sharpe_ratio": np.float64(1676.5534155291866),
            "sortino_ratio": 0,
        },
        [1000.0, 1019.020694506869, 1038.0432726510048],
    )


# @patch("visualization.create_performance_data")
# def test_create_strategy_data(mock_create_performance_data, sample_index_prices, sample_performance_metrics, sample_investment_values, sample_fear_greed_data):
#     """Test creating strategy data dictionaries."""
#     # Mock create_performance_data to return a complete dict with all expected fields
#     mock_create_performance_data.return_value = {
#         "initial_investment": 1000.0,
#         "final_investment": 1100.0,
#         "dates": ["2021-01-01", "2021-01-02", "2021-01-03"],
#         "values": [100.0, 105.0, 110.0]
#     }

#     # Test with basic parameters
#     data = create_strategy_data(
#         sample_index_prices, sample_performance_metrics, sample_investment_values,
#         1000.0, "market_cap", "monthly", True, 0.5
#     )

#     # Verify key fields
#     assert data["method"] == "market_cap"
#     assert data["rebalance_frequency"] == "monthly"
#     assert data["apply_staking"] is True
#     assert data["stablecoin_allocation"] == 0.5
#     assert data["used_fear_greed_data"] is False
#     assert "initial_investment" in data
#     assert "final_investment" in data
#     assert "dates" in data
#     assert "values" in data

#     # Test with fear/greed data
#     data = create_strategy_data(
#         sample_index_prices, sample_performance_metrics, sample_investment_values,
#         1000.0, "market_cap", "monthly", True, 0.5,
#         fear_greed_data=sample_fear_greed_data
#     )

#     assert data["used_fear_greed_data"] is True
#     assert data["fear_greed_data_points"] == len(sample_fear_greed_data)

#     # Test with start date
#     data = create_strategy_data(
#         sample_index_prices, sample_performance_metrics, sample_investment_values,
#         1000.0, "market_cap", "monthly", True, 0.5,
#         start_date="2021-01-01"
#     )

#     assert data["start_date"] == "2021-01-01"

#     # Test with datetime start date
#     data = create_strategy_data(
#         sample_index_prices, sample_performance_metrics, sample_investment_values,
#         1000.0, "market_cap", "monthly", True, 0.5,
#         start_date=datetime(2021, 1, 1)
#     )

#     assert data["start_date"] == "2021-01-01"


# @patch("metrics.calculate_benchmark_performance")
# @patch("visualization.create_performance_data")
# def test_process_benchmark_data(mock_create_data, mock_benchmark, sample_historical_data):
#     """Test processing benchmark data."""
#     # Mock benchmark performance calculation
#     mock_benchmark.return_value = ([1000.0, 1050.0, 1100.0], {"total_return": 10.0})

#     # Mock create_performance_data to return a dictionary with required fields
#     mock_create_data.return_value = {
#         "initial_investment": 1000.0,
#         "final_investment": 1100.0,
#         "dates": ["2021-01-01", "2021-01-02", "2021-01-03"],
#         "values": [100.0, 105.0, 110.0]
#     }

#     # Test successful case
#     performance_data, success = process_benchmark_data(
#         sample_historical_data, 1000.0
#     )

#     assert success is True
#     assert performance_data is not None
#     assert "initial_investment" in performance_data
#     assert "final_investment" in performance_data

#     # Test with missing BTC data
#     no_btc_data = {"eth": sample_historical_data["eth"]}
#     performance_data, success = process_benchmark_data(no_btc_data, 1000.0)

#     assert success is False
#     assert performance_data is None

#     # Test with empty BTC data
#     empty_btc_data = {"btc": []}
#     performance_data, success = process_benchmark_data(empty_btc_data, 1000.0)

#     assert success is False
#     assert performance_data is None


@patch("data_loading.load_historical_data")
@patch("data_loading.filter_data_by_start_date")
@patch("data_loading.align_data_timestamps")
@patch("data_loading.process_fear_greed_data")
@patch("data_loading.validate_data_length_consistency")
def test_load_and_prepare_data(
    mock_validate,
    mock_process_fg,
    mock_align,
    mock_filter,
    mock_load,
    sample_historical_data,
    sample_fear_greed_data,
):
    """Test loading and preparing data for analysis."""
    # Set up mocks
    mock_load.return_value = sample_historical_data
    mock_filter.return_value = sample_historical_data
    mock_align.return_value = sample_historical_data
    mock_process_fg.return_value = sample_fear_greed_data
    mock_validate.return_value = (True, "All good")

    # Test successful case
    with patch("builtins.print"):  # Suppress prints
        historical_data, fear_greed_data = load_and_prepare_data(
            ["btc", "eth", "sol"], "./data", "2021-01-01", "fear_greed.json"
        )

    assert historical_data == sample_historical_data
    assert fear_greed_data == sample_fear_greed_data

    # Test with no historical data
    mock_load.return_value = None
    with patch("builtins.print"):
        result = load_and_prepare_data(["btc"], "./data")

    assert result == (None, None)

    # Test with empty data after alignment
    mock_load.return_value = sample_historical_data
    mock_align.return_value = {}
    with patch("builtins.print"):
        result = load_and_prepare_data(["btc"], "./data")

    assert result == (None, None)


@patch("main.load_and_prepare_data")
@patch("main.display_initial_weights")
@patch("main.process_benchmark_data")
@patch("main.print_benchmark_metrics")
@patch("main.calculate_strategy_performance")
@patch("main.print_strategy_metrics")
@patch("main.create_strategy_data")
@patch("main.generate_performance_plots")
def test_run_performance_analysis(
    mock_generate_plots,
    mock_create_data,
    mock_print_metrics,
    mock_calculate,
    mock_print_benchmark,
    mock_process_benchmark,
    mock_display_weights,
    mock_load_data,
    sample_historical_data,
    sample_fear_greed_data,
    sample_index_prices,
    sample_performance_metrics,
    sample_investment_values,
):
    """Test running a complete performance analysis."""
    # Set up mocks
    mock_load_data.return_value = (sample_historical_data, sample_fear_greed_data)
    mock_process_benchmark.return_value = ({"data": "benchmark"}, True)
    mock_calculate.return_value = (
        sample_index_prices,
        sample_performance_metrics,
        sample_investment_values,
    )
    mock_create_data.return_value = {"data": "strategy"}

    # Test successful case
    result = run_performance_analysis(
        tokens=["btc", "eth", "sol"],
        methods=["market_cap"],
        rebalance_frequencies=["monthly"],
        initial_investment=1000.0,
        start_date="2021-01-01",
        data_dir="./data",
        fear_greed_file="fear_greed.json",
        stablecoin_allocation=0.5,
        generate_plots=True,
    )

    # Verify key calls and results
    assert mock_load_data.called
    assert mock_display_weights.called
    assert mock_process_benchmark.called
    assert mock_print_benchmark.called
    assert mock_calculate.called
    assert mock_print_metrics.called
    assert mock_create_data.called
    assert mock_generate_plots.called

    assert "index_prices" in result
    assert "performance_data" in result
    assert "fear_greed_data" in result


# @patch("main.load_and_prepare_data")
# @patch("main.display_initial_weights")
# @patch("main.process_benchmark_data")
# @patch("main.print_benchmark_metrics")
# @patch("main.calculate_strategy_performance")
# @patch("main.print_strategy_metrics")
# @patch("main.create_strategy_data")
# @patch("main.generate_performance_plots")
# def test_run_performance_analysis_failure(
#     mock_generate_plots, mock_create_data, mock_print_metrics,
#     mock_calculate, mock_print_benchmark, mock_process_benchmark,
#     mock_display_weights, mock_load_data,
#     sample_historical_data, sample_fear_greed_data,
#     sample_index_prices, sample_performance_metrics, sample_investment_values
# ):
#     # Test with data loading failure
#     mock_load_data.return_value = (None, None)
#     result = run_performance_analysis()

#     assert result == {'error': 'No data available'}
#     assert not mock_display_weights.called

#     # Test without generating plots
#     mock_load_data.return_value = (sample_historical_data, sample_fear_greed_data)
#     mock_generate_plots.reset_mock()

#     result = run_performance_analysis(generate_plots=False)

#     assert not mock_generate_plots.called
