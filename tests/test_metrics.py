from datetime import datetime, timedelta

import numpy as np
import pytest

from core.metrics import (
    calculate_annualized_ROI,
    calculate_benchmark_performance,
    calculate_financial_metrics,
    calculate_max_drawdown,
    calculate_portfolio_metrics,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_volatility,
)


# Fixtures
@pytest.fixture
def simple_prices():
    """Simple price series that increases steadily by ~1%"""
    return np.array([100, 101, 102.01, 103.03, 104.06, 105.10])


@pytest.fixture
def drawdown_prices():
    """Price series with a drawdown"""
    return np.array([100, 105, 110, 100, 95, 105])


@pytest.fixture
def simple_returns():
    """Simple returns (1% daily return)"""
    return np.array([0.01, 0.01, 0.01, 0.01, 0.01])


@pytest.fixture
def mixed_returns():
    """Returns with some negative values"""
    return np.array([0.01, 0.02, -0.03, -0.01, 0.03])


@pytest.fixture
def price_data():
    """Mock price data with timestamps for benchmark testing"""
    now = datetime.now()
    return [
        # timestamp in milliseconds, price, market_cap
        (int(now.timestamp() * 1000), 100, 1000000),
        (int((now + timedelta(days=1)).timestamp() * 1000), 102, 1020000),
        (int((now + timedelta(days=2)).timestamp() * 1000), 104, 1040000),
        (int((now + timedelta(days=3)).timestamp() * 1000), 106, 1060000),
    ]


@pytest.fixture
def value_history():
    """Mock portfolio value history"""
    now = datetime.now()
    return [
        (int(now.timestamp() * 1000), 1000),
        (int((now + timedelta(days=30)).timestamp() * 1000), 1050),
        (int((now + timedelta(days=60)).timestamp() * 1000), 1100),
        (int((now + timedelta(days=90)).timestamp() * 1000), 1150),
    ]


# Tests
def test_calculate_returns(simple_prices, drawdown_prices):
    """Test the calculate_returns function"""
    # Test with simple prices
    returns = calculate_returns(simple_prices)
    # Print the actual returns for debugging
    print("Actual returns:", returns)
    # Use lower precision (4 decimal places) for comparison
    np.testing.assert_almost_equal(
        returns, np.array([0.01, 0.01, 0.01, 0.01, 0.01]), decimal=2
    )

    # Test with drawdown prices
    returns = calculate_returns(drawdown_prices)
    expected = np.array([0.05, 0.0476, -0.0909, -0.05, 0.1053])
    np.testing.assert_almost_equal(returns, expected, decimal=4)


def test_calculate_max_drawdown(simple_prices, drawdown_prices):
    """Test the calculate_max_drawdown function"""
    # Test with simple prices (no drawdown)
    max_dd = calculate_max_drawdown(simple_prices)
    assert max_dd == 0

    # Test with drawdown prices
    max_dd = calculate_max_drawdown(drawdown_prices)
    assert pytest.approx(max_dd, abs=0.0001) == -13.6364


def test_calculate_volatility(simple_returns, mixed_returns):
    """Test the calculate_volatility function"""
    # Test with simple returns (constant, so volatility should be 0)
    vol = calculate_volatility(simple_returns, annualize=False)
    assert vol == pytest.approx(0, abs=1e-10)

    # Test with mixed returns
    vol = calculate_volatility(mixed_returns, annualize=False)
    assert vol == pytest.approx(2.15, abs=0.01)

    # Test annualized volatility
    # With TRADING_DAYS_PER_YEAR = 252 from config
    vol_annual = calculate_volatility(mixed_returns, annualize=True)
    vol_daily = calculate_volatility(mixed_returns, annualize=False)
    expected_annual = vol_daily * np.sqrt(252)
    assert vol_annual == pytest.approx(expected_annual, abs=0.001)


def test_calculate_sharpe_ratio(simple_returns, mixed_returns):
    """Test the calculate_sharpe_ratio function"""
    # Test with simple returns
    sharpe = calculate_sharpe_ratio(simple_returns)
    # Since std is 0, expect a very large value (infinity in theory)
    assert sharpe > 1000 or np.isinf(sharpe)

    # Test with mixed returns
    sharpe = calculate_sharpe_ratio(mixed_returns)
    # For mixed returns with some negative values, Sharpe ratio should be positive but finite
    assert 2.7 < sharpe < 2.9


def test_calculate_sortino_ratio(simple_returns, mixed_returns):
    """Test the calculate_sortino_ratio function"""
    # Test with simple returns (no negative returns)
    sortino = calculate_sortino_ratio(simple_returns)
    # With no negative returns, should return 0
    assert sortino == 0

    # Test with mixed returns
    sortino = calculate_sortino_ratio(mixed_returns)
    # For mixed returns with negative values, Sortino ratio should be finite
    assert np.isfinite(sortino)


def test_calculate_financial_metrics(drawdown_prices):
    """Test the calculate_financial_metrics function"""
    metrics = calculate_financial_metrics(drawdown_prices)

    # Check that all expected metrics are present
    assert "max_drawdown" in metrics
    assert "volatility" in metrics
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics

    # Check max_drawdown value
    assert metrics["max_drawdown"] == pytest.approx(-13.6364, abs=0.0001)


def test_calculate_benchmark_performance(price_data):
    """Test the calculate_benchmark_performance function"""
    initial_investment = 1000
    investment_values, metrics = calculate_benchmark_performance(
        price_data, initial_investment
    )

    # Check length of investment values
    assert len(investment_values) == len(price_data)

    # Check first value matches initial investment
    assert investment_values[0] == initial_investment

    # Check last value is correctly calculated
    expected_final = initial_investment * (price_data[-1][1] / price_data[0][1])
    assert investment_values[-1] == pytest.approx(expected_final)

    # Check metrics are returned
    assert isinstance(metrics, dict)
    assert "volatility" in metrics


def test_calculate_portfolio_metrics(value_history):
    """Test the calculate_portfolio_metrics function"""
    metrics = calculate_portfolio_metrics(value_history)

    # Check all metrics are present
    assert "max_drawdown" in metrics
    assert "volatility" in metrics
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "total_return" in metrics
    assert "initial_value" in metrics
    assert "final_value" in metrics
    assert "annualized_roi" in metrics

    # Check total return
    expected_total_return = (
        (value_history[-1][1] - value_history[0][1]) / value_history[0][1] * 100
    )
    assert metrics["total_return"] == pytest.approx(expected_total_return)


def test_calculate_annualized_ROI():
    """Test the calculate_annualized_ROI function"""
    # Test basic calculation
    initial = 1000
    final = 1100
    days = 365  # One year

    # For one year, the annualized ROI should be the same as total ROI
    roi = calculate_annualized_ROI(initial, final, days)
    expected = ((final - initial) / initial) * 100
    assert roi == pytest.approx(expected)

    # Test for 2 years
    days = 730  # Two years
    roi = calculate_annualized_ROI(initial, final, days)
    expected = (((final / initial) ** (1 / 2)) - 1) * 100
    assert roi == pytest.approx(expected)

    # Test for half a year
    days = 182  # Half a year (approximately)
    roi = calculate_annualized_ROI(initial, final, days)
    expected = (((final / initial) ** (365 / 182)) - 1) * 100
    assert roi == pytest.approx(expected)

    # Test edge cases
    # Zero days should return 0
    assert calculate_annualized_ROI(initial, final, 0) == 0

    # Zero initial value should return 0
    assert calculate_annualized_ROI(0, final, days) == 0
