"""
Performance metrics calculation functions for the indexfund package.
Contains functions for calculating financial metrics like returns, volatility, etc.
"""

import numpy as np

from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def calculate_returns(prices):
    """
    Calculate daily returns from a series of prices.

    Args:
        prices (list): Time series of prices

    Returns:
        numpy.ndarray: Daily returns as percentage changes
    """
    return np.diff(prices) / prices[:-1]


def calculate_max_drawdown(prices):
    """
    Calculate the maximum drawdown from a series of prices.

    Args:
        prices (list): Time series of prices

    Returns:
        float: Maximum drawdown as a percentage (negative value)
    """
    prices = np.array(prices)
    rolling_max = np.maximum.accumulate(prices)
    drawdowns = (prices - rolling_max) / rolling_max
    return np.min(drawdowns) * 100  # Convert to percentage


def calculate_volatility(returns, annualize=True):
    """
    Calculate the volatility (standard deviation) of returns.

    Args:
        returns (numpy.ndarray): Array of returns
        annualize (bool): Whether to annualize the volatility

    Returns:
        float: Volatility as a percentage
    """
    daily_volatility = np.std(returns)
    if annualize:
        return (
            daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        )  # Annualized and as percentage
    return daily_volatility * 100  # As percentage


def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """
    Calculate the Sharpe ratio.

    Args:
        returns (numpy.ndarray): Array of returns
        risk_free_rate (float): Annual risk-free rate

    Returns:
        float: Sharpe ratio
    """
    daily_rf_rate = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess_returns = returns - daily_rf_rate
    return np.sqrt(TRADING_DAYS_PER_YEAR) * np.mean(excess_returns) / np.std(returns)


def calculate_sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """
    Calculate the Sortino ratio.

    Args:
        returns (numpy.ndarray): Array of returns
        risk_free_rate (float): Annual risk-free rate

    Returns:
        float: Sortino ratio
    """
    daily_rf_rate = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess_returns = returns - daily_rf_rate

    # Calculate downside deviation (standard deviation of negative returns only)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        return (
            np.sqrt(TRADING_DAYS_PER_YEAR)
            * np.mean(excess_returns)
            / np.std(downside_returns)
        )
    return 0  # No negative returns


def calculate_financial_metrics(prices, risk_free_rate=RISK_FREE_RATE):
    """
    Calculate various financial metrics for a price series.

    Args:
        prices (list): List of prices over time
        risk_free_rate (float): Annual risk-free rate (default 5%)

    Returns:
        dict: Dictionary containing various financial metrics
    """
    # Convert prices to numpy array for easier calculations
    prices = np.array(prices)

    # Calculate daily returns
    returns = calculate_returns(prices)

    # Calculate metrics
    max_drawdown = calculate_max_drawdown(prices)
    volatility = calculate_volatility(returns)
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate)

    return {
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
    }


def calculate_benchmark_performance(price_data, initial_investment):
    """
    Calculate performance metrics for a benchmark asset.

    Args:
        price_data (list): List of [timestamp, price, market_cap] entries
        initial_investment (float): Initial investment amount

    Returns:
        tuple: (investment_values, metrics_dict)
    """
    prices = [price for _, price, _ in price_data]
    investment_values = [initial_investment * (price / prices[0]) for price in prices]

    metrics = calculate_financial_metrics(prices)

    return investment_values, metrics


def calculate_portfolio_metrics(value_history, risk_free_rate=0.05):
    """
    Calculate various financial metrics for a portfolio's value history.

    Args:
        value_history (list): List of portfolio values over time
        risk_free_rate (float): Annual risk-free rate

    Returns:
        dict: Dictionary of financial metrics
    """
    # Convert to numpy array for calculations
    values = np.array([value for _, value in value_history])

    # Calculate daily returns
    returns = np.diff(values) / values[:-1]

    # Calculate max drawdown
    max_drawdown = calculate_max_drawdown(values)

    # Calculate volatility (annualized)
    daily_volatility = np.std(returns)
    annual_volatility = daily_volatility * np.sqrt(252) * 100  # As percentage

    # Calculate Sharpe ratio
    daily_rf_rate = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = returns - daily_rf_rate
    sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

    # Calculate Sortino ratio (using only negative returns)
    downside_returns = returns[returns < 0]
    sortino_ratio = (
        np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)
        if len(downside_returns) > 0
        else 0
    )

    # Calculate total return
    initial_value = values[0]
    final_value = values[-1]
    total_return = ((final_value - initial_value) / initial_value) * 100
    annualized_roi = calculate_annualized_ROI(initial_value, final_value, len(values))
    return {
        "max_drawdown": max_drawdown,
        "volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "total_return": total_return,
        "initial_value": initial_value,
        "final_value": final_value,
        "annualized_roi": annualized_roi,
    }


def calculate_annualized_ROI(initial_value, final_value, days):
    """
    Calculate Annual Percentage Rate (yearly ROI) based on initial value, final value and number of days.

    Args:
        initial_value (float): Initial value of the investment
        final_value (float): Final value of the investment
        days (int): Number of days in the investment period

    Returns:
        float: APR as a percentage
    """
    if days <= 0 or initial_value <= 0:
        return 0

    # Calculate total return
    total_return = (final_value - initial_value) / initial_value

    # Convert to annual rate: (1 + total_return)^(365/days) - 1
    years = days / 365.0
    apr = ((1 + total_return) ** (1 / years) - 1) * 100

    return apr
