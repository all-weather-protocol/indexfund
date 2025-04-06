"""
Strategy module for the indexfund package.
Contains functions for creating, formatting, and calculating strategy performance.
"""

from datetime import datetime

# Import from metrics module
from metrics import calculate_financial_metrics

# Import from portfolio module
from portfolio import calculate_historical_index_prices

# Import from visualization module
from visualization import create_performance_data


def format_strategy_name(
    method, freq, apply_staking, stablecoin_allocation, start_date=None
):
    """
    Format a standardized strategy name based on parameters.

    Args:
        method (str): The weighting method
        freq (str): Rebalancing frequency
        apply_staking (bool): Whether staking is applied
        stablecoin_allocation (float): The stablecoin allocation (0.0-1.0)
        start_date (str or datetime, optional): The start date

    Returns:
        str: A formatted strategy name
    """
    # Create strategy name with stablecoin allocation info and staking info
    strategy_name = f"{method.replace('_', ' ').title()} ({freq.title()})"

    # Add staking status to strategy name
    strategy_name = f"{strategy_name} {'with' if apply_staking else 'without'} Staking"

    # Add stablecoin allocation info if applicable
    if stablecoin_allocation > 0:
        stable_pct = int(stablecoin_allocation * 100)
        index_pct = 100 - stable_pct
        strategy_name = f"{index_pct}% {strategy_name}, {stable_pct}% Stablecoin"

    if start_date:
        if isinstance(start_date, str):
            start_date_str = start_date
        else:
            start_date_str = start_date.strftime("%Y-%m-%d")
        strategy_name = f"{strategy_name} - Start: {start_date_str}"

    return strategy_name


def generate_strategy_key(method, freq, apply_staking, use_fear_greed, start_date=None):
    """
    Generate a standardized key for a strategy.

    Args:
        method (str): The weighting method
        freq (str): Rebalancing frequency
        apply_staking (bool): Whether staking is applied
        use_fear_greed (bool): Whether fear/greed index was used
        start_date (str or datetime, optional): The start date

    Returns:
        str: A key string representing the strategy
    """
    key = f"{method}_{freq}_staking_{apply_staking}_feargreed_{use_fear_greed}"

    if start_date:
        if isinstance(start_date, str):
            start_date_str = start_date
        else:
            start_date_str = start_date.strftime("%Y-%m-%d")
        key = f"{key}_{start_date_str.replace('-', '')}"

    return key


def calculate_strategy_performance(
    historical_data,
    method,
    freq,
    initial_investment,
    start_date=None,
    stablecoin_allocation=0.5,
    fear_greed_data=None,
    apply_staking=True,
):
    """
    Calculate performance metrics for an index strategy.

    Args:
        historical_data (dict): Dictionary of historical price data
        method (str): Weighting method
        freq (str): Rebalancing frequency
        initial_investment (float): Initial investment amount
        start_date (str or datetime): Optional start date for analysis (format: "YYYY-MM-DD")
        stablecoin_allocation (float): Percentage of portfolio to allocate to stablecoin (0.0-1.0)
        fear_greed_data (list): List of [timestamp, value, value_classification] entries for fear and greed index
        apply_staking (bool): Whether to apply staking rewards in the simulation

    Returns:
        tuple: (index_prices, metrics, investment_value)
    """
    # Calculate historical index prices
    index_prices, metrics = calculate_historical_index_prices(
        historical_data,
        method,
        rebalance_frequency=freq,
        apply_staking=apply_staking,
        start_date=start_date,
        stablecoin_allocation=stablecoin_allocation,
        fear_greed_data=fear_greed_data,
    )

    # If no data available (possibly due to start date filtering), return early
    if not index_prices:
        return [], None, []

    # Extract prices
    prices = [price for _, price in index_prices]

    # Calculate investment value over time (normalized to initial investment)
    # Fixed calculation that preserves stablecoin allocation effects
    # When using stablecoin allocation, we're already getting the correct absolute prices
    # So we just need to scale to the initial investment amount
    investment_value = [price / prices[0] * initial_investment for price in prices]

    # Note: The raw price series from calculate_historical_index_prices already incorporates
    # the stablecoin allocation effects. The portfolio starts with 100.0, so we just scale
    # by initial_investment/100.0

    # Calculate performance metrics
    performance_metrics = calculate_financial_metrics(prices)

    return index_prices, performance_metrics, investment_value


def create_strategy_data(
    index_prices,
    metrics,
    investment_value,
    initial_investment,
    method,
    freq,
    apply_staking,
    stablecoin_allocation,
    fear_greed_data=None,
    start_date=None,
):
    """
    Create performance data for a strategy.

    Args:
        index_prices (list): List of [timestamp, value] pairs
        metrics (dict): Performance metrics dictionary
        investment_value (list): List of investment values over time
        initial_investment (float): Initial investment amount
        method (str): Weighting method
        freq (str): Rebalancing frequency
        apply_staking (bool): Whether staking is applied
        stablecoin_allocation (float): Stablecoin allocation (0.0-1.0)
        fear_greed_data (list, optional): Fear and greed data
        start_date (str or datetime, optional): Start date for analysis

    Returns:
        dict: Performance data dictionary
    """
    # Convert timestamps to datetime objects
    dates = [datetime.fromtimestamp(ts / 1000) for ts, _ in index_prices]
    prices = [price for _, price in index_prices]

    # Create performance data dictionary
    performance_data = create_performance_data(
        dates, prices, investment_value, initial_investment, metrics, freq
    )

    # Add strategy metadata
    performance_data["method"] = method
    performance_data["rebalance_frequency"] = freq
    performance_data["apply_staking"] = apply_staking
    performance_data["stablecoin_allocation"] = stablecoin_allocation

    # Add start date to performance data
    if start_date:
        if isinstance(start_date, str):
            performance_data["start_date"] = start_date
        else:
            performance_data["start_date"] = start_date.strftime("%Y-%m-%d")

    # Add fear and greed data info to performance data if available
    if fear_greed_data:
        performance_data["used_fear_greed_data"] = True
        performance_data["fear_greed_data_points"] = len(fear_greed_data)
    else:
        performance_data["used_fear_greed_data"] = False

    return performance_data


def process_benchmark_data(historical_data, initial_investment, start_date=None):
    """
    Process BTC benchmark data for comparison.

    Args:
        historical_data (dict): Historical data dictionary
        initial_investment (float): Initial investment amount
        start_date (str or datetime, optional): Start date

    Returns:
        tuple: (performance_data, success_flag)
    """
    if "btc" not in historical_data:
        return None, False

    btc_prices = historical_data["btc"]
    if not btc_prices:
        return None, False

    # Import here to avoid circular imports
    from metrics import calculate_benchmark_performance

    btc_investment, btc_metrics = calculate_benchmark_performance(
        btc_prices, initial_investment
    )

    # Create BTC benchmark performance data for plotting
    btc_dates = [datetime.fromtimestamp(ts / 1000) for ts, _, _ in btc_prices]
    btc_values = [price for _, price, _ in btc_prices]

    btc_performance_data = create_performance_data(
        btc_dates,
        btc_values,
        btc_investment,
        initial_investment,
        btc_metrics,
        "N/A",
    )
    # Add start date to performance data
    if start_date:
        if isinstance(start_date, str):
            btc_performance_data["start_date"] = start_date
        else:
            btc_performance_data["start_date"] = start_date.strftime("%Y-%m-%d")

    return btc_performance_data, True
