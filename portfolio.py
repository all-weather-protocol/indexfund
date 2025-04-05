"""
Portfolio management functions for the indexfund package.

This module provides a comprehensive set of functions for:
1. Portfolio construction and initialization
2. Portfolio rebalancing (both index fund and stablecoin allocation)
3. Performance tracking and historical simulation
4. Staking rewards application
5. Market sentiment based adjustments via Fear & Greed Index

The module implements a contrarian investment strategy that adjusts allocation
between crypto assets and stablecoins based on market sentiment.
"""

from datetime import datetime
import random
import numpy as np

from config import STAKING_CONFIG
from data_loading import extract_current_data
from weighting import calculate_index_weights
from metrics import calculate_portfolio_metrics

# ------------------------------------------------------------------------------
# Portfolio Data Structure
# ------------------------------------------------------------------------------

def create_portfolio_structure(start_timestamp, stablecoin_allocation=0.5):
    """
    Create a new portfolio data structure with clear separation of units.
    
    Args:
        start_timestamp (int): First timestamp in the dataset
        stablecoin_allocation (float): Target percentage of portfolio to allocate to stablecoin
        
    Returns:
        dict: A structured portfolio with separate tracking for quantities, values, and allocations
    """
    return {
        "tokens": {},  # Will contain per-token data
        "stablecoin": {
            "quantity": 0.0,               # Amount of stablecoin
            "usd_value": 0.0,              # Current USD value of stablecoin
            "target_allocation": stablecoin_allocation  # Target % of total portfolio
        },
        "volatile_allocation": 1.0 - stablecoin_allocation,  # % allocated to volatile assets
        "total_usd_value": 0.0,            # Total portfolio value in USD
        "metadata": {
            "last_timestamp": start_timestamp,
            "last_rebalance_date": None,
            "last_allocation_rebalance_date": None
        }
    }

def initialize_portfolio(portfolio, token_weights, initial_usd_value, token_prices):
    """
    Initialize a portfolio with token quantities, USD values, and target weights.
    
    Args:
        portfolio (dict): Portfolio structure to initialize
        token_weights (dict): Weights for each token within the volatile portion
        initial_usd_value (float): Initial portfolio value in USD
        token_prices (dict): Current token prices in USD
        
    Returns:
        dict: Initialized portfolio
    """
    # Calculate allocations
    stablecoin_allocation = portfolio["stablecoin"]["target_allocation"]
    volatile_allocation = portfolio["volatile_allocation"]
    
    # Calculate USD values
    stablecoin_usd = initial_usd_value * stablecoin_allocation
    volatile_usd = initial_usd_value * volatile_allocation
    
    # Initialize stablecoin (quantity equals USD value for stablecoin)
    portfolio["stablecoin"]["quantity"] = stablecoin_usd
    portfolio["stablecoin"]["usd_value"] = stablecoin_usd
    
    # Initialize tokens
    portfolio["tokens"] = {}
    
    for token, weight in token_weights.items():
        if token in token_prices and token_prices[token] > 0:
            token_usd_value = volatile_usd * weight
            token_quantity = token_usd_value / token_prices[token]
            
            portfolio["tokens"][token] = {
                "quantity": token_quantity,           # Amount of tokens
                "usd_value": token_usd_value,         # Current USD value
                "target_weight": weight               # Target weight within volatile portion
            }
    
    # Set total portfolio value
    portfolio["total_usd_value"] = initial_usd_value
    
    # Print summary of initialized portfolio
    print(f"Initialized portfolio with {initial_usd_value:.2f} USD")
    print(f"  Stablecoin: {stablecoin_usd:.2f} USD ({stablecoin_allocation*100:.1f}%)")
    print(f"  Volatile assets: {volatile_usd:.2f} USD ({volatile_allocation*100:.1f}%)")
    
    return portfolio

def update_portfolio_values(portfolio, token_prices):
    """
    Update the USD values of all assets in the portfolio based on current prices.
    Only changes USD values, not quantities.
    
    Args:
        portfolio (dict): Portfolio structure
        token_prices (dict): Current token prices in USD
        
    Returns:
        dict: Updated portfolio with new USD values
    """
    volatile_usd_total = 0.0
    
    # Update token USD values
    for token, data in portfolio["tokens"].items():
        if token in token_prices:
            # Calculate new USD value based on quantity and current price
            data["usd_value"] = data["quantity"] * token_prices[token]
            volatile_usd_total += data["usd_value"]
    
    # Stablecoin value equals its quantity (assuming $1 price)
    stablecoin_usd = portfolio["stablecoin"]["quantity"]
    portfolio["stablecoin"]["usd_value"] = stablecoin_usd
    
    # Update total portfolio value
    portfolio["total_usd_value"] = volatile_usd_total + stablecoin_usd
    
    return portfolio

def calculate_portfolio_total_value(portfolio):
    """
    Calculate the total portfolio value in USD.
    
    Args:
        portfolio (dict): Portfolio structure
        
    Returns:
        float: Total portfolio value in USD
    """
    # Sum token values
    token_value = sum(data["usd_value"] for data in portfolio["tokens"].values())
    
    # Add stablecoin value
    total_value = token_value + portfolio["stablecoin"]["usd_value"]
    
    return total_value

# ------------------------------------------------------------------------------
# Core Portfolio Calculation Functions
# ------------------------------------------------------------------------------

def calculate_historical_index_prices(
    historical_data,
    method,
    rebalance_frequency="none",
    apply_staking=True,
    start_date=None,
    stablecoin_allocation=0.5,  # Default to 50% in stablecoin
    fear_greed_data=None,  # Optional fear and greed index data
):
    """
    Calculate historical index prices using different weighting methods and options.
    Allows for a fixed percentage allocation to stablecoin.

    Args:
        historical_data (dict): Dictionary containing historical price and market cap data
                              Format: {"token": [[timestamp, price, market_cap], ...]}
        method (str): Weighting method ("market_cap", "sqrt_market_cap")
        rebalance_frequency (str): Rebalancing frequency ("none", "monthly", "quarterly", "yearly")
        apply_staking (bool): Whether to apply staking rewards
        start_date (datetime or str): Optional start date for analysis (format: "YYYY-MM-DD")
        stablecoin_allocation (float): Percentage of total portfolio to allocate to stablecoin (0.0-1.0)
        fear_greed_data (list): List of [timestamp, value, value_classification] entries for fear and greed index

    Returns:
        tuple: (price_history, metrics) where:
            - price_history is a list of [timestamp, price] pairs
            - metrics is a dictionary of performance metrics
    """
    # --- Data Preparation ---
    processed_data = _preprocess_historical_data(historical_data, start_date)
    if not processed_data:
        return [], {}

    # Get only the index tokens (exclude stablecoin)
    index_data = {k: v for k, v in processed_data.items() if k != "stablecoin"}

    # Get sorted timestamps for analysis
    timestamps = _extract_sorted_timestamps(processed_data)
    if not timestamps:
        return [], {}

    # Process fear and greed data if provided
    fear_greed_map = _prepare_fear_greed_data(fear_greed_data)

    # --- Portfolio Initialization ---
    portfolio = create_portfolio_structure(timestamps[0], stablecoin_allocation)
    initial_value = 100.0  # Start with $100 for simplicity

    # --- Backtest Simulation ---
    result = []
    is_initialized = False

    # Summary statistics to track
    rebalance_count = 0
    fear_greed_rebalance_count = 0

    for timestamp in timestamps:
        current_date = datetime.fromtimestamp(timestamp / 1000)

        # Get market data at current timestamp
        current_market_caps, current_prices = extract_current_data(
            index_data, timestamp
        )

        # Calculate index fund weights based on market caps
        current_weights = calculate_index_weights(current_market_caps, method)

        # Initialize positions if this is the first timestamp
        if not is_initialized:
            portfolio = initialize_portfolio(
                portfolio,
                current_weights,
                initial_value,
                current_prices
            )
            is_initialized = True
            print(f"Portfolio initialized with {stablecoin_allocation*100:.1f}% stablecoin allocation")

        # Apply staking rewards if enabled (do this before rebalancing)
        if apply_staking:
            apply_staking_to_portfolio(portfolio, timestamp)

        # Get fear and greed data for this timestamp if available
        current_fear_greed = fear_greed_map.get(timestamp) if fear_greed_map else None

        # Periodic rebalancing based on frequency
        if should_rebalance(
            current_date, 
            portfolio["metadata"]["last_rebalance_date"], 
            rebalance_frequency
        ):
            rebalance_portfolio_tokens(
                portfolio,
                current_weights,
                current_prices,
                timestamp
            )
            
            # Update rebalance date
            portfolio["metadata"]["last_rebalance_date"] = current_date
            rebalance_count += 1
            
            # After rebalancing tokens, we might also need to rebalance stablecoin allocation
            # based on fear and greed if available
            if current_fear_greed:
                fear_greed_adjusted = process_fear_greed_rebalancing(
                    portfolio,
                    current_fear_greed,
                    current_prices,
                    timestamp
                )
                if fear_greed_adjusted:
                    fear_greed_rebalance_count += 1

        # Update portfolio values with current prices
        update_portfolio_values(portfolio, current_prices)
        
        # Store result
        result.append([timestamp, portfolio["total_usd_value"]])

        # Update timestamp for next iteration
        portfolio["metadata"]["last_timestamp"] = timestamp

    # Calculate performance metrics
    metrics = calculate_portfolio_metrics(result)
    
    # Add additional information to metrics
    metrics["stablecoin_allocation"] = stablecoin_allocation
    metrics["rebalance_frequency"] = rebalance_frequency
    metrics["rebalance_count"] = rebalance_count
    metrics["fear_greed_rebalance_count"] = fear_greed_rebalance_count
    
    # Final portfolio composition
    stablecoin_pct = portfolio["stablecoin"]["usd_value"] / portfolio["total_usd_value"] * 100
    volatile_pct = 100 - stablecoin_pct
    
    metrics["final_stablecoin_pct"] = stablecoin_pct
    metrics["final_volatile_pct"] = volatile_pct
    
    # Token weights within volatile portion
    if portfolio["total_usd_value"] > 0:
        token_values = {}
        for token, data in portfolio["tokens"].items():
            token_pct = (data["usd_value"] / portfolio["total_usd_value"]) * 100
            token_values[token] = {
                "usd_value": data["usd_value"],
                "percentage": token_pct
            }
        metrics["token_values"] = token_values

    # Print summary statistics
    print(f"Simulation complete: {method} with {rebalance_frequency} rebalancing")
    print(f"  Initial allocation: {stablecoin_allocation*100:.1f}% stablecoin, {(1-stablecoin_allocation)*100:.1f}% crypto")
    print(f"  Final allocation: {stablecoin_pct:.1f}% stablecoin, {volatile_pct:.1f}% crypto")
    print(f"  Initial value: ${initial_value:.2f}")
    print(f"  Final value: ${portfolio['total_usd_value']:.2f}")
    print(f"  Return: {metrics['total_return']:.2f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"  Rebalances performed: {rebalance_count}")
    if fear_greed_map:
        print(f"  Fear & Greed adjustments: {fear_greed_rebalance_count}")

    return result, metrics


# ------------------------------------------------------------------------------
# Staking and Rewards Functions
# ------------------------------------------------------------------------------

def calculate_staking_rewards(amount, apr, days):
    """
    Calculate staking rewards with daily compounding.

    Args:
        amount (float): Initial token amount
        apr (float): Annual percentage rate (as decimal)
        days (float): Number of days to calculate rewards for

    Returns:
        float: Rewards earned over the specified period
    """
    daily_rate = apr / 365
    return amount * ((1 + daily_rate) ** days - 1)


def apply_staking_to_portfolio(portfolio, current_timestamp):
    """
    Apply staking rewards to all assets in the portfolio.
    Updates token quantities but not USD values.

    Args:
        portfolio (dict): Portfolio data structure
        current_timestamp (int): Current timestamp in milliseconds
    """
    # Calculate days since last update
    days = (current_timestamp - portfolio["metadata"]["last_timestamp"]) / (1000 * 60 * 60 * 24)

    if days <= 0:
        return

    # Apply staking to each token
    for token, data in portfolio["tokens"].items():
        if token in STAKING_CONFIG and STAKING_CONFIG[token] > 0:
            reward_quantity = calculate_staking_rewards(
                data["quantity"], STAKING_CONFIG[token], days
            )
            # Add rewards to quantity (auto-compound)
            data["quantity"] += reward_quantity

    # Apply staking to stablecoin if configured
    if "stablecoin" in STAKING_CONFIG and STAKING_CONFIG["stablecoin"] > 0:
        stablecoin_reward = calculate_staking_rewards(
            portfolio["stablecoin"]["quantity"], 
            STAKING_CONFIG["stablecoin"], 
            days
        )
        portfolio["stablecoin"]["quantity"] += stablecoin_reward


# ------------------------------------------------------------------------------
# Rebalancing Logic
# ------------------------------------------------------------------------------

def should_rebalance(current_date, last_rebalance_date, frequency):
    """
    Determine if rebalancing should occur based on the frequency.

    Args:
        current_date (datetime): Current date
        last_rebalance_date (datetime): Last rebalance date
        frequency (str): Rebalancing frequency ('none', 'monthly', 'quarterly', 'yearly')

    Returns:
        bool: True if rebalancing should occur, False otherwise
    """
    if frequency == "none":
        return False

    if last_rebalance_date is None:
        return True

    if frequency == "monthly":
        return current_date.year > last_rebalance_date.year or (
            current_date.year == last_rebalance_date.year
            and current_date.month > last_rebalance_date.month
        )
    elif frequency == "quarterly":
        current_quarter = (current_date.month - 1) // 3 + 1
        last_quarter = (last_rebalance_date.month - 1) // 3 + 1
        return current_date.year > last_rebalance_date.year or (
            current_date.year == last_rebalance_date.year
            and current_quarter > last_quarter
        )
    elif frequency == "yearly":
        return current_date.year > last_rebalance_date.year

    return False


def rebalance_portfolio_tokens(portfolio, target_weights, token_prices, timestamp):
    """
    Rebalance the token portion of the portfolio to match target weights.
    Updates token quantities but not USD values.

    Args:
        portfolio (dict): Portfolio data structure
        target_weights (dict): Target weights for each token
        token_prices (dict): Current token prices
    """
    # Calculate the current USD value of all volatile assets
    current_volatile_value = sum(
        data["quantity"] * token_prices.get(token, 0)
        for token, data in portfolio["tokens"].items()
        if token in token_prices
    )
    
    # Log rebalancing action
    print(f"Rebalancing portfolio: volatile assets worth ${current_volatile_value:.2f} at {datetime.fromtimestamp(timestamp / 1000).date()}")
    
    # Update each token's target weight in the portfolio
    for token, weight in target_weights.items():
        if token in portfolio["tokens"]:
            portfolio["tokens"][token]["target_weight"] = weight
    
    # Calculate new quantities based on target weights
    for token, data in portfolio["tokens"].items():
        if token in token_prices and token_prices[token] > 0:
            # Calculate target USD value based on weight
            target_usd = current_volatile_value * data["target_weight"]
            
            # Calculate new quantity needed
            new_quantity = target_usd / token_prices[token]
            
            # Update quantity
            data["quantity"] = new_quantity
    
    return portfolio


def rebalance_stablecoin_allocation(portfolio, new_allocation, token_prices):
    """
    Rebalance the allocation between stablecoin and volatile assets.
    Updates quantities but not USD values.

    Args:
        portfolio (dict): Portfolio data structure
        new_allocation (float): New target stablecoin allocation (0.0-1.0)
        token_prices (dict): Current token prices
    """
    # Calculate current values
    volatile_value = sum(
        data["quantity"] * token_prices.get(token, 0)
        for token, data in portfolio["tokens"].items()
        if token in token_prices
    )
    stablecoin_value = portfolio["stablecoin"]["quantity"]
    total_value = volatile_value + stablecoin_value
    
    # Calculate target values based on new allocation
    target_stablecoin_value = total_value * new_allocation
    target_volatile_value = total_value * (1 - new_allocation)
    
    # Adjust stablecoin quantity
    stablecoin_adjustment = target_stablecoin_value - stablecoin_value
    portfolio["stablecoin"]["quantity"] = target_stablecoin_value
    portfolio["stablecoin"]["target_allocation"] = new_allocation
    portfolio["volatile_allocation"] = 1.0 - new_allocation
    
    # If volatile value is zero, we can't adjust token quantities proportionally
    if volatile_value <= 0:
        return portfolio
    
    # Scale all token quantities to match target volatile value
    scaling_factor = target_volatile_value / volatile_value
    
    for token_data in portfolio["tokens"].values():
        token_data["quantity"] *= scaling_factor
    
    # Log the rebalancing action
    action = "Increased" if stablecoin_adjustment > 0 else "Decreased"
    print(f"{action} stablecoin allocation to {new_allocation:.2f} " +
          f"(adjusted by ${abs(stablecoin_adjustment):.2f})")
    
    return portfolio


# ------------------------------------------------------------------------------
# Sentiment-Based Allocation Adjustment (Fear & Greed)
# ------------------------------------------------------------------------------

def process_fear_greed_rebalancing(portfolio, fear_greed_data, token_prices, timestamp):
    """
    Process rebalancing based on fear and greed index data.
    Uses a contrarian approach - more crypto in fear, more stablecoin in greed.

    Args:
        portfolio (dict): Portfolio data structure 
        fear_greed_data (dict): Fear and greed data for the current timestamp
        token_prices (dict): Current token prices
        timestamp (int): Current timestamp for logging
        
    Returns:
        bool: True if rebalancing occurred, False otherwise
    """
    if not fear_greed_data:
        return False
    
    # Define allocation limits and adjustment size
    STABLECOIN_MIN_ALLOCATION = 0.01
    STABLECOIN_MAX_ALLOCATION = 0.99
    ADJUSTMENT_SIZE = 0.1
    
    # Get current stablecoin allocation
    base_allocation = portfolio["stablecoin"]["target_allocation"]
    
    # Default to no change
    new_allocation = base_allocation
    
    # Get sentiment classification
    classification = fear_greed_data["classification"]
    
    # Apply contrarian strategy based on market sentiment
    reason = None
    
    if classification == "Extreme Fear":
        # During extreme fear: reduce stablecoin (buy more crypto)
        new_allocation = max(base_allocation - ADJUSTMENT_SIZE, STABLECOIN_MIN_ALLOCATION)
        reason = "Extreme Fear (buying opportunity)"
    elif classification == "Extreme Greed":
        # During extreme greed: increase stablecoin (take profits)
        new_allocation = min(base_allocation + ADJUSTMENT_SIZE, STABLECOIN_MAX_ALLOCATION)
        reason = "Extreme Greed (taking profits)"
        
    # If no change needed, return early
    if new_allocation == base_allocation:
        return False
        
    # Log the allocation change
    date_str = datetime.fromtimestamp(timestamp / 1000).date()
    action = "Increased" if new_allocation > base_allocation else "Decreased"
    print(f"Contrarian strategy: {action} stablecoin to {new_allocation:.2f} due to {reason} at {date_str}")
    
    # Apply the new allocation
    rebalance_stablecoin_allocation(portfolio, new_allocation, token_prices)
    
    return True


def _prepare_fear_greed_data(fear_greed_data):
    """
    Transform fear and greed data into a lookup map.

    Args:
        fear_greed_data (list or None): List of [timestamp, value, classification] entries

    Returns:
        dict or None: Mapping of timestamps to fear/greed data
    """
    if not fear_greed_data:
        return None

    return {
        entry[0]: {"value": entry[1], "classification": entry[2]}
        for entry in fear_greed_data
    }


# ------------------------------------------------------------------------------
# Data Processing and Validation
# ------------------------------------------------------------------------------


def _preprocess_historical_data(historical_data, start_date):
    """
    Preprocess and filter historical data based on start date.

    Args:
        historical_data (dict): Dictionary with historical token data
        start_date (datetime or str): Optional start date for filtering

    Returns:
        dict: Processed historical data
    """
    if not historical_data:
        return {}

    if start_date:
        # Convert string date to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        # Convert to timestamp in milliseconds
        start_timestamp = int(start_date.timestamp() * 1000)

        # Filter data
        return filter_data_by_start_date(historical_data, start_timestamp)

    return historical_data


def _extract_sorted_timestamps(historical_data):
    """
    Extract and sort all unique timestamps from historical data.

    Args:
        historical_data (dict): Dictionary with historical token data

    Returns:
        list: Sorted list of unique timestamps
    """
    all_timestamps = set()
    for token_data in historical_data.values():
        all_timestamps.update(timestamp for timestamp, _, _ in token_data)
    return sorted(list(all_timestamps))


def filter_data_by_start_date(historical_data, start_timestamp):
    """
    Filter historical data to only include data points after the start timestamp.

    Args:
        historical_data (dict): Dictionary containing historical price and market cap data
        start_timestamp (int): Start timestamp in milliseconds

    Returns:
        dict: Filtered historical data
    """
    filtered_data = {}

    for token, data in historical_data.items():
        # Filter data points that are >= start_timestamp
        filtered_token_data = [entry for entry in data if entry[0] >= start_timestamp]

        # Only include tokens that have data after the start date
        if filtered_token_data:
            filtered_data[token] = filtered_token_data

    # Validate that all filtered data has the same length
    is_valid, message = validate_data_length_consistency(filtered_data)
    if not is_valid:
        print(f"Warning: {message}")

    return filtered_data


def validate_data_length_consistency(historical_data):
    """
    Validate that all tokens in the historical data have the same number of data points.

    Args:
        historical_data (dict): Dictionary mapping token symbols to their historical data

    Returns:
        tuple: (is_valid, validation_message)
            - is_valid (bool): True if all token data has the same length, False otherwise
            - validation_message (str): Description of validation results
    """
    if not historical_data or len(historical_data) <= 1:
        return True, "Only one token present or no data available"

    lengths = {token: len(data) for token, data in historical_data.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) == 1:
        # All tokens have the same length
        return (
            True,
            f"All tokens have the same length: {next(iter(unique_lengths))} data points",
        )
    else:
        # Different lengths detected
        length_details = ", ".join(
            [f"{token}: {length}" for token, length in lengths.items()]
        )
        return False, f"Tokens have different data lengths: {length_details}"

