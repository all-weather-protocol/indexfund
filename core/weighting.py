"""
Weight calculation functions for the indexfund package.
Contains implementations of different weighting strategies for creating crypto indexes.
"""

import math


def calculate_weight_market_cap(market_cap):
    """Calculate weights based on direct market cap"""
    return {asset: mc for asset, mc in market_cap.items()}


def calculate_weight_sqrt_market_cap(market_cap):
    """Calculate weights based on square root of market cap"""
    return {asset: math.sqrt(mc) for asset, mc in market_cap.items()}


def normalize_weights(weights):
    """Normalize weights to sum to 1.0"""
    total = sum(weights.values())
    return {asset: w / total for asset, w in weights.items()}


def calculate_index_weights(market_cap, method):
    """
    Calculate the weighted index allocation using different weighting methods.

    Args:
        market_cap (dict): Dictionary of asset market caps (in USD)
        method (str): Weighting method ("market_cap", "sqrt_market_cap")

    Returns:
        dict: Normalized weight distribution as percentages
    """
    # Calculate weights based on method
    if method == "market_cap":
        weights = calculate_weight_market_cap(market_cap)
    elif method == "sqrt_market_cap":
        weights = calculate_weight_sqrt_market_cap(market_cap)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Normalize weights
    return normalize_weights(weights)


def print_portfolio_weights(weights):
    """
    Print current portfolio weights in a formatted way.

    Args:
        weights (dict): Dictionary mapping token symbols to their weights
    """
    print("\n=== Portfolio Weights ===")

    # Format weights as percentages and sort by weight descending
    formatted_weights = {token: weight * 100 for token, weight in weights.items()}
    sorted_weights = sorted(formatted_weights.items(), key=lambda x: x[1], reverse=True)

    # Print each token's weight
    for token, weight in sorted_weights:
        print(f"{token.upper()}: {weight:.2f}%")
    print("")  # Empty line for separation


def display_initial_weights(historical_data, methods):
    """
    Calculate and display the initial portfolio weights for each method.

    Args:
        historical_data (dict): Dictionary of historical price data
        methods (list): List of weighting methods to analyze
    """
    # Find the earliest timestamp in the data
    timestamps = []
    for token_data in historical_data.values():
        timestamps.extend(ts for ts, _, _ in token_data)

    if not timestamps:
        print("No data available to calculate weights")
        return

    first_timestamp = min(timestamps)

    # Extract market caps at the earliest timestamp
    market_caps = {}
    for token, data in historical_data.items():
        for ts, _, mcap in data:
            if ts == first_timestamp:
                market_caps[token] = mcap
                break

    # Calculate and display weights for each method
    for method in methods:
        print(f"\nInitial weights for {method.replace('_', ' ').title()}:")
        weights = calculate_index_weights(market_caps, method)
        print_portfolio_weights(weights)
