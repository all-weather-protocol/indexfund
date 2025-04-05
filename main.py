"""
Main module for the indexfund package.
Contains the main analysis function and command-line interface.
"""

import argparse
from datetime import datetime

# Import from config module
from config import (
    DEFAULT_INITIAL_INVESTMENT,
    DEFAULT_METHODS,
    DEFAULT_REBALANCE_FREQUENCIES,
    DEFAULT_TOKENS,
)

# Import from data_loading module
from data_loading import (
    align_data_timestamps,
    filter_data_by_start_date,
    load_historical_data,
    process_fear_greed_data,
)

# Import from metrics module
# Import from metrics module
from metrics import calculate_benchmark_performance, calculate_financial_metrics

# Import from portfolio module
from portfolio import calculate_historical_index_prices

# Import from utils module
# Import from utils module
from utils import validate_data_length_consistency

# Import from visualization module
from visualization import (
    create_performance_data,
    plot_detailed_performance,
    plot_metrics_only,
    plot_performance_only,
    print_performance_metrics,
)

# Import from weighting module
from weighting import display_initial_weights


def calculate_and_save_strategy_performance(
    historical_data,
    method,
    freq,
    initial_investment,
    start_date=None,
    stablecoin_allocation=0.5,
    fear_greed_data=None,
    apply_staking=True,
):
    print("stablecoin_allocation", stablecoin_allocation)
    """
    Calculate, display, and save performance metrics for an index strategy.

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
        tuple: (index_prices, performance_data)
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
        print(f"No data available for {method} ({freq}) with start date {start_date}")
        return [], None

    # Convert timestamps to datetime objects
    dates = [datetime.fromtimestamp(ts / 1000) for ts, _ in index_prices]
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

    metrics = calculate_financial_metrics(prices)

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

    # Print performance metrics
    print_performance_metrics(
        strategy_name, initial_investment, investment_value[-1], prices, metrics
    )

    # Create performance data dictionary
    performance_data = create_performance_data(
        dates, prices, investment_value, initial_investment, metrics, freq
    )

    # Add start date and stablecoin allocation to performance data
    if start_date:
        if isinstance(start_date, str):
            performance_data["start_date"] = start_date
        else:
            performance_data["start_date"] = start_date.strftime("%Y-%m-%d")

    performance_data["stablecoin_allocation"] = stablecoin_allocation
    performance_data["apply_staking"] = apply_staking

    # Add fear and greed data info to performance data if available
    if fear_greed_data:
        performance_data["used_fear_greed_data"] = True
        performance_data["fear_greed_data_points"] = len(fear_greed_data)

    # Save to JSON file
    filename_parts = [method, freq]
    if start_date:
        if isinstance(start_date, str):
            filename_parts.append(start_date.replace("-", ""))
        else:
            filename_parts.append(start_date.strftime("%Y%m%d"))

    return index_prices, performance_data


def run_performance_analysis(
    tokens=DEFAULT_TOKENS,
    methods=DEFAULT_METHODS,
    rebalance_frequencies=DEFAULT_REBALANCE_FREQUENCIES,
    initial_investment=DEFAULT_INITIAL_INVESTMENT,
    start_date=None,
    data_dir="./",
    fear_greed_file=None,
    stablecoin_allocation=0.5,  # Default to 50% stablecoin allocation
):
    """
    Run a complete performance analysis for the specified tokens and strategies.

    Args:
        tokens (list): List of token symbols to analyze
        methods (list): List of weighting methods to use
        rebalance_frequencies (list): List of rebalancing frequencies to test
        initial_investment (float): Initial investment amount
        start_date (str or datetime): Optional start date for analysis
        data_dir (str): Directory containing the CSV files
        fear_greed_file (str): Path to the fear and greed index JSON file
        stablecoin_allocation (float): Percentage of portfolio to allocate to stablecoin (0.0-1.0)

    Returns:
        dict: Dictionary of index prices for each strategy and fear/greed data if available
    """
    # Load historical data
    historical_data = load_historical_data(tokens, data_dir)

    if not historical_data:
        print("Error: No historical data could be loaded.")
        return {}

    # Filter data by start date if provided
    if start_date:
        if isinstance(start_date, str):
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_date_obj = start_date

        start_timestamp = int(start_date_obj.timestamp() * 1000)
        historical_data = filter_data_by_start_date(historical_data, start_timestamp)

    historical_data = align_data_timestamps(historical_data)

    if not historical_data:
        print("Error: No usable data after filtering/alignment.")
        return {}

    # Load and process fear and greed data if provided
    fear_greed_data = None
    if fear_greed_file:
        print(f"Loading fear and greed index data from {fear_greed_file}")
        fear_greed_data = process_fear_greed_data(
            fear_greed_file,
            start_date=start_date,
            historical_data=historical_data,
        )
        if fear_greed_data:
            print(
                f"Successfully loaded {len(fear_greed_data)} fear and greed data points"
            )
        else:
            print("Warning: No usable fear and greed data after processing")

    # Perform validation
    is_valid, message = validate_data_length_consistency(historical_data)
    print(f"Data validation: {message}")

    # Display initial portfolio weights
    display_initial_weights(historical_data, methods)

    all_index_prices = {}

    # Get start date string for display and filenames
    start_date_str = None
    if start_date:
        if isinstance(start_date, str):
            start_date_str = start_date
        else:
            start_date_str = start_date.strftime("%Y-%m-%d")

    # Store performance data for plotting
    all_performance_data = {}

    # Get BTC benchmark data for comparison if BTC is included in tokens
    if "btc" in historical_data:
        btc_prices = historical_data["btc"]

        if btc_prices:
            btc_investment, btc_metrics = calculate_benchmark_performance(
                btc_prices, initial_investment
            )

            # Print BTC benchmark performance
            btc_values = [price for _, price, _ in btc_prices]
            benchmark_name = "BTC Benchmark"
            if start_date_str:
                benchmark_name = f"{benchmark_name} - Start: {start_date_str}"

            print_performance_metrics(
                benchmark_name,
                initial_investment,
                btc_investment[-1],
                btc_values,
                btc_metrics,
            )

            # Create BTC benchmark performance data for plotting
            btc_dates = [datetime.fromtimestamp(ts / 1000) for ts, _, _ in btc_prices]
            btc_performance_data = create_performance_data(
                btc_dates,
                btc_values,
                btc_investment,
                initial_investment,
                btc_metrics,
                "N/A",
            )
            all_performance_data["BTC Benchmark"] = btc_performance_data
        else:
            print("No BTC benchmark data available for the selected time period.")

    # Process each method and rebalancing frequency
    for method in methods:
        for freq in rebalance_frequencies:
            for apply_staking, use_fear_greed in [(False, False), (True, True)]:
                # Only use fear/greed data if it's available and we want to use it
                current_fear_greed = (
                    fear_greed_data if use_fear_greed and fear_greed_data else None
                )

                # Calculate, display, and save strategy performance
                index_prices, performance_data = (
                    calculate_and_save_strategy_performance(
                        historical_data,
                        method,
                        freq,
                        initial_investment,
                        start_date=start_date,
                        stablecoin_allocation=stablecoin_allocation,
                        fear_greed_data=current_fear_greed,
                        apply_staking=apply_staking,
                    )
                )

                # Store results only if we have data
                if index_prices:
                    key = f"{method}_{freq}_staking_{apply_staking}_feargreed_{use_fear_greed}"
                    if start_date_str:
                        key = f"{key}_{start_date_str.replace('-', '')}"
                    all_index_prices[key] = index_prices

                    # Create a descriptive strategy label for the plot
                    strategy_label = (
                        f"{method.replace('_', ' ').title()} ({freq.title()})"
                    )

                    # Add staking and fear/greed info to the label
                    strategy_parts = []
                    if not apply_staking:
                        strategy_parts.append("No Staking")
                    else:
                        strategy_parts.append("With Staking")

                    if use_fear_greed and fear_greed_data:
                        strategy_parts.append("With Fear/Greed")
                    else:
                        strategy_parts.append("No Fear/Greed")

                    # Combine all parts
                    strategy_label = f"{strategy_label} - {', '.join(strategy_parts)}"

                    all_performance_data[strategy_label] = performance_data

    # Generate performance comparison plot if requested
    if all_performance_data:
        base_filename = "strategy_comparison"
        if start_date_str:
            base_filename = f"{base_filename}_{start_date_str.replace('-', '')}"

        # Generate detailed performance plot if requested
        detailed_plot_filename = f"detailed_{base_filename}.png"
        print(
            f"Generating detailed performance analysis plot: {detailed_plot_filename}"
        )
        plot_detailed_performance(
            all_performance_data, output_file=detailed_plot_filename
        )

        # Generate performance-only plot without risk metrics
        performance_only_filename = f"performance_only_{base_filename}.png"
        print(f"Generating performance-only plot: {performance_only_filename}")
        plot_performance_only(
            all_performance_data, output_file=performance_only_filename
        )

        # Generate metrics-only plot without performance chart
        metrics_only_filename = f"metrics_only_{base_filename}.png"
        print(f"Generating metrics-only comparison plot: {metrics_only_filename}")
        plot_metrics_only(all_performance_data, output_file=metrics_only_filename)

    result = all_index_prices
    if fear_greed_data:
        result["fear_greed_data"] = fear_greed_data

    return result


def main():
    """Main entry point for command-line execution."""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Calculate crypto index fund performance"
    )
    parser.add_argument(
        "--start-date", type=str, help="Start date for analysis (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--tokens",
        type=str,
        nargs="+",
        default=DEFAULT_TOKENS,
        help="Token symbols to include in analysis",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_METHODS,
        help="Weighting methods to use",
    )
    parser.add_argument(
        "--rebalance",
        type=str,
        nargs="+",
        default=DEFAULT_REBALANCE_FREQUENCIES,
        help="Rebalancing frequencies to test",
    )
    parser.add_argument(
        "--investment",
        type=float,
        default=DEFAULT_INITIAL_INVESTMENT,
        help="Initial investment amount",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./dataset",
        help="Directory containing token data CSV files",
    )
    parser.add_argument(
        "--fear-greed-file",
        type=str,
        help="Path to the fear and greed index JSON file",
    )
    parser.add_argument(
        "--stablecoin-allocation",
        type=float,
        default=0.5,
        help="Percentage of portfolio to allocate to stablecoin (0.0-1.0)",
    )

    args = parser.parse_args()

    # Run the analysis with command line parameters
    run_performance_analysis(
        tokens=args.tokens,
        methods=args.methods,
        rebalance_frequencies=args.rebalance,
        initial_investment=args.investment,
        start_date=args.start_date,
        data_dir=args.data_dir,
        fear_greed_file=args.fear_greed_file,
        stablecoin_allocation=args.stablecoin_allocation,
    )


if __name__ == "__main__":
    main()
