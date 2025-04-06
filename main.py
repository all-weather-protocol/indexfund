"""
Main module for the indexfund package.
Contains the main analysis function and command-line interface.
"""

import argparse

# Import from config module
from config import (
    DEFAULT_INITIAL_INVESTMENT,
    DEFAULT_METHODS,
    DEFAULT_REBALANCE_FREQUENCIES,
    DEFAULT_TOKENS,
)

# Import from data_loading module
from core.data_loading import load_and_prepare_data

# Import from reporting module
from core.reporting import (
    generate_performance_plots,
    print_benchmark_metrics,
    print_strategy_metrics,
)

# Import from strategy module
from core.strategy import (
    calculate_strategy_performance,
    create_strategy_data,
    format_strategy_name,
    generate_strategy_key,
    process_benchmark_data,
)

# Import from weighting module
from core.weighting import display_initial_weights


def run_performance_analysis(
    tokens=DEFAULT_TOKENS,
    methods=DEFAULT_METHODS,
    rebalance_frequencies=DEFAULT_REBALANCE_FREQUENCIES,
    initial_investment=DEFAULT_INITIAL_INVESTMENT,
    start_date=None,
    data_dir="./",
    fear_greed_file=None,
    stablecoin_allocation=0.5,  # Default to 50% stablecoin allocation
    generate_plots=True,
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
        generate_plots (bool): Whether to generate performance plots

    Returns:
        dict: Dictionary containing analysis results:
            - 'index_prices': Dict of index prices for each strategy
            - 'performance_data': Dict of performance data for each strategy
            - 'fear_greed_data': Fear and greed data if available
    """
    # Load and prepare all necessary data
    historical_data, fear_greed_data = load_and_prepare_data(
        tokens, data_dir, start_date, fear_greed_file
    )

    if not historical_data:
        return {"error": "No data available"}

    # Display initial portfolio weights
    display_initial_weights(historical_data, methods)

    # Initialize result dictionaries
    all_index_prices = {}
    all_performance_data = {}

    # Process benchmark data
    btc_performance_data, has_benchmark = process_benchmark_data(
        historical_data, initial_investment, start_date
    )
    if has_benchmark:
        # Print benchmark metrics
        print_benchmark_metrics(historical_data, initial_investment, start_date)

        # Add to performance data
        all_performance_data["BTC Benchmark"] = btc_performance_data

    # Process each method and rebalancing frequency
    for method in methods:
        for freq in rebalance_frequencies:
            for apply_staking, use_fear_greed in [(False, False), (True, True)]:
                # Only use fear/greed data if it's available and we want to use it
                current_fear_greed = (
                    fear_greed_data if use_fear_greed and fear_greed_data else None
                )

                # Calculate strategy performance
                index_prices, performance_metrics, investment_value = (
                    calculate_strategy_performance(
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

                # Skip if no data is available
                if not index_prices:
                    continue

                # Print strategy metrics
                prices = [price for _, price in index_prices]
                print_strategy_metrics(
                    method,
                    freq,
                    apply_staking,
                    stablecoin_allocation,
                    start_date,
                    initial_investment,
                    investment_value,
                    prices,
                    performance_metrics,
                )

                # Create performance data
                performance_data = create_strategy_data(
                    index_prices,
                    performance_metrics,
                    investment_value,
                    initial_investment,
                    method,
                    freq,
                    apply_staking,
                    stablecoin_allocation,
                    current_fear_greed,
                    start_date,
                )

                # Store results
                strategy_key = generate_strategy_key(
                    method, freq, apply_staking, use_fear_greed, start_date
                )
                all_index_prices[strategy_key] = index_prices

                # Create a descriptive strategy label for the plot
                strategy_label = format_strategy_name(
                    method, freq, apply_staking, stablecoin_allocation, start_date
                )
                if use_fear_greed and fear_greed_data:
                    strategy_label += " (With Fear/Greed)"

                all_performance_data[strategy_label] = performance_data

    # Generate plots if requested
    if generate_plots:
        generate_performance_plots(all_performance_data, start_date)

    # Prepare result dictionary
    result = {
        "index_prices": all_index_prices,
        "performance_data": all_performance_data,
    }

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
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generation of performance plots",
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
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
