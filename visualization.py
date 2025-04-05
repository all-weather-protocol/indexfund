"""
Performance metrics calculation functions for the indexfund package.
Contains functions for calculating financial metrics like returns, volatility, etc.
"""

from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from config import STAKING_CONFIG


def print_performance_metrics(
    asset_name, initial_investment, final_value, prices, metrics
):
    """
    Print performance metrics for an investment.

    Args:
        asset_name (str): Name of the asset or strategy
        initial_investment (float): Initial investment amount
        final_value (float): Final investment value
        prices (list): List of prices over time
        metrics (dict): Dictionary of financial metrics
    """
    print(f"\n===== {asset_name} Performance =====")
    print(f"Initial Investment: ${initial_investment:.2f}")
    print(f"Final Value: ${final_value:.2f}")
    print(f"Total Return: {((prices[-1] - prices[0]) / prices[0]) * 100:.2f}%")
    print(f"Profit/Loss: ${final_value - initial_investment:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Volatility: {metrics['volatility']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")


def create_performance_data(
    dates, prices, investment_values, initial_investment, metrics, rebalance_frequency
):
    """
    Create a structured performance data dictionary for saving or analysis.

    Args:
        dates (list): List of datetime objects
        prices (list): List of index prices
        investment_values (list): List of investment values over time
        initial_investment (float): Initial investment amount
        metrics (dict): Dictionary of financial metrics
        rebalance_frequency (str): Rebalancing frequency used

    Returns:
        dict: Structured performance data dictionary
    """
    return {
        "dates": [date.strftime("%Y-%m-%d %H:%M:%S") for date in dates],
        "prices": prices,
        "investment_values": investment_values,
        "initial_investment": initial_investment,
        "final_value": investment_values[-1],
        "total_return": ((prices[-1] - prices[0]) / prices[0]) * 100,
        "profit_loss": investment_values[-1] - initial_investment,
        "staking_config": STAKING_CONFIG,
        "rebalance_frequency": rebalance_frequency,
        "metrics": {
            "max_drawdown": metrics["max_drawdown"],
            "volatility": metrics["volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
        },
    }


def plot_detailed_performance(performance_data_dict, output_file=None):
    """
    Create a comprehensive multi-panel plot showing various performance metrics.

    Args:
        performance_data_dict (dict): Dictionary mapping strategy names to their performance data
        output_file (str, optional): Path to save the plot image. If None, displays the plot

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig)

    # Plot 1: Investment Value Over Time
    ax1 = fig.add_subplot(gs[0, :])
    for strategy_name, data in performance_data_dict.items():
        # Convert date strings to datetime objects, then to matplotlib date numbers
        dates = [
            mdates.date2num(datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
            for date in data["dates"]
        ]
        ax1.plot(dates, data["investment_values"], label=strategy_name)

    ax1.set_title("Investment Value Over Time", fontsize=14)
    ax1.set_ylabel("Value ($)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Plot 2: Performance Metrics Comparison (Bar Chart)
    ax2 = fig.add_subplot(gs[1, 0])

    # Prepare data for bar chart
    strategies = list(performance_data_dict.keys())
    total_returns = [data["total_return"] for data in performance_data_dict.values()]

    x = np.arange(len(strategies))
    width = 0.7

    bars = ax2.bar(x, total_returns, width)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax2.set_title("Total Return (%)", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Risk Metrics (Bar Chart)
    ax3 = fig.add_subplot(gs[1, 1])

    # Prepare volatility data
    volatilities = [
        data["metrics"]["volatility"] for data in performance_data_dict.values()
    ]

    bars = ax3.bar(x, volatilities, width)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax3.set_title("Volatility (%)", fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, rotation=45, ha="right")
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Max Drawdown
    ax4 = fig.add_subplot(gs[2, 0])

    # Prepare drawdown data
    drawdowns = [
        data["metrics"]["max_drawdown"] for data in performance_data_dict.values()
    ]

    bars = ax4.bar(x, drawdowns, width, color="r")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax4.set_title("Maximum Drawdown (%)", fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies, rotation=45, ha="right")
    ax4.grid(True, alpha=0.3, axis="y")

    # Plot 5: Sharpe and Sortino Ratios
    ax5 = fig.add_subplot(gs[2, 1])

    # Prepare ratio data
    sharpe_ratios = [
        data["metrics"]["sharpe_ratio"] for data in performance_data_dict.values()
    ]
    sortino_ratios = [
        data["metrics"]["sortino_ratio"] for data in performance_data_dict.values()
    ]

    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax5.bar(x - width / 2, sharpe_ratios, width, label="Sharpe Ratio")
    bars2 = ax5.bar(x + width / 2, sortino_ratios, width, label="Sortino Ratio")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    ax5.set_title("Risk-Adjusted Return Ratios", fontsize=14)
    ax5.set_xticks(x)
    ax5.set_xticklabels(strategies, rotation=45, ha="right")
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")

    # Add title and adjust layout
    plt.suptitle("Detailed Performance Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        return None
    else:
        return fig


def plot_performance_only(performance_data_dict, output_file=None):
    """
    Create a simple plot showing only investment value over time without risk metrics.

    Args:
        performance_data_dict (dict): Dictionary mapping strategy names to their performance data
        output_file (str, optional): Path to save the plot image. If None, displays the plot

    Returns:
        matplotlib.figure.Figure: The figure object if not saved to file
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot investment value over time for each strategy
    for strategy_name, data in performance_data_dict.items():
        # Convert date strings to datetime objects, then to matplotlib date numbers
        dates = [
            mdates.date2num(datetime.strptime(date, "%Y-%m-%d %H:%M:%S"))
            for date in data["dates"]
        ]
        ax.plot(dates, data["investment_values"], label=strategy_name)

    # Add labels and styling
    ax.set_title("Investment Value Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Value ($)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Add initial and final values in legend
    legend_texts = []
    for strategy_name, data in performance_data_dict.items():
        initial = data["initial_investment"]
        final = data["final_value"]
        total_return = data["total_return"]
        legend_texts.append(
            f"{strategy_name}: ${initial:.0f} → ${final:.2f} ({total_return:.2f}%)"
        )

    ax.legend(legend_texts, loc="best")

    # Format x-axis to prevent date overlap
    plt.gcf().autofmt_xdate()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        return None
    else:
        return fig


def plot_metrics_only(performance_data_dict, output_file=None):
    """
    Create a plot showing only metrics comparison without the performance chart.

    Args:
        performance_data_dict (dict): Dictionary mapping strategy names to their performance data
        output_file (str, optional): Path to save the plot image. If None, displays the plot

    Returns:
        matplotlib.figure.Figure: The figure object if not saved to file
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Prepare data for bar charts
    strategies = list(performance_data_dict.keys())
    total_returns = [data["total_return"] for data in performance_data_dict.values()]
    volatilities = [
        data["metrics"]["volatility"] for data in performance_data_dict.values()
    ]
    drawdowns = [
        data["metrics"]["max_drawdown"] for data in performance_data_dict.values()
    ]
    sharpe_ratios = [
        data["metrics"]["sharpe_ratio"] for data in performance_data_dict.values()
    ]
    sortino_ratios = [
        data["metrics"]["sortino_ratio"] for data in performance_data_dict.values()
    ]

    x = np.arange(len(strategies))
    width = 0.7

    # Plot 1: Total Returns
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x, total_returns, width)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax1.set_title("Total Return (%)", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Volatility
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(x, volatilities, width)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax2.set_title("Volatility (%)", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Max Drawdown
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(x, drawdowns, width, color="r")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax3.set_title("Maximum Drawdown (%)", fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, rotation=45, ha="right")
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Sharpe and Sortino Ratios
    ax4 = fig.add_subplot(gs[1, 1])
    width = 0.35

    bars1 = ax4.bar(x - width / 2, sharpe_ratios, width, label="Sharpe Ratio")
    bars2 = ax4.bar(x + width / 2, sortino_ratios, width, label="Sortino Ratio")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    ax4.set_title("Risk-Adjusted Return Ratios", fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies, rotation=45, ha="right")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    # Add title and adjust layout
    plt.suptitle("Performance Metrics Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        return None
    else:
        return fig
