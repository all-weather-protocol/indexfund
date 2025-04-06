"""
Configuration settings for the indexfund package.
Contains constants and configuration values used across the application.
"""

# Staking configuration
STAKING_CONFIG = {
    "btc": 0.01,  # 1% APR for BTC
    "eth": 0.04,  # 4% APR for ETH staking
    "aave": 0.03,  # 3% APR for AAVE staking
    "uni": 0.02,  # 2% APR for UNI
    "pendle": 0.15,  # 15% APR for Pendle staking
    "link": 0.03,  # 3% APR for LINK
    "sol": 0.1,  # 10% APR for SOL staking
    "stablecoin": 0.15,
}

# Financial calculation constants
RISK_FREE_RATE = 0.05  # 5% annual risk-free rate
TRADING_DAYS_PER_YEAR = 252  # Number of trading days in a year

# Date format constants
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default portfolio parameters
DEFAULT_TOKENS = ["btc", "eth", "sol", "stablecoin"]
DEFAULT_METHODS = ["market_cap"]
DEFAULT_REBALANCE_FREQUENCIES = ["quarterly"]
DEFAULT_INITIAL_INVESTMENT = 10000

# Rebalancing configuration
DEFAULT_SWAP_FEE = 0.01  # 1% swap fee for simulating exchange trading costs
