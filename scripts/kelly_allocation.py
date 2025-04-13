import numpy as np


def kelly_allocation(expected_returns, volatility, pool_names=None, rf=0.0):
    """
    Calculate Kelly-optimal allocation weights for multiple stablecoin pools.

    Parameters:
    - expected_returns: list or np.array of expected APY for each pool (e.g., [0.06, 0.08, 0.10])
    - volatility: list or np.array of standard deviation (risk) for each pool (e.g., [0.01, 0.02, 0.03])
    - pool_names: list of pool names (e.g., ['A', 'B', 'C']) — optional
    - rf: risk-free rate, default is 0

    Returns:
    - Dictionary of pool_name -> weight (as percentage)
    """
    expected_returns = np.array(expected_returns)
    volatility = np.array(volatility)
    if pool_names is None:
        pool_names = [f"Pool_{i+1}" for i in range(len(expected_returns))]

    # Covariance matrix assuming independence (diagonal only)
    cov_matrix = np.diag(volatility**2)
    excess_return = expected_returns - rf

    # Kelly formula
    inv_cov = np.linalg.inv(cov_matrix)
    raw_weights = inv_cov.dot(excess_return)
    norm_weights = raw_weights / raw_weights.sum()  # Normalize to sum to 1

    # Format result
    return {
        name: round(weight * 100, 2) for name, weight in zip(pool_names, norm_weights)
    }


expected_returns = [0.2, 0.15, 0.11, 0.28]
volatility = [0.09, 0.1, 0.1, 0.1]  # Estimated standard deviation
pool_names = ["ousdt", "msusd", "susd", "alp"]

weights = kelly_allocation(expected_returns, volatility, pool_names)
print("\nKelly Optimal Allocation Weights:")
print("-" * 30)
for pool, weight in weights.items():
    print(f"{pool}: {weight:.2f}%")
print("-" * 30)
