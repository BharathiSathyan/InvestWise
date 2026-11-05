
import os
import pandas as pd
import numpy as np

from risklens.data import get_aligned_prices

from risklens.metrics import calculate_metrics

from risklens.optimize import (
    generate_portfolios,
    RISK_PROFILES
)

from risklens.costs import calculate_transaction_cost

from risklens.backtest import run_backtest


os.makedirs("outputs", exist_ok=True)

# Load data
prices = get_aligned_prices()

# Calculate metrics
metrics = calculate_metrics(prices)

mu = metrics["expected_returns"]

cov = metrics["sample_covariance"]

# Investor inputs
capital = 500000

gamma = RISK_PROFILES["Moderate"]

max_weight = 0.30

cost_rate = 0.001

n = len(mu)

current = np.ones(n) / n

# Optimize
portfolios = generate_portfolios(
    mu.values,
    cov.values,
    current,
    gamma,
    max_weight,
    cost_rate
)

# Save allocations
allocation = pd.DataFrame({
    "Stock": mu.index,
    "Expected Return": mu.values,
    "Equal Weight": portfolios["equal_weight"],
    "Markowitz Weight": portfolios["markowitz"]["weights"],
    "Cost Aware Weight": portfolios["cost_aware"]["weights"]
})

allocation.to_csv(
    "outputs/allocations.csv",
    index=False
)

# Save backtest
backtest = run_backtest(
    prices,
    initial_capital=capital,
    risk_profile="Moderate",
    max_weight=max_weight,
    transaction_cost_rate=cost_rate
)

backtest["nav"].to_csv(
    "outputs/performance.csv"
)

# Save performance summary
summary = pd.DataFrame(
    backtest["performance"]
).T

summary.to_csv(
    "outputs/performance_summary.csv"
)

print("All outputs saved successfully!")
