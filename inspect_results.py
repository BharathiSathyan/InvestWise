from risklens.data import get_aligned_prices, load_price_data
from risklens.metrics import calculate_metrics
from risklens.optimize import generate_portfolios
import pandas as pd

print("\n========== RAW PRICE DATA ==========")

raw = load_price_data()

print("Shape:", raw.shape)
print("Columns:", raw.columns.tolist())
print(raw.head(3))
print(raw.tail(3))

print("\n========== ALIGNED PRICE DATA ==========")

prices = get_aligned_prices()

print("Shape:", prices.shape)
print("Columns:", prices.columns.tolist())
print(prices.head(3))
print(prices.tail(3))

print("\n========== FINANCIAL METRICS ==========")

metrics = calculate_metrics(prices)

for key, value in metrics.items():
    print(f"\n{key}:")
    if isinstance(value, pd.DataFrame):
        print("Shape:", value.shape)
        print(value.head(3))
    elif isinstance(value, pd.Series):
        print(value.head(5))
    else:
        print(value)

print("\n========== PORTFOLIO GENERATION ==========")

mu = metrics["expected_returns"]
cov = metrics["sample_covariance"]

current_weights = pd.Series(
    0.1,
    index=mu.index
)

portfolios = generate_portfolios(
    mu.values,
    cov.values,
    current_weights.values,
    3.0,
    0.30,
    0.001
)

print("Portfolio result type:", type(portfolios))

if isinstance(portfolios, dict):
    for key, value in portfolios.items():
        print(f"\n{key}:")
        print(value)
else:
    print(portfolios)