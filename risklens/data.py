
import pandas as pd
import os

from risklens.config import DATA_FILE, TICKERS, BENCHMARK


def load_price_data():

    # Check whether CSV exists
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "prices.csv not found. Run the download script first."
        )

    # Read CSV
    prices = pd.read_csv(
        DATA_FILE,
        index_col=0,
        parse_dates=True
    )

    # Sort dates
    prices = prices.sort_index()

    # Convert all price columns to numeric
    prices = prices.apply(pd.to_numeric, errors="coerce")

    # Remove duplicate dates
    prices = prices[~prices.index.duplicated(keep="last")]

    # Check required columns
    missing_columns = [
        col for col in TICKERS + [BENCHMARK]
        if col not in prices.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in CSV: {missing_columns}"
        )

    print("CSV loaded successfully!")
    print("Dataset shape:", prices.shape)

    return prices


def get_aligned_prices():

    prices = load_price_data()

    # Use the common trading dates of all selected assets
    prices = prices[TICKERS + [BENCHMARK]].dropna()

    return prices
