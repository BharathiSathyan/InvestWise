
import yfinance as yf
import pandas as pd
import os

from risklens.config import (
    ALL_TICKERS,
    START_DATE,
    END_DATE
)

# Create data folder if it does not exist
os.makedirs("data", exist_ok=True)

print("Downloading historical stock data...")

# Download adjusted closing prices
data = yf.download(
    ALL_TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False
)

# Extract closing prices
prices = data["Close"]

# Remove rows where all prices are missing
prices = prices.dropna(how="all")

# Sort dates
prices = prices.sort_index()

# Save as CSV
prices.to_csv("data/prices.csv")

print("\nDownload completed!")

print("Dataset shape:", prices.shape)

print("\nFirst five rows:")
print(prices.head())

print("\nLast five rows:")
print(prices.tail())

print("\nMissing values:")
print(prices.isnull().sum())

print("\nSaved successfully to data/prices.csv")
