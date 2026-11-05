import pandas as pd
import yfinance as yf

from pathlib import Path
from datetime import date, timedelta

from risklens.config import (
    DATA_FILE,
    TICKERS,
    BENCHMARK,
    ALL_TICKERS
)


DATA_PATH = Path(__file__).resolve().parent.parent / DATA_FILE


def load_price_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "prices.csv not found. Download market data first."
        )

    prices = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True
    )

    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    return prices


def get_aligned_prices():
    prices = load_price_data()

    missing = [
        ticker for ticker in ALL_TICKERS
        if ticker not in prices.columns
    ]

    if missing:
        raise ValueError(f"Missing tickers: {missing}")

    return prices[ALL_TICKERS].dropna()


def refresh_market_data(force_refresh=False, max_age_days=3):
    """
    Incrementally update the local historical price cache.

    Returns:
        prices: Updated price DataFrame
        status: Information about the refresh
    """

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        existing = load_price_data()
        last_date = existing.index.max().date()

        if (
            not force_refresh
            and (date.today() - last_date).days <= max_age_days
        ):
            return existing, {
                "updated": False,
                "message": "Local market data is sufficiently recent.",
                "last_date": str(last_date)
            }

        start_date = last_date - timedelta(days=5)

    else:
        existing = pd.DataFrame()
        start_date = date(2021, 1, 1)

    end_date = date.today() + timedelta(days=1)

    try:
        downloaded = yf.download(
            ALL_TICKERS,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=True,
            progress=False
        )

        if downloaded.empty:
            raise ValueError("No new market data was returned.")

        if isinstance(downloaded.columns, pd.MultiIndex):
            if "Close" in downloaded.columns.get_level_values(0):
                new_prices = downloaded["Close"]
            else:
                new_prices = downloaded.xs(
                    "Close",
                    axis=1,
                    level=1
                )
        else:
            new_prices = downloaded

        new_prices.index = pd.to_datetime(new_prices.index)
        new_prices.index.name = "Date"

        new_prices = new_prices.apply(
            pd.to_numeric,
            errors="coerce"
        )

        new_prices = new_prices.reindex(columns=ALL_TICKERS)

        if existing.empty:
            combined = new_prices
        else:
            combined = pd.concat([existing, new_prices])

        combined = combined.sort_index()
        combined = combined[
            ~combined.index.duplicated(keep="last")
        ]

        # Preserve dates with complete observations for all assets.
        combined = combined.dropna(subset=ALL_TICKERS)

        if combined.empty:
            raise ValueError("No complete price observations available.")

        combined.to_csv(DATA_PATH)

        return combined, {
            "updated": True,
            "message": "Market data refreshed successfully.",
            "last_date": str(combined.index.max().date())
        }

    except Exception as error:
        if not existing.empty:
            return existing, {
                "updated": False,
                "message": f"Refresh failed; using cached data. Error: {error}",
                "last_date": str(existing.index.max().date())
            }

        raise RuntimeError(
            f"Unable to fetch market data: {error}"
        ) from error