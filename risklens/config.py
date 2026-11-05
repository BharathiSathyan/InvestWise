
# Common configuration for InvestWise

TICKERS = [
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "RELIANCE.NS",
    "ITC.NS",
    "SUNPHARMA.NS",
    "BHARTIARTL.NS",
    "LT.NS",
    "M&M.NS"
]

BENCHMARK = "^NSEI"

ALL_TICKERS = TICKERS + [BENCHMARK]

START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

TRADING_DAYS = 252

DATA_FILE = "data/prices.csv"
