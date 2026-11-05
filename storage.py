import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "data"

PROFILE_FILE = DATA_DIR / "user_profile.json"
HOLDINGS_FILE = DATA_DIR / "holdings.json"
ALLOCATION_FILE = DATA_DIR / "last_allocation.json"


DEFAULT_PROFILE = {
    "profile_id": "default_user",
    "capital": 100000.0,
    "risk_profile": "Moderate",
    "investment_horizon": "Long-term (above 3 years)",
    "target_return": 12.0,
    "max_weight": 0.30
}


def load_json(file_path, default):
    """Load JSON data or return its default value."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        return default.copy()

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return default.copy()


def save_json(file_path, data):
    """Save data as formatted JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_profile():
    return load_json(PROFILE_FILE, DEFAULT_PROFILE)


def save_profile(profile):
    save_json(PROFILE_FILE, profile)


def load_holdings():
    holdings = load_json(HOLDINGS_FILE, {"holdings": []})
    return holdings.get("holdings", [])


def save_holdings(holdings):
    save_json(HOLDINGS_FILE, {"holdings": holdings})


def load_last_allocation():
    return load_json(ALLOCATION_FILE, {"weights": {}})


def save_last_allocation(allocation):
    save_json(ALLOCATION_FILE, allocation)


def holdings_to_dataframe(holdings):
    columns = ["Ticker", "Quantity", "Average Purchase Price"]

    if not holdings:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(holdings, columns=columns)