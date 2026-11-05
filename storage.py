import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
PROFILE_FILE = DATA_DIR / "user_profile.json"
HOLDINGS_FILE = DATA_DIR / "holdings.json"
ALLOCATION_FILE = DATA_DIR / "last_allocation.json"

DEFAULT_PROFILE = {
    "profile_id": "default_user",
    "capital": 10000.0,
    "risk_profile": "Moderate",
    "investment_horizon": "Long-term (above 3 years)",
    "target_return": 12.0,
    "max_weight": 0.30,
}


def _load_json(path, default):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return default.copy()
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else default.copy()
    except (OSError, json.JSONDecodeError):
        return default.copy()


def _save_json(path, value):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=4)
    temporary.replace(path)


def load_profile():
    return {**DEFAULT_PROFILE, **_load_json(PROFILE_FILE, DEFAULT_PROFILE)}


def save_profile(profile):
    _save_json(PROFILE_FILE, profile)


def load_holdings():
    payload = _load_json(HOLDINGS_FILE, {"holdings": []})
    holdings = payload.get("holdings", [])
    return holdings if isinstance(holdings, list) else []


def save_holdings(holdings):
    _save_json(HOLDINGS_FILE, {"holdings": holdings})


def load_last_allocation():
    return _load_json(ALLOCATION_FILE, {"weights": {}})


def save_last_allocation(allocation):
    _save_json(ALLOCATION_FILE, allocation)
