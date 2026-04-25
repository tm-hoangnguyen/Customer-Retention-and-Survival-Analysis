"""
Data loading utilities — single source of truth for raw data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["transaction_date"])
    assert df.shape[1] == 3, f"Expected 3 columns, got {df.shape[1]}"
    assert df["transaction_date"].notna().all(), "Null transaction_date values found"
    assert df["amount"].notna().all(), "Null amount values found"
    return df


def load_customers() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["signup_date"])
    assert df.shape[1] == 3, f"Expected 3 columns, got {df.shape[1]}"
    assert df["true_lifetime_days"].notna().all(), "Null true_lifetime_days values found"
    return df
