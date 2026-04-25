"""
Centralised constants — mirror the values used in explore_data.ipynb.
Any change here propagates to all training, scoring and API code.
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Time anchors
# ---------------------------------------------------------------------------
CUTOFF_DATE: pd.Timestamp = pd.Timestamp("2025-10-02")  # 2025-12-31 - 90 days
PREDICTION_WINDOW: int = 90  # days used to define churn labels

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
LGBM_PARAMS: dict = {
    "objective": "binary",
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": 42,
    "metric": "average_precision",
}

XGB_PARAMS: dict = {
    "objective": "binary:logistic",
    "n_estimators": 100,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": 42,
    "eval_metric": "aucpr",
}

BGNBD_PENALIZER: float = 0.001
GG_PENALIZER: float = 0.01

TRAIN_FRAC: float = 0.7
TRAIN_RANDOM_STATE: int = 42
F1_THRESHOLD_RANGE: tuple[float, float, float] = (0.1, 0.9, 0.01)

# ---------------------------------------------------------------------------
# CLV / NPV assumptions
# ---------------------------------------------------------------------------
ANNUAL_IRR: float = 0.10
CLV_HORIZON_MONTHS: int = 12
GG_CLV_HORIZON_MONTHS: int = 3       # used in gg.customer_lifetime_value
GG_DISCOUNT_RATE: float = 0.01       # monthly rate passed to lifetimes

# ---------------------------------------------------------------------------
# Churn label thresholds
# ---------------------------------------------------------------------------
CHURN_HIGH_RISK_THRESHOLD: float = 0.6
CHURN_MEDIUM_RISK_THRESHOLD: float = 0.4

# ---------------------------------------------------------------------------
# Feature columns (order must match training)
# ---------------------------------------------------------------------------
CHURN_FEATURES: list[str] = [
    "recency",
    "frequency",
    "monetary",
    "freq_30d",
    "freq_90d",
    "amount_30d",
    "amount_90d",
    "freq_past_90d_to_cutoff",
    "freq_past_270d_to_180d",
    "max_gap",
    "freq30/90",
    "amount30/90",
    "subtraction_90_270",
]

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
from pathlib import Path

ARTIFACTS_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"
LGBM_PATH = ARTIFACTS_DIR / "lgbm.pkl"
BG_PATH = ARTIFACTS_DIR / "bg.pkl"
GG_PATH = ARTIFACTS_DIR / "gg.pkl"
CPH_PATH = ARTIFACTS_DIR / "cph.pkl"
LIFETIME_DF_PATH = ARTIFACTS_DIR / "lifetime_df.pkl"
SURVIVAL_DF_PATH = ARTIFACTS_DIR / "survival_df.pkl"
CHURN_DATA_PATH = ARTIFACTS_DIR / "churn_data.pkl"
BEST_THRESHOLD_PATH = ARTIFACTS_DIR / "best_threshold.pkl"
SCORES_DF_PATH = ARTIFACTS_DIR / "scores_df.pkl"
