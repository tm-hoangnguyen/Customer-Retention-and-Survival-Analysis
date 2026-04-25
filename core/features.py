"""
Feature engineering — exact replication of explore_data.ipynb logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import CUTOFF_DATE, PREDICTION_WINDOW


# ---------------------------------------------------------------------------
# RFM
# ---------------------------------------------------------------------------

def compute_rfm(transactions: pd.DataFrame, analysis_date: pd.Timestamp) -> pd.DataFrame:
    rfm = transactions.groupby("customer_id").agg(
        recency=("transaction_date", lambda x: (analysis_date - x.max()).days),
        frequency=("transaction_date", "count"),
        monetary=("amount", "sum"),
    ).reset_index()
    return rfm


def _rfm_segment_rules(row: pd.Series) -> str:
    r, f = row["R_score"], row["F_score"]
    if r >= 4 and f >= 4:
        return "Champions"
    if 2 <= r < 4 and f >= 3:
        return "Loyal Customers"
    if r <= 2 and f >= 4:
        return "Cannot Lose Them"
    if r <= 2 and 2 <= f <= 4:
        return "At Risk"
    if r <= 2 and f <= 2:
        return "Hibernating"
    if 2 <= r <= 3 and 2 <= f <= 3:
        return "Need Attention"
    if 2 <= r <= 3 and f <= 2:
        return "About To Sleep"
    if r >= 3 and 1 <= f <= 3:
        return "Potential Loyalists"
    if 3 <= r <= 4 and f <= 1:
        return "Promising"
    if r >= 4 and f <= 1:
        return "New Customers"
    return "Others"


def compute_rfm_segments(transactions: pd.DataFrame, analysis_date: pd.Timestamp) -> pd.DataFrame:
    rfm = compute_rfm(transactions, analysis_date)
    rfm_q = rfm.copy()
    rfm_q["R_score"] = pd.qcut(rfm_q["recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm_q["F_score"] = pd.qcut(rfm_q["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm_q["M_score"] = pd.qcut(rfm_q["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm_q["segment"] = rfm_q.apply(_rfm_segment_rules, axis=1)
    return rfm_q


def compute_seg_stats(rfm_q: pd.DataFrame) -> pd.DataFrame:
    seg_stats = rfm_q.groupby("segment").agg(
        count=("customer_id", "count"),
        avg_monetary=("monetary", "mean"),
    ).reset_index()
    seg_stats["pct"] = seg_stats["count"] / seg_stats["count"].sum()
    return seg_stats


# ---------------------------------------------------------------------------
# Churn labels
# ---------------------------------------------------------------------------

def create_churn_labels(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp = CUTOFF_DATE,
    prediction_window: int = PREDICTION_WINDOW,
) -> pd.DataFrame:
    future = df[
        (df["transaction_date"] > cutoff_date)
        & (df["transaction_date"] <= cutoff_date + pd.Timedelta(days=prediction_window))
    ]
    active_customers = future["customer_id"].unique()
    labels = (
        df[df["transaction_date"] <= cutoff_date][["customer_id"]]
        .drop_duplicates()
    )
    labels["churn"] = ~labels["customer_id"].isin(active_customers)
    return labels.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Churn features
# ---------------------------------------------------------------------------

def create_churn_features(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp = CUTOFF_DATE,
) -> pd.DataFrame:
    hist = df[df["transaction_date"] <= cutoff_date]

    agg = hist.groupby("customer_id").agg(
        recency=("transaction_date", lambda x: (cutoff_date - x.max()).days),
        frequency=("transaction_date", "count"),
        monetary=("amount", "sum"),
        freq_30d=("transaction_date", lambda x: (x >= cutoff_date - pd.Timedelta(days=30)).sum()),
        freq_90d=("transaction_date", lambda x: (x >= cutoff_date - pd.Timedelta(days=90)).sum()),
        amount_30d=("amount", lambda s: s[hist.loc[s.index, "transaction_date"] >= cutoff_date - pd.Timedelta(days=30)].sum()),
        amount_90d=("amount", lambda s: s[hist.loc[s.index, "transaction_date"] >= cutoff_date - pd.Timedelta(days=90)].sum()),
        freq_past_90d_to_cutoff=("transaction_date", lambda x: (x >= cutoff_date - pd.Timedelta(days=90)).sum()),
        freq_past_270d_to_180d=("transaction_date", lambda x: (
            (x >= cutoff_date - pd.Timedelta(days=270)) & (x < cutoff_date - pd.Timedelta(days=180))
        ).sum()),
        max_gap=("transaction_date", lambda x: x.sort_values().diff().dt.days.max()),
    )

    agg["freq30/90"] = agg["freq_30d"] / (agg["freq_90d"] + 1e-6)
    agg["amount30/90"] = agg["amount_30d"] / (agg["amount_90d"] + 1e-6)
    agg["subtraction_90_270"] = agg["freq_past_90d_to_cutoff"] - agg["freq_past_270d_to_180d"]

    return agg.reset_index()


def build_churn_dataset(
    transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp = CUTOFF_DATE,
    prediction_window: int = PREDICTION_WINDOW,
) -> pd.DataFrame:
    features = create_churn_features(transactions, cutoff_date)
    labels = create_churn_labels(transactions, cutoff_date, prediction_window)
    return features.merge(labels, on="customer_id")


# ---------------------------------------------------------------------------
# Survival dataset
# ---------------------------------------------------------------------------

def build_survival_df(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    cutoff_date: pd.Timestamp = CUTOFF_DATE,
) -> pd.DataFrame:
    hist = transactions[transactions["transaction_date"] <= cutoff_date]

    survival_df = hist.groupby("customer_id").agg(
        first_txn_date=("transaction_date", "min"),
        num_transactions=("transaction_date", "count"),
        avg_amount=("amount", "mean"),
    ).reset_index()

    survival_df["tenure"] = (cutoff_date - survival_df["first_txn_date"]).dt.days
    survival_df = survival_df.merge(
        customers[["customer_id", "true_lifetime_days"]], on="customer_id", how="left"
    )
    survival_df["residual_days"] = survival_df["true_lifetime_days"] - survival_df["tenure"]

    observation_end = transactions["transaction_date"].max()
    max_possible_residual = (
        (observation_end - survival_df["first_txn_date"]).dt.days - survival_df["tenure"]
    )
    survival_df["event"] = (survival_df["residual_days"] < max_possible_residual).astype(int)

    survival_df = survival_df[survival_df["residual_days"] > 0].copy()
    survival_df = survival_df.drop(columns=["first_txn_date", "residual_days"])

    return survival_df
