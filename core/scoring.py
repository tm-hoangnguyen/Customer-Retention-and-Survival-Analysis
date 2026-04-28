"""
Scoring functions — replicates the scoring logic from explore_data.ipynb.
All functions are pure (no side-effects) and accept pre-loaded model objects.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.config import (
    ANNUAL_IRR,
    CHURN_FEATURES,
    CHURN_HIGH_RISK_THRESHOLD,
    CHURN_MEDIUM_RISK_THRESHOLD,
    CLV_HORIZON_MONTHS,
    CUTOFF_DATE,
    GG_CLV_HORIZON_MONTHS,
    GG_DISCOUNT_RATE,
)
from core.features import create_churn_features


def _resolve_cutoff(
    transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp | None,
) -> pd.Timestamp:
    """Return cutoff_date if provided, otherwise the latest transaction date."""
    if cutoff_date is not None:
        return cutoff_date
    return transactions["transaction_date"].max()


# ---------------------------------------------------------------------------
# Customer profile helpers
# ---------------------------------------------------------------------------

def get_customer_profile(
    customer_id: str,
    transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One-row DataFrame of CoxPH covariates for a given customer."""
    cutoff_date = _resolve_cutoff(transactions, cutoff_date)
    hist = transactions[
        (transactions["customer_id"] == customer_id)
        & (transactions["transaction_date"] <= cutoff_date)
    ]
    if hist.empty:
        raise ValueError(f"No transactions found for '{customer_id}' up to {cutoff_date.date()}")

    first_txn = hist["transaction_date"].min()
    return pd.DataFrame({
        "num_transactions": [len(hist)],
        "avg_amount": [hist["amount"].mean()],
        "tenure": [(cutoff_date - first_txn).days],
    })


def get_monthly_profit(
    customer_id: str,
    num_months: int,
    lifetime_df: pd.DataFrame,
    transactions: pd.DataFrame,
    bg: Any,
    cutoff_date: pd.Timestamp | None = None,
) -> np.ndarray:
    """
    Expected monthly profit array of length num_months.
    monthly_profit[n] = expected_avg_order_value x BG/NBD purchase increment in month n+1
    """
    cutoff_date = _resolve_cutoff(transactions, cutoff_date)
    row = lifetime_df.loc[lifetime_df.index == customer_id]

    if row.empty or pd.isna(row["expected_avg_order_value"].iloc[0]):
        hist = transactions[
            (transactions["customer_id"] == customer_id)
            & (transactions["transaction_date"] <= cutoff_date)
        ]
        avg = float(hist["amount"].mean()) if not hist.empty else 0.0
        return np.full(num_months, avg)

    avg_order_value = float(row["expected_avg_order_value"].iloc[0])
    freq = float(row["frequency"].iloc[0])
    rec = float(row["recency"].iloc[0])
    T = float(row["T"].iloc[0])

    purchases = [
        float(
            bg.predict(t=n * 30, frequency=freq, recency=rec, T=T)
            - bg.predict(t=(n - 1) * 30, frequency=freq, recency=rec, T=T)
        )
        for n in range(1, num_months + 1)
    ]
    return avg_order_value * np.array(purchases)


def get_payback_df(
    customer_id: str,
    transactions: pd.DataFrame,
    lifetime_df: pd.DataFrame,
    bg: Any,
    cph: Any,
    num_months: int = CLV_HORIZON_MONTHS,
    annual_irr: float = ANNUAL_IRR,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Survival-weighted NPV table — one row per contract month."""
    cutoff_date = _resolve_cutoff(transactions, cutoff_date)
    profile = get_customer_profile(customer_id, transactions, cutoff_date)
    tenure = int(profile["tenure"].iloc[0])
    irr = annual_irr / 12
    monthly_profits = get_monthly_profit(
        customer_id, num_months, lifetime_df, transactions, bg, cutoff_date
    )

    time_points = [tenure + 30 * n for n in range(num_months)]
    surv = cph.predict_survival_function(profile, conditional_after=[tenure], times=time_points)
    surv_values = surv.iloc[:, 0].values

    month0 = pd.DataFrame({
        "Contract Month": [0],
        "Survival Probability": [1.0],
        "Monthly Profit": [0.0],
        "Avg Expected Monthly Profit": [0.0],
        "NPV of Avg Expected Monthly Profit": [0.0],
        "Cumulative NPV": [0.0],
    })

    df = pd.DataFrame({
        "Contract Month": range(1, num_months + 1),
        "Survival Probability": np.round(surv_values, 4),
        "Monthly Profit": np.round(monthly_profits, 2),
    })
    df["Avg Expected Monthly Profit"] = np.round(df["Survival Probability"] * df["Monthly Profit"], 2)
    df["NPV of Avg Expected Monthly Profit"] = np.round(
        df["Avg Expected Monthly Profit"] / ((1 + irr) ** df["Contract Month"]), 2
    )
    df["Cumulative NPV"] = df["NPV of Avg Expected Monthly Profit"].cumsum().round(2)

    return pd.concat([month0, df], ignore_index=True).set_index("Contract Month")


# ---------------------------------------------------------------------------
# Individual model scorers
# ---------------------------------------------------------------------------

def _churn_label(prob: float) -> str:
    if prob >= CHURN_HIGH_RISK_THRESHOLD:
        return "high_risk"
    if prob >= CHURN_MEDIUM_RISK_THRESHOLD:
        return "medium_risk"
    return "low_risk"


def score_churn(
    customer_id: str,
    transactions: pd.DataFrame,
    lgbm: Any,
    cutoff_date: pd.Timestamp = CUTOFF_DATE,
) -> dict:
    features = create_churn_features(transactions, cutoff_date)
    row = features[features["customer_id"] == customer_id]
    if row.empty:
        raise ValueError(f"No feature data for '{customer_id}'")
    X = row[CHURN_FEATURES]
    prob = float(lgbm.predict_proba(X)[0, 1])
    return {"churn_probability": round(prob, 4), "churn_label": _churn_label(prob)}


def bgnbd_gg_horizon_floor(horizon_months: int) -> int:
    """Mirror Streamlit CLV slider (min 3 months) before GG customer_lifetime_value."""
    return max(int(horizon_months), 3)


def score_bgnbd(
    customer_id: str,
    lifetime_df: pd.DataFrame,
    bg: Any,
    gg: Any,
    *,
    horizon_months: int | None = None,
) -> dict:
    """
    If ``horizon_months`` is set, recomputes CLV via ``gg.customer_lifetime_value``
    at that horizon (same as Streamlit). Otherwise reads precomputed ``CLV_3M`` column.
    """
    row = lifetime_df.loc[lifetime_df.index == customer_id]
    if row.empty:
        raise ValueError(f"Customer '{customer_id}' not in lifetime_df")

    p_alive = float(
        np.atleast_1d(
            bg.conditional_probability_alive(
                frequency=float(row["frequency"].iloc[0]),
                recency=float(row["recency"].iloc[0]),
                T=float(row["T"].iloc[0]),
            )
        )[0]
    )

    if horizon_months is not None:
        t_h = bgnbd_gg_horizon_floor(horizon_months)
        mv = float(row["monetary_value"].iloc[0])
        if mv > 0:
            clv_raw = gg.customer_lifetime_value(
                bg,
                row["frequency"],
                row["recency"],
                row["T"],
                row["monetary_value"],
                time=t_h,
                discount_rate=GG_DISCOUNT_RATE,
                freq="D",
            )
            clv_val = float(clv_raw.iloc[0])
        else:
            clv_val = (
                float(row["CLV_3M"].iloc[0])
                if not pd.isna(row.get("CLV_3M", pd.Series([np.nan])).iloc[0])
                else 0.0
            )
    else:
        clv_val = float(row["CLV_3M"].iloc[0]) if not pd.isna(row.get("CLV_3M", pd.Series([np.nan])).iloc[0]) else 0.0
    return {"p_alive": round(p_alive, 4), "clv_bgnbd": round(clv_val, 2)}


def score_survival(
    customer_id: str,
    transactions: pd.DataFrame,
    cph: Any,
    cutoff_date: pd.Timestamp = CUTOFF_DATE,
) -> dict:
    profile = get_customer_profile(customer_id, transactions, cutoff_date)
    tenure = int(profile["tenure"].iloc[0])

    time_points = [tenure + 30 * n for n in range(1, 13)]
    surv = cph.predict_survival_function(profile, conditional_after=[tenure], times=time_points)
    surv_vals = surv.iloc[:, 0].values

    survival_curve = [
        {"day": 30 * n, "prob": round(float(p), 4)}
        for n, p in enumerate(surv_vals, start=1)
    ]

    full_surv = cph.predict_survival_function(profile, conditional_after=[tenure])
    expected_remaining_lifetime_threshold = 0.5
    below = full_surv[full_surv.iloc[:, 0] <= expected_remaining_lifetime_threshold]
    if below.empty:
        expected_remaining = float(full_surv.index[-1]) - tenure
    else:
        expected_remaining = float(below.index[0]) - tenure

    return {
        "survival_curve": survival_curve,
        "expected_remaining_lifetime": round(max(expected_remaining, 0) / 30, 2),
    }


# ---------------------------------------------------------------------------
# Unified scorer
# ---------------------------------------------------------------------------

def score_customer(
    customer_id: str,
    transactions: pd.DataFrame,
    lifetime_df: pd.DataFrame,
    lgbm: Any,
    bg: Any,
    gg: Any,
    cph: Any,
    num_months: int = CLV_HORIZON_MONTHS,
    cutoff_date: pd.Timestamp | None = None,
) -> dict:
    # Churn + survival snapshot: both use CUTOFF_DATE so metrics share the same Oct 2 reference
    # NPV / CLV: uses latest available date for maximum accuracy in forward projections
    npv_cutoff = _resolve_cutoff(transactions, cutoff_date)
    churn = score_churn(customer_id, transactions, lgbm, CUTOFF_DATE)
    bgnbd = score_bgnbd(customer_id, lifetime_df, bg, gg)
    surv = score_survival(customer_id, transactions, cph, CUTOFF_DATE)
    payback = get_payback_df(
        customer_id, transactions, lifetime_df, bg, cph, num_months, ANNUAL_IRR, npv_cutoff
    )
    clv_survival = float(payback["Cumulative NPV"].iloc[-1])

    return {
        "customer_id": customer_id,
        "churn_probability": churn["churn_probability"],
        "churn_label": churn["churn_label"],
        "p_alive": bgnbd["p_alive"],
        "clv_bgnbd": bgnbd["clv_bgnbd"],
        "expected_remaining_lifetime": surv["expected_remaining_lifetime"],
        "clv_survival": round(clv_survival, 2),
    }


# ---------------------------------------------------------------------------
# Batch scoring for retention ranking
# ---------------------------------------------------------------------------

def score_all_customers(
    transactions: pd.DataFrame,
    lifetime_df: pd.DataFrame,
    lgbm: Any,
    bg: Any,
    gg: Any,
    cph: Any,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Vectorised batch scoring — runs in seconds by avoiding per-customer loops.
    Churn features use CUTOFF_DATE (training cutoff) for distributional consistency.
    Survival / CLV use the latest available transaction date for accuracy.
    """
    survival_cutoff = _resolve_cutoff(transactions, cutoff_date)
    hist = transactions[transactions["transaction_date"] <= survival_cutoff]

    # --- Churn features + probabilities (fully vectorised) ---
    # Always use training CUTOFF_DATE so features match the training distribution
    churn_features = create_churn_features(transactions, CUTOFF_DATE)
    churn_probas = lgbm.predict_proba(churn_features[CHURN_FEATURES])[:, 1]
    churn_df = churn_features[["customer_id"]].copy()
    churn_df["churn_probability"] = np.round(churn_probas, 4)
    churn_df["churn_label"] = churn_df["churn_probability"].map(_churn_label)

    # --- BG/NBD p_alive (vectorised) ---
    ldf = lifetime_df.copy().reset_index()
    p_alive_vals = np.atleast_1d(
        bg.conditional_probability_alive(
            frequency=ldf["frequency"].values,
            recency=ldf["recency"].values,
            T=ldf["T"].values,
        )
    )
    bgnbd_df = pd.DataFrame({
        "customer_id": ldf["customer_id"] if "customer_id" in ldf.columns else ldf.index,
        "p_alive": np.round(p_alive_vals, 4),
        "clv_bgnbd": ldf["CLV_3M"].fillna(0).round(2).values,
    })

    # --- CoxPH: build all profiles at once, predict survival in one call ---
    profile_df = hist.groupby("customer_id").agg(
        first_txn_date=("transaction_date", "min"),
        num_transactions=("transaction_date", "count"),
        avg_amount=("amount", "mean"),
    ).reset_index()
    profile_df["tenure"] = (survival_cutoff - profile_df["first_txn_date"]).dt.days

    # Predict full survival function for all customers at once
    cox_input = profile_df[["num_transactions", "avg_amount", "tenure"]].copy()
    surv_all = cph.predict_survival_function(cox_input)  # index=days, columns=customers

    # For each customer compute expected remaining lifetime and 12-month CLV
    # Using surv_all (already computed for all customers) — no extra model calls needed
    surv_index = surv_all.index.values  # sorted day values from training
    clv_survival_list = []
    expected_lifetime_list = []
    irr = ANNUAL_IRR / 12
    months_arr = np.arange(1, CLV_HORIZON_MONTHS + 1)

    for col_idx, (_, prow) in enumerate(profile_df.iterrows()):
        tenure = int(prow["tenure"])
        cid = prow["customer_id"]
        surv_vals = surv_all.iloc[:, col_idx].values  # raw S(t) for this customer

        # S(tenure) — find closest index at or below tenure
        idx_tenure = np.searchsorted(surv_index, tenure, side="right") - 1
        idx_tenure = max(idx_tenure, 0)
        s_tenure = max(float(surv_vals[idx_tenure]), 1e-9)

        # Conditional survival at monthly intervals: S(tenure + 30n) / S(tenure)
        monthly_time_points = np.array([tenure + 30 * n for n in range(CLV_HORIZON_MONTHS)])
        idx_monthly = np.clip(
            np.searchsorted(surv_index, monthly_time_points, side="right") - 1,
            0, len(surv_vals) - 1,
        )
        surv_monthly = surv_vals[idx_monthly] / s_tenure

        # Expected remaining lifetime (first conditional survival <= 0.5)
        cond_full = surv_vals / s_tenure
        below_mask = cond_full <= 0.5
        if below_mask.any():
            exp_life = round(max(float(surv_index[below_mask][0]) - tenure, 0) / 30, 2)
        else:
            exp_life = round(max(float(surv_index[-1]) - tenure, 0) / 30, 2)
        expected_lifetime_list.append(exp_life)

        # Monthly profit from Gamma-Gamma + BG/NBD
        row_ldf = lifetime_df.loc[lifetime_df.index == cid]
        if row_ldf.empty or pd.isna(row_ldf["expected_avg_order_value"].iloc[0]):
            avg_profit = float(hist[hist["customer_id"] == cid]["amount"].mean() or 0)
        else:
            avg_profit = (
                float(row_ldf["expected_avg_order_value"].iloc[0])
                * float(row_ldf["predicted_purchases_3M"].iloc[0]) / 3
            )

        npv = float(np.sum(surv_monthly * avg_profit / ((1 + irr) ** months_arr)))
        clv_survival_list.append(round(npv, 2))

    profile_df["expected_remaining_lifetime"] = expected_lifetime_list
    profile_df["clv_survival"] = clv_survival_list

    cox_df = profile_df[["customer_id", "expected_remaining_lifetime", "clv_survival"]]

    # --- Merge all signals ---
    result = (
        churn_df
        .merge(bgnbd_df, on="customer_id", how="left")
        .merge(cox_df, on="customer_id", how="left")
    )
    result["p_alive"] = result["p_alive"].fillna(0.5)
    result["clv_bgnbd"] = result["clv_bgnbd"].fillna(0)
    result["clv_survival"] = result["clv_survival"].fillna(0)
    result["expected_remaining_lifetime"] = result["expected_remaining_lifetime"].fillna(0)

    return result.reset_index(drop=True)


def rank_customers(
    scores_df: pd.DataFrame,
    strategy: str,
    top_k: int,
) -> pd.DataFrame:
    df = scores_df.copy()

    if strategy == "high_churn_probability":
        df["priority_score"] = df["churn_probability"]
    elif strategy == "low_palive":
        df["priority_score"] = 1 - df["p_alive"]
    elif strategy == "high_clv_high_churn":
        clv_min, clv_max = df["clv_bgnbd"].min(), df["clv_bgnbd"].max()
        clv_norm = (df["clv_bgnbd"] - clv_min) / (clv_max - clv_min + 1e-9)
        df["priority_score"] = df["churn_probability"] * clv_norm
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'")

    df["priority_score"] = df["priority_score"].round(4)
    return df.sort_values("priority_score", ascending=False).head(top_k).reset_index(drop=True)
