"""
One-time training script.

Run:
    python train.py

Trains all 4 models (lgbm, bg, gg, cph), saves them and supporting
dataframes to artifacts/ so the API and Streamlit can load them at startup.
Also pre-computes batch scores for all customers so API startup is instant.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import dill  # handles lifetimes lambdas that pickle cannot serialize
import lifetimes as lt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

from core.config import (
    ARTIFACTS_DIR,
    BEST_THRESHOLD_PATH,
    BG_PATH,
    BGNBD_PENALIZER,
    CHURN_DATA_PATH,
    CHURN_FEATURES,
    CPH_PATH,
    CUTOFF_DATE,
    F1_THRESHOLD_RANGE,
    GG_CLV_HORIZON_MONTHS,
    GG_DISCOUNT_RATE,
    GG_PATH,
    GG_PENALIZER,
    LGBM_PARAMS,
    LGBM_PATH,
    LIFETIME_DF_PATH,
    PREDICTION_WINDOW,
    SCORES_DF_PATH,
    SURVIVAL_DF_PATH,
    TRAIN_FRAC,
    TRAIN_RANDOM_STATE,
)
from core.data import load_customers, load_transactions
from core.features import build_churn_dataset, build_survival_df
from core.scoring import score_all_customers


def _needs_dill(obj: object) -> bool:
    try:
        pickle.dumps(obj)
        return False
    except Exception:
        return True


def _save(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializer = dill if _needs_dill(obj) else pickle
    with open(path, "wb") as f:
        serializer.dump(obj, f)
    print(f"  Saved -> {path.relative_to(Path.cwd())}")


def train_churn_classifier(transactions: pd.DataFrame) -> tuple:
    print("\n[1/4] Training churn classifier (LightGBM)...")
    data = build_churn_dataset(transactions, CUTOFF_DATE, PREDICTION_WINDOW)

    train_ids = data["customer_id"].sample(frac=TRAIN_FRAC, random_state=TRAIN_RANDOM_STATE)
    train = data[data["customer_id"].isin(train_ids)]
    test = data[~data["customer_id"].isin(train_ids)]

    X_train = train[CHURN_FEATURES]
    y_train = train["churn"].astype(int)
    X_test = test[CHURN_FEATURES]
    y_test = test["churn"].astype(int)

    lgbm = LGBMClassifier(**LGBM_PARAMS)
    lgbm.fit(X_train, y_train)

    thresholds = np.arange(*F1_THRESHOLD_RANGE)
    probas = lgbm.predict_proba(X_test)[:, 1]
    f1_scores = [f1_score(y_test, (probas >= t).astype(int)) for t in thresholds]
    best_threshold = float(thresholds[np.argmax(f1_scores)])
    print(f"  Best F1 threshold: {best_threshold:.3f}  F1: {max(f1_scores):.3f}")

    _save(lgbm, LGBM_PATH)
    _save(best_threshold, BEST_THRESHOLD_PATH)
    _save(data, CHURN_DATA_PATH)
    return lgbm, best_threshold, data


def train_bgnbd_gg(transactions: pd.DataFrame) -> tuple:
    print("\n[2/4] Training BG/NBD + Gamma-Gamma...")
    observation_end = transactions["transaction_date"].max()
    print(f"  BG/NBD observation period end: {observation_end.date()} (full dataset)")
    lifetime_df = lt.utils.summary_data_from_transaction_data(
        transactions,
        customer_id_col="customer_id",
        datetime_col="transaction_date",
        monetary_value_col="amount",
        observation_period_end=observation_end,
        freq="D",
    )

    bg = lt.BetaGeoFitter(penalizer_coef=BGNBD_PENALIZER)
    bg.fit(
        frequency=lifetime_df["frequency"],
        recency=lifetime_df["recency"],
        T=lifetime_df["T"],
    )
    print(bg.summary)

    gg = lt.GammaGammaFitter(penalizer_coef=GG_PENALIZER)
    lifetime_df_ggf = lifetime_df[lifetime_df["monetary_value"] > 0]
    gg.fit(
        frequency=lifetime_df_ggf["frequency"],
        monetary_value=lifetime_df_ggf["monetary_value"],
    )

    lifetime_df.loc[lifetime_df_ggf.index, "expected_avg_order_value"] = (
        gg.conditional_expected_average_profit(
            lifetime_df_ggf["frequency"],
            lifetime_df_ggf["monetary_value"],
        )
    )

    lifetime_df["predicted_purchases_3M"] = bg.predict(
        t=90,
        frequency=lifetime_df["frequency"],
        recency=lifetime_df["recency"],
        T=lifetime_df["T"],
    )

    lifetime_df["CLV_3M"] = gg.customer_lifetime_value(
        bg,
        lifetime_df_ggf["frequency"],
        lifetime_df_ggf["recency"],
        lifetime_df_ggf["T"],
        lifetime_df_ggf["monetary_value"],
        time=GG_CLV_HORIZON_MONTHS,
        discount_rate=GG_DISCOUNT_RATE,
        freq="D",
    )

    _save(bg, BG_PATH)
    _save(gg, GG_PATH)
    _save(lifetime_df, LIFETIME_DF_PATH)
    return bg, gg, lifetime_df


def train_coxph(transactions: pd.DataFrame, customers: pd.DataFrame) -> tuple:
    print("\n[3/4] Training CoxPH...")
    survival_df = build_survival_df(transactions, customers, CUTOFF_DATE)
    print(f"  Shape: {survival_df.shape}  Event rate: {survival_df['event'].mean():.2%}")

    cph = CoxPHFitter()
    cph.fit(
        survival_df.drop(columns=["customer_id"]),
        duration_col="true_lifetime_days",
        event_col="event",
        entry_col="tenure",
    )
    cph.print_summary()

    _save(cph, CPH_PATH)
    _save(survival_df, SURVIVAL_DF_PATH)
    return cph, survival_df


def main() -> None:
    print("Loading data...")
    transactions = load_transactions()
    customers = load_customers()
    print(f"  Transactions: {len(transactions):,}  Customers: {len(customers):,}")

    lgbm, best_threshold, churn_data = train_churn_classifier(transactions)
    bg, gg, lifetime_df = train_bgnbd_gg(transactions)
    cph, survival_df = train_coxph(transactions, customers)

    print("\n[4/4] Pre-computing batch scores for all customers...")
    scores_df = score_all_customers(
        transactions=transactions,
        lifetime_df=lifetime_df,
        lgbm=lgbm,
        bg=bg,
        gg=gg,
        cph=cph,
    )
    print(f"  Scored {len(scores_df):,} customers.")
    _save(scores_df, SCORES_DF_PATH)

    print("\nAll models trained and saved to artifacts/")


if __name__ == "__main__":
    main()
