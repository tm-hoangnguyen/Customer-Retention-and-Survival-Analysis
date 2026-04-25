"""
Chart functions — each returns a matplotlib Figure.
Replicates every visualization from explore_data.ipynb.
"""
from __future__ import annotations

from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lifetimes import plotting as lt_plotting
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# ---------------------------------------------------------------------------
# 1. RFM Segmentation Grid (Cell 7)
# ---------------------------------------------------------------------------

SEGMENT_COLORS = {
    "Champions": "#2ecc71",
    "Loyal": "#3498db",
    "New": "#9b59b6",
    "At-Risk": "#e74c3c",
    "Hibernating": "#95a5a6",
    "Others": "#f39c12",
}


def plot_rfm_grid(rfm_q: pd.DataFrame, seg_stats: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(24, 10))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("Recency Score", fontsize=13)
    ax.set_ylabel("Frequency Score", fontsize=13)
    ax.set_title("RFM Customer Segmentation", fontsize=16, fontweight="bold")
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))

    for _, row in rfm_q.iterrows():
        r, f, seg = row["R_score"], row["F_score"], row["segment"]
        color = SEGMENT_COLORS.get(seg, "#bdc3c7")
        rect = patches.FancyBboxPatch(
            (r - 0.45, f - 0.45), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            linewidth=0.3,
            edgecolor="white",
            facecolor=color,
            alpha=0.6,
        )
        ax.add_patch(rect)

    handles = [
        patches.Patch(facecolor=c, label=s) for s, c in SEGMENT_COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Confusion Matrix (Cell 21)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    lgbm: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> plt.Figure:
    probas = lgbm.predict_proba(X_test)[:, 1]
    preds = (probas >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds, normalize="all")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"]).plot(
        values_format=".1%", ax=ax
    )
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. SHAP summary beeswarm (Cell 23)
# ---------------------------------------------------------------------------

def plot_shap_summary(explainer: Any, X_test: pd.DataFrame) -> plt.Figure:
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance (LightGBM)")
    fig = plt.gcf()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. SHAP force plot (Cells 24 / 26)
# ---------------------------------------------------------------------------

def plot_shap_force(explainer: Any, X_test: pd.DataFrame, idx: int) -> plt.Figure:
    shap_values = explainer.shap_values(X_test)
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        X_test.iloc[idx],
        matplotlib=True,
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. P(alive) history for a customer (Cell 37)
# ---------------------------------------------------------------------------

def plot_history_alive(
    bg: Any,
    customer_id: str,
    transactions: pd.DataFrame,
) -> plt.Figure:
    customer_txns = (
        transactions[transactions["customer_id"] == customer_id]
        .assign(transaction_date=lambda x: x["transaction_date"].astype("datetime64[ns]"))
    )
    fig, ax = plt.subplots()
    lt_plotting.plot_history_alive(
        bg,
        t=90,
        transactions=customer_txns,
        datetime_col="transaction_date",
        ax=ax,
    )
    plt.xticks(rotation=45, ha="right")
    ax.set_title(f"P(alive) History — Customer {customer_id}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Gamma KDE — transaction amount distribution (Cell 41)
# ---------------------------------------------------------------------------

def plot_gamma_kde(gg: Any, n_samples: int = 100_000, seed: int = 42) -> plt.Figure:
    p, q, v = gg.params_["p"], gg.params_["q"], gg.params_["v"]
    rng = np.random.default_rng(seed)
    samples = rng.gamma(shape=p, scale=v, size=n_samples)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(samples, ax=ax, fill=True, color="skyblue")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Transaction Amounts (Gamma Distribution)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Individual survival curves (Cell 47)
# ---------------------------------------------------------------------------

def plot_survival_curves(
    cph: Any,
    survival_df: pd.DataFrame,
    n: int = 5,
    seed: int = 1,
) -> plt.Figure:
    sample = survival_df.sample(n, random_state=seed)
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in sample.iterrows():
        surv_fn = cph.predict_survival_function(
            row[["num_transactions", "avg_amount", "tenure"]].to_frame().T
        )
        ax.plot(surv_fn.index, surv_fn.values.flatten(), label=f"Customer {row['customer_id']}")
    ax.set_title("Individual Survival Curves (CoxPH)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
