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
    "Champions":           "#5B9BD5",
    "Loyal Customers":     "#A9B7C6",
    "Cannot Lose Them":    "#E06666",
    "At Risk":             "#F6B26B",
    "Hibernating":         "#B6D7E8",
    "About To Sleep":      "#CFD8DC",
    "Need Attention":      "#CEAD00",
    "Potential Loyalists": "#6FCF97",
    "Promising":           "#C5E1A5",
    "New Customers":       "#C5E1A5",
    "Others":              "#DDDDDD",
}

# Grid layout: segment -> (x_left, y_bottom, width, height) in R×F coordinate space
_SEGMENT_GRID: dict[str, tuple[float, float, float, float]] = {
    "Champions":           (4, 3, 1, 2),
    "Loyal Customers":     (2, 3, 2, 2),
    "Cannot Lose Them":    (0, 4, 2, 1),
    "At Risk":             (0, 2, 2, 2),
    "Hibernating":         (0, 0, 2, 2),
    "About To Sleep":      (2, 0, 1, 2),
    "Need Attention":      (2, 2, 1, 1),
    "Potential Loyalists": (3, 1, 2, 2),
    "Promising":           (3, 0, 1, 1),
    "New Customers":       (4, 0, 1, 1),
}


def plot_rfm_grid(rfm_q: pd.DataFrame, seg_stats: pd.DataFrame) -> plt.Figure:
    stats = seg_stats.set_index("segment") if "segment" in seg_stats.columns else seg_stats.copy()

    fig, ax = plt.subplots(figsize=(24, 10))
    ax.set_facecolor("#FAFAFA")

    for seg, (x, y, w, h) in _SEGMENT_GRID.items():
        color = SEGMENT_COLORS.get(seg, "#DDDDDD")
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="white",
            facecolor=color,
            alpha=0.85,
        )
        ax.add_patch(rect)

        if seg in stats.index:
            count   = int(stats.loc[seg, "count"])
            pct     = float(stats.loc[seg, "pct"])      # ratio 0-1
            avg_m   = float(stats.loc[seg, "avg_monetary"])
        else:
            count, pct, avg_m = 0, 0.0, 0.0

        cx, cy = x + w / 2, y + h / 2
        ax.text(cx, cy + h * 0.20, seg,
                ha="center", va="center", fontsize=18, fontweight="bold", color="white")
        ax.text(cx, cy - h * 0.05, f"Customers: {count:,} ({pct:.2%})",
                ha="center", va="center", fontsize=14, color="white")
        ax.text(cx, cy - h * 0.25, f"Avg. Monetary: ${avg_m:,.2f}",
                ha="center", va="center", fontsize=14, color="white")

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("Recency Score", fontsize=16)
    ax.set_ylabel("Frequency Score", fontsize=16)
    ax.set_title("RFM Customer Segmentation", fontsize=20, fontweight="bold", pad=15)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.grid(False)
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
