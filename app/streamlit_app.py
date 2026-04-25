"""
Streamlit frontend — 4 pages:
  1. Customer Lookup
  2. RFM Analysis
  3. Model Performance
  4. Retention Ranking

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import dill
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.charts import (
    plot_confusion_matrix,
    plot_gamma_kde,
    plot_history_alive,
    plot_rfm_grid,
    plot_shap_force,
    plot_shap_summary,
    plot_survival_curves,
)
from core.config import (
    ARTIFACTS_DIR,
    BEST_THRESHOLD_PATH,
    BG_PATH,
    CHURN_DATA_PATH,
    CHURN_FEATURES,
    CLV_HORIZON_MONTHS,
    CPH_PATH,
    GG_PATH,
    LGBM_PATH,
    LIFETIME_DF_PATH,
    SCORES_DF_PATH,
    SURVIVAL_DF_PATH,
)
from core.data import load_customers, load_transactions
from core.features import compute_rfm_segments, compute_seg_stats
from core.scoring import get_payback_df, rank_customers, score_customer


def _load(path: Path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return dill.load(f)


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return {
        "lgbm": _load(LGBM_PATH),
        "bg": _load(BG_PATH),
        "gg": _load(GG_PATH),
        "cph": _load(CPH_PATH),
        "best_threshold": _load(BEST_THRESHOLD_PATH),
        "lifetime_df": _load(LIFETIME_DF_PATH),
        "survival_df": _load(SURVIVAL_DF_PATH),
        "churn_data": _load(CHURN_DATA_PATH),
        "scores_df": _load(SCORES_DF_PATH) if SCORES_DF_PATH.exists() else None,
    }


@st.cache_data(show_spinner="Loading transaction data...")
def load_data():
    return load_transactions(), load_customers()


def check_artifacts() -> bool:
    required = [LGBM_PATH, BG_PATH, GG_PATH, CPH_PATH,
                BEST_THRESHOLD_PATH, LIFETIME_DF_PATH, SURVIVAL_DF_PATH, CHURN_DATA_PATH]
    return all(p.exists() for p in required)


# ---------------------------------------------------------------------------
# Page 0 — Transactions Explorer
# ---------------------------------------------------------------------------

def page_transactions(transactions: pd.DataFrame):
    st.header("Transaction Explorer")
    st.write(f"Total records: **{len(transactions):,}** across **{transactions['customer_id'].nunique():,}** customers.")

    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns([2, 2, 1])

        # Customer ID filter
        customer_search = col1.text_input(
            "Customer ID (contains)",
            placeholder="e.g. C013",
        )

        # Date range filter
        date_min = transactions["transaction_date"].min().date()
        date_max = transactions["transaction_date"].max().date()
        date_range = col2.date_input(
            "Transaction date range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )

        # Amount range filter
        amt_min = float(transactions["amount"].min())
        amt_max = float(transactions["amount"].max())
        amount_range = col3.slider(
            "Amount range ($)",
            min_value=amt_min,
            max_value=amt_max,
            value=(amt_min, amt_max),
            step=1.0,
        )

    # Apply filters
    filtered = transactions.copy()

    if customer_search.strip():
        filtered = filtered[
            filtered["customer_id"].str.contains(customer_search.strip(), case=False, na=False)
        ]

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["transaction_date"].dt.date >= start_date)
            & (filtered["transaction_date"].dt.date <= end_date)
        ]

    filtered = filtered[
        (filtered["amount"] >= amount_range[0])
        & (filtered["amount"] <= amount_range[1])
    ]

    st.write(f"Showing **{len(filtered):,}** records after filters.")

    st.dataframe(
        filtered.sort_values("transaction_date", ascending=False)
        .reset_index(drop=True)
        .style.format({
            "transaction_date": lambda v: v.strftime("%Y-%m-%d") if pd.notna(v) else "",
            "amount": "${:.2f}",
        }),
        use_container_width=True,
        height=500,
    )

    # Summary stats for filtered slice
    if not filtered.empty:
        st.markdown("### Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Transactions", f"{len(filtered):,}")
        s2.metric("Unique Customers", f"{filtered['customer_id'].nunique():,}")
        s3.metric("Total Revenue", f"${filtered['amount'].sum():,.2f}")
        s4.metric("Avg Order Value", f"${filtered['amount'].mean():,.2f}")

# ---------------------------------------------------------------------------
# Page 1 — RFM Analysis
# ---------------------------------------------------------------------------

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


def page_rfm(transactions: pd.DataFrame, customers: pd.DataFrame):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    st.header("RFM Customer Segmentation")
    analysis_date = transactions["transaction_date"].max() + pd.Timedelta(days=1)

    with st.spinner("Computing RFM segments..."):
        rfm = transactions.groupby("customer_id").agg(
            recency=("transaction_date", lambda x: (analysis_date - x.max()).days),
            frequency=("transaction_date", "count"),
            monetary=("amount", "sum"),
        ).reset_index()

        rfm_q = rfm.copy()
        rfm_q["R_score"] = pd.qcut(rfm_q["recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
        rfm_q["F_score"] = pd.qcut(rfm_q["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm_q["M_score"] = pd.qcut(rfm_q["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm_q["segment"] = rfm_q.apply(_rfm_segment_rules, axis=1)

        total_customers = len(rfm_q)
        seg_stats = (
            rfm_q.groupby("segment")
            .agg(count=("customer_id", "count"), avg_monetary=("monetary", "mean"))
            .reset_index()
        )
        seg_stats["pct"] = (seg_stats["count"] / total_customers * 100).round(2)
        seg_stats = seg_stats.set_index("segment")

    # ── Segmentation Grid (exact notebook code) ──────────────────────────────
    grid = {
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
    colors = {
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
    }

    fig, ax = plt.subplots(figsize=(24, 10))
    ax.set_facecolor("#FAFAFA")

    for seg, (x, y, w, h) in grid.items():
        color = colors.get(seg, "#DDDDDD")
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02",
            linewidth=1.5, edgecolor="white", facecolor=color, alpha=0.85,
        )
        ax.add_patch(rect)

        if seg in seg_stats.index:
            count = int(seg_stats.loc[seg, "count"])
            pct   = float(seg_stats.loc[seg, "pct"])
            avg_m = float(seg_stats.loc[seg, "avg_monetary"])
        else:
            count, pct, avg_m = 0, 0.0, 0.0

        cx, cy = x + w / 2, y + h / 2
        ax.text(cx, cy + h * 0.20, seg,
                ha="center", va="center", fontsize=18, fontweight="bold", color="white")
        ax.text(cx, cy - h * 0.05, f"Customers: {count:,} ({pct}%)",
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

    st.pyplot(fig)

    # ── Segment Summary Table ─────────────────────────────────────────────────
    st.markdown("### Segment Summary")
    display_stats = seg_stats.reset_index().rename(columns={"pct": "pct (%)"})
    display_stats["avg_monetary"] = display_stats["avg_monetary"].map("${:,.0f}".format)
    display_stats = display_stats.sort_values("pct (%)", ascending=False).reset_index(drop=True)
    st.dataframe(display_stats, use_container_width=True)

    st.markdown("### Full RFM Table")
    st.dataframe(rfm_q, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 2 — Customer Lookup
# ---------------------------------------------------------------------------

def page_customer_lookup(models: dict, transactions: pd.DataFrame):
    st.header("Customer Lookup")
    st.write("Enter a customer ID to see their full risk and value profile.")

    customer_id = st.text_input("Customer ID", value="C01329", placeholder="e.g. C01329")
    num_months = st.slider("CLV / NPV horizon (months)", min_value=3, max_value=24, value=CLV_HORIZON_MONTHS)

    if st.button("Score Customer", type="primary"):
        with st.spinner("Scoring..."):
            try:
                result = score_customer(
                    customer_id=customer_id,
                    transactions=transactions,
                    lifetime_df=models["lifetime_df"],
                    lgbm=models["lgbm"],
                    bg=models["bg"],
                    gg=models["gg"],
                    cph=models["cph"],
                    num_months=num_months,
                )
            except ValueError as e:
                st.error(str(e))
                return

        st.markdown("### Scores")
        cols = st.columns(3)
        cols[0].metric("Churn Probability", f"{result['churn_probability']:.1%}", delta=result["churn_label"])
        cols[1].metric("P(alive) — BG/NBD", f"{result['p_alive']:.1%}")
        cols[2].metric("Expected Remaining Lifetime", f"{result['expected_remaining_lifetime']:.1f} months")

        cols2 = st.columns(2)
        cols2[0].metric("CLV — BG/NBD + GG", f"${result['clv_bgnbd']:,.2f}")
        cols2[1].metric("CLV — Survival NPV", f"${result['clv_survival']:,.2f}")

        st.markdown("### P(alive) History")
        try:
            fig = plot_history_alive(models["bg"], customer_id, transactions)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not render P(alive) chart: {e}")

        st.markdown(f"### Survival-weighted NPV ({num_months} months)")
        try:
            payback = get_payback_df(
                customer_id=customer_id,
                transactions=transactions,
                lifetime_df=models["lifetime_df"],
                bg=models["bg"],
                cph=models["cph"],
                num_months=num_months,
            )
            st.dataframe(payback.style.format({
                "Survival Probability": "{:.1%}",
                "Monthly Profit": "${:.2f}",
                "Avg Expected Monthly Profit": "${:.2f}",
                "NPV of Avg Expected Monthly Profit": "${:.2f}",
                "Cumulative NPV": "${:.2f}",
            }), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute payback table: {e}")

        st.markdown("### Survival Curve")
        try:
            import matplotlib.pyplot as plt
            from core.scoring import get_customer_profile
            profile = get_customer_profile(customer_id, transactions)
            tenure = int(profile["tenure"].iloc[0])
            surv = models["cph"].predict_survival_function(profile, conditional_after=[tenure])
            fig2, ax = plt.subplots(figsize=(8, 4))
            ax.plot(surv.index, surv.values.flatten(), color="#3498db")
            ax.axvline(x=tenure, color="gray", linestyle="--", alpha=0.5, label="Current tenure")
            ax.set_xlabel("Days (lifetime)")
            ax.set_ylabel("P(survival)")
            ax.set_title(f"Survival Curve — {customer_id}")
            ax.legend()
            fig2.tight_layout()
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Could not render survival curve: {e}")


# ---------------------------------------------------------------------------
# Page 3 — Model Performance
# ---------------------------------------------------------------------------

def page_model_performance(models: dict):
    st.header("Model Performance")

    churn_data = models["churn_data"]
    train_ids = churn_data["customer_id"].sample(frac=0.7, random_state=42)
    test = churn_data[~churn_data["customer_id"].isin(train_ids)]
    X_test = test[CHURN_FEATURES]
    y_test = test["churn"].astype(int)

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "SHAP Summary", "SHAP Force Plot"])

    with tab1:
        threshold = st.slider(
            "Classification threshold",
            min_value=0.1, max_value=0.9,
            value=float(models["best_threshold"]), step=0.01,
        )
        fig = plot_confusion_matrix(models["lgbm"], X_test, y_test, threshold)
        st.pyplot(fig)

    with tab2:
        st.write("Feature importance via SHAP values (LightGBM).")
        import shap
        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(models["lgbm"])
            fig = plot_shap_summary(explainer, X_test)
        st.pyplot(fig)

    with tab3:
        idx = st.number_input("Row index in test set", min_value=0, max_value=len(X_test) - 1, value=1)
        import shap
        with st.spinner("Computing SHAP force plot..."):
            explainer = shap.TreeExplainer(models["lgbm"])
            fig = plot_shap_force(explainer, X_test, int(idx))
        st.pyplot(fig)

    st.markdown("### Gamma-Gamma Amount Distribution")
    fig_kde = plot_gamma_kde(models["gg"])
    st.pyplot(fig_kde)

    st.markdown("### Sample Survival Curves (CoxPH)")
    fig_surv = plot_survival_curves(models["cph"], models["survival_df"])
    st.pyplot(fig_surv)


# ---------------------------------------------------------------------------
# Page 4 — Retention Ranking
# ---------------------------------------------------------------------------

def page_retention(models: dict, transactions: pd.DataFrame):
    st.header("Retention Ranking")
    st.write("Identify which customers to prioritise with a retention budget.")

    col1, col2 = st.columns(2)
    strategy = col1.selectbox(
        "Strategy",
        options=["high_churn_probability", "low_palive", "high_clv_high_churn"],
        index=2,
    )
    top_k = col2.slider("Top K customers", min_value=10, max_value=500, value=100, step=10)

    if st.button("Rank Customers", type="primary"):
        scores_df = models.get("scores_df")
        if scores_df is None:
            st.error("Batch scores not found. Please run `python train.py` to generate them.")
            return

        with st.spinner("Ranking..."):
            ranked = rank_customers(scores_df, strategy, top_k)

        st.success(f"Top {len(ranked)} customers ranked by **{strategy}**")
        st.dataframe(
            ranked[["customer_id", "churn_probability", "p_alive", "clv_bgnbd", "clv_survival", "priority_score"]]
            .style.format({
                "churn_probability": "{:.1%}",
                "p_alive": "{:.1%}",
                "clv_bgnbd": "${:.0f}",
                "clv_survival": "${:.0f}",
                "priority_score": "{:.4f}",
            }),
            use_container_width=True,
        )

        st.markdown("### Priority Score Distribution")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(ranked["priority_score"], bins=30, color="#3498db", edgecolor="white")
        ax.set_xlabel("Priority Score")
        ax.set_ylabel("Count")
        ax.set_title(f"Priority Score — {strategy}")
        fig.tight_layout()
        st.pyplot(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Customer Growth & Retention",
        page_icon="📊",
        layout="wide",
    )
    st.title("Customer Growth & Retention Dashboard")

    if not check_artifacts():
        st.error(
            "Model artifacts not found. Please run `python train.py` first to train and save the models."
        )
        st.code("python train.py", language="bash")
        return

    models = load_models()
    transactions, customers = load_data()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Transactions",
            "RFM Analysis",
            "Customer Lookup",
            "Model Performance",
            "Retention Ranking",
        ],
    )

    if page == "Transactions":
        page_transactions(transactions)
    elif page == "RFM Analysis":
        page_rfm(transactions, customers)
    elif page == "Customer Lookup":
        page_customer_lookup(models, transactions)
    elif page == "Model Performance":
        page_model_performance(models)
    elif page == "Retention Ranking":
        page_retention(models, transactions)


if __name__ == "__main__":
    main()
