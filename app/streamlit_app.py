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
# Page 1 — Customer Lookup
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
# Page 2 — RFM Analysis
# ---------------------------------------------------------------------------

def page_rfm(transactions: pd.DataFrame, customers: pd.DataFrame):
    st.header("RFM Customer Segmentation")
    analysis_date = transactions["transaction_date"].max() + pd.Timedelta(days=1)
    st.write(f"Analysis date: **{analysis_date.date()}**")

    with st.spinner("Computing RFM segments..."):
        rfm_q = compute_rfm_segments(transactions, analysis_date)
        seg_stats = compute_seg_stats(rfm_q)

    st.markdown("### Segment Summary")
    display_stats = seg_stats.copy()
    display_stats["pct"] = display_stats["pct"].map("{:.1%}".format)
    display_stats["avg_monetary"] = display_stats["avg_monetary"].map("${:.0f}".format)
    st.dataframe(display_stats, use_container_width=True)

    st.markdown("### Segmentation Grid")
    fig = plot_rfm_grid(rfm_q, seg_stats)
    st.pyplot(fig)

    st.markdown("### Full RFM Table")
    st.dataframe(rfm_q, use_container_width=True)


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
        ["Customer Lookup", "RFM Analysis", "Model Performance", "Retention Ranking"],
    )

    if page == "Customer Lookup":
        page_customer_lookup(models, transactions)
    elif page == "RFM Analysis":
        page_rfm(transactions, customers)
    elif page == "Model Performance":
        page_model_performance(models)
    elif page == "Retention Ranking":
        page_retention(models, transactions)


if __name__ == "__main__":
    main()
