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
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.charts import (
    convert_logodds_to_probability,
    plot_confusion_matrix,
    plot_gamma_kde,
    plot_history_alive,
    plot_rfm_grid,
    plot_shap_force,
    plot_shap_summary,
    plot_survival_curves,
)
from core.config import (
    ANNUAL_IRR,
    ARTIFACTS_DIR,
    BEST_THRESHOLD_PATH,
    BG_PATH,
    CHURN_DATA_PATH,
    CHURN_FEATURES,
    CHURN_HIGH_RISK_THRESHOLD,
    CHURN_MEDIUM_RISK_THRESHOLD,
    CLV_HORIZON_MONTHS,
    CPH_PATH,
    CUTOFF_DATE,
    GG_CLV_HORIZON_MONTHS,
    GG_DISCOUNT_RATE,
    GG_PATH,
    LGBM_PARAMS,
    LGBM_PATH,
    LIFETIME_DF_PATH,
    PREDICTION_WINDOW,
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


def _artifact_mtimes() -> tuple[float, ...]:
    """Return a tuple of artifact modification times — used as cache key for load_models."""
    paths = [
        LGBM_PATH, BG_PATH, GG_PATH, CPH_PATH,
        BEST_THRESHOLD_PATH, LIFETIME_DF_PATH, SURVIVAL_DF_PATH,
        CHURN_DATA_PATH, SCORES_DF_PATH,
    ]
    return tuple(p.stat().st_mtime if p.exists() else 0.0 for p in paths)


@st.cache_resource(show_spinner="Loading models...")
def load_models(mtimes: tuple[float, ...]):  # mtimes is the cache-busting key
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
    import matplotlib.pyplot as plt
    import numpy as np
    from core.scoring import get_customer_profile, score_bgnbd, score_churn, score_survival

    st.header("Customer Lookup")
    st.write("Enter a customer ID to see their full risk and value profile.")

    customer_id = st.text_input("Customer ID", value="C01329", placeholder="e.g. C01329")

    if st.button("Score Customer", type="primary"):
        with st.spinner("Scoring..."):
            try:
                churn = score_churn(customer_id, transactions, models["lgbm"])
                bgnbd = score_bgnbd(customer_id, models["lifetime_df"], models["bg"], models["gg"])
                surv  = score_survival(customer_id, transactions, models["cph"])
            except ValueError as e:
                st.error(str(e))
                # Clear any stale scores for a different customer
                st.session_state.pop("base_scores", None)
                st.session_state.pop("scored_customer", None)
                st.stop()

        st.session_state["base_scores"] = {**churn, **bgnbd, **surv}
        st.session_state["scored_customer"] = customer_id

    # ── Section 1: slider-independent scores ─────────────────────────────────
    if st.session_state.get("scored_customer") == customer_id and "base_scores" in st.session_state:
        scores = st.session_state["base_scores"]
        scoring_cutoff = transactions["transaction_date"].max()

        st.markdown("### Churn Classification with LightGBM")
        st.caption(
            f"**Churn Probability** is scored at the training cutoff **{CUTOFF_DATE.date()}** "
            f"(predicts whether the customer will buy in the 90-day window Oct 2 → Dec 31, 2025). "
            f"Risk label: **high_risk** ≥ 60%, **medium_risk** 40–59%, **low_risk** < 40%. "
            f"**Expected Remaining Lifetime** is the number of months from **{CUTOFF_DATE.date()}** "
            f"until the conditional survival probability first drops to or below 0.5 — "
            f"i.e. the point at which there is less than a 50% chance the customer is still active."
        )

        cols = st.columns(2)
        prob = scores["churn_probability"]
        label = scores["churn_label"]
        if prob >= CHURN_HIGH_RISK_THRESHOLD:
            border, text = "#e74c3c", "#e74c3c"
        elif prob >= CHURN_MEDIUM_RISK_THRESHOLD:
            border, text = "#f39c12", "#f39c12"
        else:
            border, text = "#27ae60", "#27ae60"
        bg = "transparent"
        cols[0].markdown(
            f"""<div style="border-left:4px solid {border}; background:{bg};
                            padding:10px 16px; border-radius:4px; line-height:1.4;">
                <div style="font-size:0.85em; color:#555;">Churn Probability</div>
                <div style="font-size:2em; font-weight:700; color:{text};">{prob:.1%}</div>
                <div style="font-size:0.85em; color:{text};">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        cols[1].metric("Expected Remaining Lifetime", f"{scores['expected_remaining_lifetime']:.1f} months")

        st.markdown("### Survival Curve (CoxPH)")
        st.markdown(
            f"<small>"
            f"<b>Gray dashed</b> — past: S(t) from first purchase to cut-off ({CUTOFF_DATE.date()}), "
            f"using this customer's covariates (transaction count, avg order value, tenure). "
            f"<b>Blue solid</b> — future: the same S(t) curve continuing past the cut-off on the same absolute scale "
            f". Both segments are one continuous CoxPH prediction; the split at the cut-off "
            f"marks where observed history ends and the forward projection begins. "
            f"The CoxPH model uses <code>true_lifetime_days</code> from <code>customers.csv</code> as the "
            f"duration column, with <code>tenure</code> as the left-truncation entry point."
            f"</small>",
            unsafe_allow_html=True,
        )
        try:
            profile = get_customer_profile(customer_id, transactions, CUTOFF_DATE)
            tenure = int(profile["tenure"].iloc[0])
            first_purchase_date = transactions[
                transactions["customer_id"] == customer_id
            ]["transaction_date"].min()

            # Unconditional S(t) for the full time grid
            surv_uncond = models["cph"].predict_survival_function(profile)
            t_grid = surv_uncond.index.values
            s_grid = surv_uncond.iloc[:, 0].values

            # Past segment: (0, 1.0) → ... → (tenure, S(tenure))
            past_t = np.concatenate([[0], t_grid[t_grid <= tenure], [tenure]])
            past_s = np.interp(past_t, t_grid, s_grid)

            # Future segment: continues on the same absolute scale — no normalisation
            future_t = np.concatenate([[tenure], t_grid[t_grid > tenure]])
            future_s = np.interp(future_t, t_grid, s_grid)

            fig2, ax = plt.subplots(figsize=(9, 4))

            # Past: gray dashed — already survived this portion
            ax.plot(past_t, past_s, color="#95a5a6", lw=1.5, linestyle="--",
                    label=f"Past (unconditional) — up to {CUTOFF_DATE.date()}")

            # Future: solid blue — forward projection on the same absolute scale
            ax.plot(future_t, future_s, color="#3498db", lw=2,
                    label="Future projection")

            # Cutoff vertical line
            ax.axvline(x=tenure, color="#e74c3c", linestyle=":", lw=1.5, alpha=0.8,
                       label=f"Cut-off: {CUTOFF_DATE.date()} (day {tenure})")

            # First purchase annotation — label sits in the left margin, arrow points into the chart
            ax.annotate(
                f"First purchase\n{first_purchase_date.date()}",
                xy=(0, 1.0),
                xytext=(0.07, 1.15),
                xycoords=("axes fraction", "data"),
                textcoords=("axes fraction", "data"),
                fontsize=8, color="#27ae60", ha="right", va="top",
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1,
                                connectionstyle="arc3,rad=0.0"),
            )

            ax.set_xlim(left=0)
            ax.set_ylim(0, 1.05)
            fig2.subplots_adjust(left=0.18)
            ax.set_xlabel("Days since first purchase")
            ax.set_ylabel("P(survival)")
            ax.set_title(f"Survival Curve — {customer_id}")
            ax.legend(fontsize=8)
            fig2.tight_layout()
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Could not render survival curve: {e}")

        # ── Section 2: slider-dependent (BG/NBD + NPV) ───────────────────────
        st.divider()
        st.markdown("### BG/NBD & Survival-weighted NPV")
        st.write("Adjust the horizon — all metrics and charts below update accordingly.")
        num_months = st.slider(
            "Horizon (months)",
            min_value=3, max_value=24, value=GG_CLV_HORIZON_MONTHS,
            key="npv_slider",
        )
        t_days = num_months * 30

        st.caption(
            f"**BG/NBD + Gamma-Gamma assumptions** — "
            f"Observation period end: **2025-12-31** (full-year data; BG/NBD and Gamma-Gamma retrained with all transactions). "
            f"Monthly discount rate: **{GG_DISCOUNT_RATE:.0%}** (passed to `gg.customer_lifetime_value`). "
            f"CLV is recomputed live for the selected horizon using the customer's fitted frequency, recency, T, and monetary value. "
            f"The first transaction is treated as the customer's entry point (defining T and recency) and is excluded from frequency — "
            f"the model only learns purchase and churn rates from repeat behaviour. "
            f"Customers with only one purchase (frequency = 0) are excluded from Gamma-Gamma fitting (no repeat spend to model) and their CLV is shown as **$0.00**."
        )

        # CLV BG/NBD recomputed live for the selected horizon
        try:
            row = models["lifetime_df"].loc[models["lifetime_df"].index == customer_id]
            if not row.empty and float(row["monetary_value"].iloc[0]) > 0:
                clv_bgnbd_live = float(
                    models["gg"].customer_lifetime_value(
                        models["bg"],
                        row["frequency"],
                        row["recency"],
                        row["T"],
                        row["monetary_value"],
                        time=num_months,
                        discount_rate=GG_DISCOUNT_RATE,
                        freq="D",
                    ).iloc[0]
                )
            else:
                clv_bgnbd_live = scores["clv_bgnbd"]
            st.metric(f"CLV — BG/NBD + Gamma-Gamma ({num_months}M)", f"${clv_bgnbd_live:,.2f}")
        except Exception as e:
            st.warning(f"Could not recompute BG/NBD CLV: {e}")

        st.markdown(f"### P(alive) History ({num_months} months)")
        try:
            from lifetimes import plotting as lt_plotting
            customer_txns = (
                transactions[transactions["customer_id"] == customer_id]
                .assign(transaction_date=lambda x: x["transaction_date"].astype("datetime64[ns]"))
            )
            fig_alive, ax_alive = plt.subplots(figsize=(9, 4))
            lt_plotting.plot_history_alive(
                models["bg"],
                t=t_days,
                transactions=customer_txns,
                datetime_col="transaction_date",
                ax=ax_alive,
            )
            plt.xticks(rotation=45, ha="right")
            ax_alive.set_title(f"P(alive) History — {customer_id} ({num_months} months)")

            # Annotate the second purchase (first repeat transaction = first red dashed line)
            sorted_dates = sorted(customer_txns["transaction_date"].unique())
            if len(sorted_dates) >= 2:
                second_purchase = pd.Timestamp(sorted_dates[1])
                ax_alive.annotate(
                    f"2nd purchase\n{second_purchase.date()}",
                    xy=(second_purchase, 1.0),
                    xytext=(second_purchase, 1.14),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    fontsize=8, color="#27ae60", ha="center", va="bottom",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1,
                                    connectionstyle="arc3,rad=0.0"),
                )
                fig_alive.subplots_adjust(top=0.82)

            st.pyplot(fig_alive)
        except Exception as e:
            st.warning(f"Could not render P(alive) chart: {e}")

        st.markdown(f"### Survival-weighted NPV ({num_months} months)")
        st.caption(
            f"**Survival-weighted NPV assumptions** — "
            f"Projection anchor: **2025-12-31** (same as BG/NBD + GG, enabling direct comparison). "
            f"Monthly discount rate: **{ANNUAL_IRR/12:.2%}** (= {ANNUAL_IRR:.0%} annual IRR ÷ 12). "
            f"Survival probability: CoxPH conditional survival S(t | alive at tenure), where tenure is days from first purchase to 2025-12-31. "
            f"Monthly profit: BG/NBD predicted purchase increment × Gamma-Gamma expected order value. "
            f"Known limitation: CoxPH was trained with features as of **{CUTOFF_DATE.date()}** — "
            f"the 90-day feature shift to Dec 31 introduces a small covariate bias in the survival estimates."
        )
        try:
            payback = get_payback_df(
                customer_id=customer_id,
                transactions=transactions,
                lifetime_df=models["lifetime_df"],
                bg=models["bg"],
                cph=models["cph"],
                num_months=num_months,
            )
            clv_survival = float(payback["Cumulative NPV"].iloc[-1])
            st.metric("CLV — Survival NPV", f"${clv_survival:,.2f}")
            st.dataframe(payback.style.format({
                "Survival Probability": "{:.1%}",
                "Monthly Profit": "${:.2f}",
                "Avg Expected Monthly Profit": "${:.2f}",
                "NPV of Avg Expected Monthly Profit": "${:.2f}",
                "Cumulative NPV": "${:.2f}",
            }), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute NPV table: {e}")


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

    tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "Gamma-Gamma", "SHAP Summary", "SHAP Force Plot"])

    with tab1:
        st.caption(
            f"Threshold for the **LightGBM churn classifier**. "
            f"Predicted probabilities above this value are labelled **churned** (positive class). "
            f"Default is the F1-optimal threshold found during training ({float(models['best_threshold']):.2f}). "
            f"Lower it → more customers flagged as churned: **FN decreases, FP increases** (higher recall, lower precision). "
            f"Raise it → only flag high-confidence churners: **FP decreases, FN increases** (higher precision, lower recall)."
        )
        threshold = st.slider(
            "Classification threshold",
            min_value=0.1, max_value=0.9,
            value=float(models["best_threshold"]), step=0.01,
        )
        fig = plot_confusion_matrix(models["lgbm"], X_test, y_test, threshold)
        st.pyplot(fig)

    with tab2:
        st.markdown("### Gamma-Gamma Amount Distribution")
        fig_kde = plot_gamma_kde(models["gg"])
        st.pyplot(fig_kde)

    with tab3:
        st.caption(
            f"**Model:** LightGBM binary classifier — "
            f"training cutoff: **{CUTOFF_DATE.date()}**, "
            f"label window: **{PREDICTION_WINDOW} days** (Oct 2 → Dec 31, 2025). "
            f"Hyperparameters: n_estimators={LGBM_PARAMS['n_estimators']}, "
            f"learning_rate={LGBM_PARAMS['learning_rate']}, "
            f"max_depth={LGBM_PARAMS['max_depth']}, "
            f"objective={LGBM_PARAMS['objective']}, "
            f"metric={LGBM_PARAMS['metric']}."
        )
        st.write("Feature importance via SHAP values (LightGBM).")
        import shap
        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(models["lgbm"])
            fig = plot_shap_summary(explainer, X_test)
        st.pyplot(fig)

    with tab4:
        # ── Build one representative customer per RFM segment from the test set ──
        analysis_date = pd.Timestamp("2026-01-01")  # day after max transaction date
        rfm = (
            churn_data.groupby("customer_id")
            .agg(
                recency=("recency", "first"),
                frequency=("frequency", "first"),
                monetary=("monetary", "first"),
            )
            .reset_index()
        )
        rfm["R_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
        rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm["segment"] = rfm.apply(_rfm_segment_rules, axis=1)

        # Keep only customers in the test set, preserving row position in X_test
        test_customers = test[["customer_id"]].reset_index(drop=True)
        test_customers["_row_idx"] = test_customers.index
        rfm_test = rfm.merge(test_customers, on="customer_id", how="inner")

        # One representative per segment (first match)
        rep_rows = (
            rfm_test.groupby("segment", sort=False)
            .first()
            .reset_index()[["segment", "customer_id", "recency", "frequency", "monetary",
                             "R_score", "F_score", "M_score", "_row_idx"]]
            .sort_values("segment")
        )

        st.markdown("#### Select a customer by RFM segment")
        st.caption("Click a row to load that customer's SHAP force plot below.")
        display_cols = ["customer_id", "recency", "frequency", "monetary",
                        "R_score", "F_score", "M_score", "segment"]
        selection = st.dataframe(
            rep_rows[display_cols],
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        selected_rows = selection.selection.rows
        table_row = selected_rows[0] if selected_rows else 0
        idx = int(rep_rows.iloc[table_row]["_row_idx"])

        import shap
        with st.spinner("Computing SHAP force plot..."):
            explainer = shap.TreeExplainer(models["lgbm"])
            shap_vals = explainer.shap_values(X_test)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
                base_val = float(explainer.expected_value[1])
            else:
                base_val = float(explainer.expected_value)
            logodds = base_val + float(shap_vals[idx].sum())
            prob = convert_logodds_to_probability(logodds)
            fig = plot_shap_force(explainer, X_test, idx)
        st.pyplot(fig)
        e_val = float(np.exp(-logodds))
        denom = 1 + e_val
        st.markdown(f"**log-odds = {logodds:.2f} → Churn Probability = {prob:.2%}**")
        st.latex(
            rf"p = \frac{{1}}{{1 + e^{{-(\text{{log-odds}})}}}} = "
            rf"\frac{{1}}{{1 + e^{{{-logodds:.2f}}}}} = "
            rf"\frac{{1}}{{1 + {e_val:.3f}}} = "
            rf"\frac{{1}}{{{denom:.3f}}} \approx {prob:.3f}"
        )



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

    models = load_models(_artifact_mtimes())
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
