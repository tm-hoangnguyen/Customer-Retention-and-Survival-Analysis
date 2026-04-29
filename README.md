# Customer Growth & Retention — Production System

End-to-end customer analytics platform combining churn classification (LightGBM),
BG/NBD + Gamma-Gamma CLV modelling, and CoxPH survival analysis.

---

## Data

Raw inputs live under `data/` as CSV files loaded by `core/data.py`:

- **`transactions.csv`**: one purchase per row with `customer_id`, `transaction_date`, and `amount` (transaction value). Dates are parsed as timestamps; loaders assert no missing dates or amounts.
- **`customers.csv`**: one customer per row with `customer_id`, `signup_date`, and `true_lifetime_days` (ground-truth total lifetime in days).

Together these tables support RFM and transaction-history features, probabilistic churn, repeat-purchase BG/NBD and Gamma-Gamma valuation, and CoxPH survival modelling.

---

## Project Structure

```
survival_analysis/
├── data/                   # Raw CSVs (transactions.csv, customers.csv)
├── artifacts/              # Trained model files and pre-computed scores
├── core/
│   ├── config.py           # Centralised constants and artifact paths
│   ├── data.py             # Data loading utilities
│   ├── features.py         # Feature engineering (RFM, churn, survival)
│   ├── scoring.py          # Scoring functions (individual + batch)
│   └── charts.py           # Matplotlib chart helpers
├── api/
│   ├── main.py             # FastAPI app entry point
│   └── routers/            # Endpoints: score, churn, survival, clv, retention
├── app/
│   └── streamlit_app.py    # Streamlit dashboard
├── train.py                # One-time training script
├── explore_data.ipynb      # Reference notebook (read-only)
└── requirements.txt
```

---

## Quickstart

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train models

Trains all models (LightGBM, BG/NBD, Gamma-Gamma, CoxPH) and saves artifacts
to `artifacts/`, including pre-computed batch scores.

```bash
python train.py
```

### 4. Run the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens at [http://localhost:8501](http://localhost:8501)

### 5. Run the FastAPI (optional)

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Streamlit Dashboard Pages

| Page | Description |
|---|---|
| **Transactions** | Browse all transactions with filters by customer ID, date range, and amount |
| **RFM Analysis** | Recency × Frequency score grid (0–5 axes) with named segments, customer counts, % of base, and average monetary value — see figure below. |
| **Customer Lookup** | Individual risk profile: churn probability, P(alive), survival curve, CLV/NPV |
| **Model Performance** | Confusion matrix, SHAP feature importance, Gamma-Gamma KDE, survival curves |
| **Retention Ranking** | Rank customers by churn risk or CLV for targeted retention campaigns |

![RFM Customer Segmentation — Recency vs Frequency grid](docs/images/rfm_customer_segmentation.png)

---

## FastAPI

The app loads trained artifacts and `transactions.csv` once at startup (`api/main`). All scoring routes are **POST** with JSON bodies; OpenAPI request/response shapes live at [http://localhost:8000/docs](http://localhost:8000/docs) when the server is running.

| Route | Purpose |
|---|---|
| `POST /score_customer` | One call: churn probability/label, P(alive), BG/NBD CLV, Cox expected remaining lifetime, survival-weighted CLV (NPV); returns embedded scoring assumptions. |
| `POST /predict_churn` | LightGBM churn probability and risk band for a customer. |
| `POST /predict_survival` | CoxPH-based survival summary (e.g. expected remaining lifetime) at the training cutoff. |
| `POST /estimate_clv` | CLV by method: `bgnbd` (Gamma-Gamma + BG/NBD) or `survival` (Cox-weighted cashflow NPV via `get_payback_df`). |
| `POST /rank_customers_for_retention` | Ranks customers using pre-computed `artifacts/scores_df.pkl` (run `train.py` first); strategy `high_clv_high_churn` combines Cox dropout risk over a horizon with min–max normalized BG/NBD CLV. |
