# Customer Growth & Retention — Production System

End-to-end customer analytics platform combining churn classification (LightGBM),
BG/NBD + Gamma-Gamma CLV modelling, and CoxPH survival analysis.

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

## Dashboard Pages

| Page | Description |
|---|---|
| **Transactions** | Browse all transactions with filters by customer ID, date range, and amount |
| **RFM Analysis** | Customer segmentation grid (10 segments) with counts and avg. monetary value |
| **Customer Lookup** | Individual risk profile: churn probability, P(alive), survival curve, CLV/NPV |
| **Model Performance** | Confusion matrix, SHAP feature importance, Gamma-Gamma KDE, survival curves |
| **Retention Ranking** | Rank customers by churn risk or CLV for targeted retention campaigns |

---

## Model Cutoff Dates

| Model | Training cutoff | Notes |
|---|---|---|
| LightGBM churn | `2025-10-02` | Features Jan–Oct 2; labels Oct 2 – Dec 31 (90-day window) |
| BG/NBD + Gamma-Gamma | `2025-12-31` | Full year for most accurate alive probability and CLV |
| CoxPH survival | `2025-10-02` | Tenure and event detection require a future observation window |
