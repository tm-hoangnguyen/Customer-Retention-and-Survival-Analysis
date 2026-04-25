"""
FastAPI application entry point.

Startup:
    uvicorn api.main:app --reload --port 8000

All models and supporting dataframes are loaded once at startup via the
lifespan context manager and attached to app.state so routers can access them.
Batch scores are loaded from artifacts/scores_df.pkl (pre-computed by train.py)
so startup is fast.
"""
from __future__ import annotations

import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import dill
from fastapi import FastAPI

from api.routers import churn, clv, retention, score, survival
from core.config import (
    BEST_THRESHOLD_PATH,
    BG_PATH,
    CHURN_DATA_PATH,
    CPH_PATH,
    GG_PATH,
    LGBM_PATH,
    LIFETIME_DF_PATH,
    SCORES_DF_PATH,
    SURVIVAL_DF_PATH,
)
from core.data import load_transactions


def _load(path: Path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return dill.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models and data…")
    app.state.lgbm = _load(LGBM_PATH)
    app.state.bg = _load(BG_PATH)
    app.state.gg = _load(GG_PATH)
    app.state.cph = _load(CPH_PATH)
    app.state.best_threshold = _load(BEST_THRESHOLD_PATH)
    app.state.lifetime_df = _load(LIFETIME_DF_PATH)
    app.state.survival_df = _load(SURVIVAL_DF_PATH)
    app.state.churn_data = _load(CHURN_DATA_PATH)
    app.state.transactions = load_transactions()

    if SCORES_DF_PATH.exists():
        print("Loading pre-computed batch scores…")
        app.state.scores_df = _load(SCORES_DF_PATH)
        print(f"  Loaded {len(app.state.scores_df):,} customer scores.")
    else:
        print("scores_df.pkl not found — run train.py to generate it.")
        app.state.scores_df = None

    print("API ready.")
    yield


app = FastAPI(
    title="Customer Growth & Retention API",
    description=(
        "Production scoring API for churn prediction, BG/NBD, "
        "survival analysis, and CLV estimation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(score.router, tags=["Unified Scoring"])
app.include_router(churn.router, tags=["Churn Classification"])
app.include_router(survival.router, tags=["Survival Analysis"])
app.include_router(clv.router, tags=["CLV Estimation"])
app.include_router(retention.router, tags=["Retention Ranking"])


@app.get("/health")
async def health():
    return {"status": "ok"}
