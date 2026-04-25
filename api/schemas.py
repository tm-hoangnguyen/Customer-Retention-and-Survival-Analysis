"""Pydantic request / response schemas for all API endpoints."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScoreCustomerRequest(BaseModel):
    customer_id: str = Field(..., example="C01329")


class ScoreCustomerResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_label: str
    p_alive: float
    clv_bgnbd: float
    expected_remaining_lifetime: float
    clv_survival: float


class PredictChurnRequest(BaseModel):
    customer_id: str = Field(..., example="C01329")
    horizon_days: int = Field(90, ge=1, le=365, example=90)


class PredictChurnResponse(BaseModel):
    customer_id: str
    horizon_days: int
    churn_probability: float
    churn_label: str
    note: str | None = None


class PredictSurvivalRequest(BaseModel):
    customer_id: str = Field(..., example="C01329")


class SurvivalPoint(BaseModel):
    day: int
    prob: float


class PredictSurvivalResponse(BaseModel):
    customer_id: str
    survival_curve: list[SurvivalPoint]
    expected_remaining_lifetime: float


class EstimateCLVRequest(BaseModel):
    customer_id: str = Field(..., example="C01329")
    method: Literal["bgnbd", "survival"] = Field("bgnbd", example="bgnbd")
    horizon_months: int = Field(12, ge=1, le=60, example=12)


class EstimateCLVResponse(BaseModel):
    customer_id: str
    method: str
    clv: float
    horizon_months: int


class RankCustomersRequest(BaseModel):
    top_k: int = Field(100, ge=1, le=5000, example=100)
    strategy: Literal[
        "high_churn_probability",
        "low_palive",
        "high_clv_high_churn",
    ] = Field("high_clv_high_churn", example="high_clv_high_churn")


class RankedCustomer(BaseModel):
    customer_id: str
    churn_probability: float
    clv: float
    priority_score: float


class RankCustomersResponse(BaseModel):
    strategy: str
    top_k: int
    customers: list[RankedCustomer]
