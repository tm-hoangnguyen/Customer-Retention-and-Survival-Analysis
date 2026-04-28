"""Pydantic request / response schemas for all API endpoints."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScoreCustomerRequest(BaseModel):
    customer_id: str = Field(..., example="C01329")


class ScoringAssumptions(BaseModel):
    churn_cutoff_date: str
    churn_prediction_window_days: int
    p_alive_observation_period_end: str
    clv_bgnbd_horizon_months: int
    clv_bgnbd_monthly_discount_rate: float
    clv_survival_horizon_months: int
    clv_survival_annual_irr: float
    cph_training_cutoff_date: str
    cph_scoring_cutoff_date: str


class ScoreCustomerResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_label: str
    p_alive: float
    clv_bgnbd: float
    expected_remaining_lifetime: float
    clv_survival: float
    assumptions: ScoringAssumptions


class PredictChurnRequest(BaseModel):
    customer_id: str = Field(..., example="C01329")
    horizon_days: int = Field(90, ge=1, le=365, example=90)


class ChurnAssumptions(BaseModel):
    churn_cutoff_date: str
    training_prediction_window_days: int = Field(
        ...,
        description="Days defining the churn label window after the training cutoff.",
    )


class PredictChurnResponse(BaseModel):
    customer_id: str
    horizon_days: int
    churn_probability: float
    churn_label: str
    note: str | None = None
    assumptions: ChurnAssumptions


class PredictSurvivalRequest(BaseModel):
    customer_id: str = Field(..., example="C01329")


class SurvivalPoint(BaseModel):
    day: int
    prob: float


class SurvivalAssumptions(BaseModel):
    training_cutoff_date: str = Field(
        ...,
        description="CoxPH trained with tenure as of this date (left truncation).",
    )
    scoring_cutoff_date: str = Field(
        ...,
        description="Tenure and covariates computed as of this date for scoring.",
    )
    conditional_survival_horizon_months: int = Field(
        ...,
        description="Number of 30-day steps in survival_curve (12 points).",
    )


class PredictSurvivalResponse(BaseModel):
    customer_id: str
    survival_curve: list[SurvivalPoint]
    expected_remaining_lifetime: float
    assumptions: SurvivalAssumptions


class EstimateCLVRequest(BaseModel):
    customer_id: str = Field(..., example="C01329")
    method: Literal["bgnbd", "survival"] = Field("bgnbd", example="bgnbd")
    horizon_months: int = Field(12, ge=1, le=60, example=12)


class CLVBgnbdAssumptions(BaseModel):
    observation_period_end: str = Field(
        ...,
        description="BG/NBD + Gamma-Gamma lifetime_df anchor.",
    )
    gg_clv_horizon_months: int = Field(
        ...,
        description=(
            "Effective GG horizon (months): max(request horizon, 3), passed as time= "
            "to gg.customer_lifetime_value (same as Streamlit live CLV)."
        ),
    )
    monthly_discount_rate_gg: float


class CLVSurvivalAssumptions(BaseModel):
    projection_anchor_date: str = Field(
        ...,
        description="max(transaction_date): tenure and NPV horizon start here.",
    )
    annual_irr: float = Field(
        ...,
        description="NPV discount: annual_irr / 12 applied per contract month.",
    )


class EstimateCLVAssumptions(BaseModel):
    """Unified block: method-specific fields filled in by the endpoint."""

    method: str
    horizon_months_applied: int = Field(
        ...,
        description=(
            "For survival: NPV table length; for bgnbd: requested horizon echoed "
            "(GG uses max(request, 3 months) — see gg_clv_horizon_months)."
        ),
    )
    bgnbd: CLVBgnbdAssumptions | None = None
    survival: CLVSurvivalAssumptions | None = None


class EstimateCLVResponse(BaseModel):
    customer_id: str
    method: str
    clv: float
    horizon_months: int
    assumptions: EstimateCLVAssumptions


class RetentionRankingAssumptions(BaseModel):
    churn_cutoff_date: str
    p_alive_observation_period_end: str
    cph_training_cutoff_date: str
    cph_scoring_reference: str = Field(
        ...,
        description="Batch survival/CLV aligned to max(transaction_date).",
    )
    batch_artifact_source: str = Field(
        ...,
        description="Pre-computed artifact file from train.py.",
    )


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
    assumptions: RetentionRankingAssumptions
