from fastapi import APIRouter, HTTPException, Request
import pandas as pd

from api.schemas import (
    RankCustomersRequest,
    RankCustomersResponse,
    RankedCustomer,
    RetentionRankingAssumptions,
)
from core.config import CUTOFF_DATE
from core.scoring import rank_customers

router = APIRouter()


def _retention_assumptions(
    request: Request,
    cox_horizon_days: int | None = None,
) -> RetentionRankingAssumptions:
    state = request.app.state
    obs_end = str(state.transactions["transaction_date"].max().date())
    return RetentionRankingAssumptions(
        churn_cutoff_date=str(CUTOFF_DATE.date()),
        p_alive_observation_period_end=obs_end,
        cph_training_cutoff_date=str(CUTOFF_DATE.date()),
        cph_scoring_reference=obs_end,
        batch_artifact_source="artifacts/scores_df.pkl",
        cox_horizon_days=cox_horizon_days,
    )


@router.post("/rank_customers_for_retention", response_model=RankCustomersResponse)
async def rank_customers_endpoint(body: RankCustomersRequest, request: Request):
    state = request.app.state

    if state.scores_df is None:
        raise HTTPException(
            status_code=503,
            detail="Batch scores not yet computed. Check server startup logs.",
        )

    try:
        ranked = rank_customers(
            state.scores_df,
            body.strategy,
            body.top_k,
            cph=state.cph,
            horizon_days=body.horizon_days,
            transactions=state.transactions,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    churn_ref = state.scores_df[["customer_id", "churn_probability"]].drop_duplicates(
        subset=["customer_id"],
    )
    ranked = ranked.merge(churn_ref, on="customer_id", how="left")

    has_cdr = "cox_dropout_risk" in ranked.columns
    customers = [
        RankedCustomer(
            customer_id=row["customer_id"],
            churn_probability=float(row["churn_probability"]),
            clv=float(row["clv_bgnbd"]),
            priority_score=float(row["priority_score"]),
            cox_dropout_risk=(
                float(row["cox_dropout_risk"])
                if has_cdr and pd.notna(row["cox_dropout_risk"])
                else None
            ),
        )
        for _, row in ranked.iterrows()
    ]
    hz = body.horizon_days
    return RankCustomersResponse(
        strategy=body.strategy,
        top_k=body.top_k,
        customers=customers,
        assumptions=_retention_assumptions(request, cox_horizon_days=hz),
    )
