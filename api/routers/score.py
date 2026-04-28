from fastapi import APIRouter, HTTPException, Request

from api.schemas import ScoreCustomerRequest, ScoreCustomerResponse, ScoringAssumptions
from core.config import (
    ANNUAL_IRR,
    CLV_HORIZON_MONTHS,
    CUTOFF_DATE,
    GG_CLV_HORIZON_MONTHS,
    GG_DISCOUNT_RATE,
    PREDICTION_WINDOW,
)
from core.scoring import score_customer

router = APIRouter()


def _scoring_assumptions(request: Request) -> ScoringAssumptions:
    obs_end = str(request.app.state.transactions["transaction_date"].max().date())
    return ScoringAssumptions(
        churn_cutoff_date=str(CUTOFF_DATE.date()),
        churn_prediction_window_days=PREDICTION_WINDOW,
        p_alive_observation_period_end=obs_end,
        clv_bgnbd_horizon_months=GG_CLV_HORIZON_MONTHS,
        clv_bgnbd_monthly_discount_rate=GG_DISCOUNT_RATE,
        clv_survival_horizon_months=CLV_HORIZON_MONTHS,
        clv_survival_annual_irr=ANNUAL_IRR,
        cph_training_cutoff_date=str(CUTOFF_DATE.date()),
        cph_scoring_cutoff_date=obs_end,
    )


@router.post("/score_customer", response_model=ScoreCustomerResponse)
async def score_customer_endpoint(body: ScoreCustomerRequest, request: Request):
    state = request.app.state
    try:
        result = score_customer(
            customer_id=body.customer_id,
            transactions=state.transactions,
            lifetime_df=state.lifetime_df,
            lgbm=state.lgbm,
            bg=state.bg,
            gg=state.gg,
            cph=state.cph,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {**result, "assumptions": _scoring_assumptions(request)}
