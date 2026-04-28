from fastapi import APIRouter, HTTPException, Request

from api.schemas import PredictSurvivalRequest, PredictSurvivalResponse, SurvivalAssumptions, SurvivalPoint
from core.config import CUTOFF_DATE
from core.scoring import score_survival

router = APIRouter()


_SURVIVAL_ASSUMPTIONS = SurvivalAssumptions(
    training_cutoff_date=str(CUTOFF_DATE.date()),
    scoring_cutoff_date=str(CUTOFF_DATE.date()),
    conditional_survival_horizon_months=12,
)


@router.post("/predict_survival", response_model=PredictSurvivalResponse)
async def predict_survival_endpoint(body: PredictSurvivalRequest, request: Request):
    state = request.app.state
    try:
        result = score_survival(
            customer_id=body.customer_id,
            transactions=state.transactions,
            cph=state.cph,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictSurvivalResponse(
        customer_id=body.customer_id,
        survival_curve=[SurvivalPoint(**pt) for pt in result["survival_curve"]],
        expected_remaining_lifetime=result["expected_remaining_lifetime"],
        assumptions=_SURVIVAL_ASSUMPTIONS,
    )
