from fastapi import APIRouter, HTTPException, Request

from api.schemas import PredictChurnRequest, PredictChurnResponse
from core.config import PREDICTION_WINDOW
from core.scoring import score_churn

router = APIRouter()


@router.post("/predict_churn", response_model=PredictChurnResponse)
async def predict_churn_endpoint(body: PredictChurnRequest, request: Request):
    state = request.app.state
    note = None
    if body.horizon_days != PREDICTION_WINDOW:
        note = (
            f"Model was trained on a {PREDICTION_WINDOW}-day churn window. "
            f"Prediction for horizon_days={body.horizon_days} is approximate."
        )
    try:
        result = score_churn(
            customer_id=body.customer_id,
            transactions=state.transactions,
            lgbm=state.lgbm,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictChurnResponse(
        customer_id=body.customer_id,
        horizon_days=body.horizon_days,
        churn_probability=result["churn_probability"],
        churn_label=result["churn_label"],
        note=note,
    )
