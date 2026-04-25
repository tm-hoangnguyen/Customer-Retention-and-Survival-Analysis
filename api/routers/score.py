from fastapi import APIRouter, HTTPException, Request

from api.schemas import ScoreCustomerRequest, ScoreCustomerResponse
from core.scoring import score_customer

router = APIRouter()


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
    return result
