from fastapi import APIRouter, HTTPException, Request

from api.schemas import EstimateCLVRequest, EstimateCLVResponse
from core.scoring import get_payback_df, score_bgnbd

router = APIRouter()


@router.post("/estimate_clv", response_model=EstimateCLVResponse)
async def estimate_clv_endpoint(body: EstimateCLVRequest, request: Request):
    state = request.app.state
    try:
        if body.method == "bgnbd":
            result = score_bgnbd(
                customer_id=body.customer_id,
                lifetime_df=state.lifetime_df,
                bg=state.bg,
                gg=state.gg,
            )
            clv = result["clv_bgnbd"]
        else:
            payback = get_payback_df(
                customer_id=body.customer_id,
                transactions=state.transactions,
                lifetime_df=state.lifetime_df,
                bg=state.bg,
                cph=state.cph,
                num_months=body.horizon_months,
            )
            clv = float(payback["Cumulative NPV"].iloc[-1])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return EstimateCLVResponse(
        customer_id=body.customer_id,
        method=body.method,
        clv=round(clv, 2),
        horizon_months=body.horizon_months,
    )
