from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    CLVBgnbdAssumptions,
    CLVSurvivalAssumptions,
    EstimateCLVAssumptions,
    EstimateCLVRequest,
    EstimateCLVResponse,
)
from core.config import ANNUAL_IRR, GG_DISCOUNT_RATE
from core.scoring import bgnbd_gg_horizon_floor, get_payback_df, score_bgnbd

router = APIRouter()


@router.post("/estimate_clv", response_model=EstimateCLVResponse)
async def estimate_clv_endpoint(body: EstimateCLVRequest, request: Request):
    state = request.app.state
    obs_end = str(state.transactions["transaction_date"].max().date())
    try:
        if body.method == "bgnbd":
            gg_horizon_eff = bgnbd_gg_horizon_floor(body.horizon_months)
            result = score_bgnbd(
                customer_id=body.customer_id,
                lifetime_df=state.lifetime_df,
                bg=state.bg,
                gg=state.gg,
                horizon_months=body.horizon_months,
            )
            clv = result["clv_bgnbd"]
            assumptions = EstimateCLVAssumptions(
                method=body.method,
                horizon_months_applied=body.horizon_months,
                bgnbd=CLVBgnbdAssumptions(
                    observation_period_end=obs_end,
                    gg_clv_horizon_months=gg_horizon_eff,
                    monthly_discount_rate_gg=GG_DISCOUNT_RATE,
                ),
                survival=None,
            )
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
            assumptions = EstimateCLVAssumptions(
                method=body.method,
                horizon_months_applied=body.horizon_months,
                bgnbd=None,
                survival=CLVSurvivalAssumptions(
                    projection_anchor_date=obs_end,
                    annual_irr=ANNUAL_IRR,
                ),
            )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return EstimateCLVResponse(
        customer_id=body.customer_id,
        method=body.method,
        clv=round(clv, 2),
        horizon_months=body.horizon_months,
        assumptions=assumptions,
    )
