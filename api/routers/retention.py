from fastapi import APIRouter, HTTPException, Request

from api.schemas import RankCustomersRequest, RankCustomersResponse, RankedCustomer
from core.scoring import rank_customers

router = APIRouter()


@router.post("/rank_customers_for_retention", response_model=RankCustomersResponse)
async def rank_customers_endpoint(body: RankCustomersRequest, request: Request):
    state = request.app.state

    if state.scores_df is None:
        raise HTTPException(
            status_code=503,
            detail="Batch scores not yet computed. Check server startup logs.",
        )

    try:
        ranked = rank_customers(state.scores_df, body.strategy, body.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    customers = [
        RankedCustomer(
            customer_id=row["customer_id"],
            churn_probability=row["churn_probability"],
            clv=row["clv_bgnbd"],
            priority_score=row["priority_score"],
        )
        for _, row in ranked.iterrows()
    ]
    return RankCustomersResponse(
        strategy=body.strategy,
        top_k=body.top_k,
        customers=customers,
    )
