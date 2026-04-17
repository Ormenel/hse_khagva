import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    OASRequest, OASResponse,
    LoanInput, HullWhiteInput, PrepaymentInput, YieldCurveInput,
)
from .hull_white import HullWhiteParams, YieldCurve
from .cashflow_engine import LoanParams, PrepaymentParams
from .oas_calculator import compute_oas, compute_oas_ml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="OAS Spread Calculator",
    description=(
        "Compute the Option-Adjusted Spread for fixed-rate mortgages "
        "using Hull-White 1-factor Monte Carlo simulation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints

@app.get("/oas/health")
def health():
    return {"status": "ok"}


@app.get("/oas/defaults")
def defaults():
    return {
        "loan": LoanInput().model_dump(),
        "hull_white": HullWhiteInput().model_dump(),
        "prepayment": PrepaymentInput().model_dump(),
        "yield_curve": YieldCurveInput().model_dump(),
        "market_price": 100.0,
        "n_paths": 500,
        "seed": 42,
    }


@app.post("/oas/compute", response_model=OASResponse)
def compute(req: OASRequest):
    try:
        # Build domain objects from Pydantic schemas
        loan = LoanParams(
            coupon=req.loan.coupon,
            orig_term=req.loan.orig_term,
            orig_balance=req.loan.orig_balance,
            loan_age=req.loan.loan_age,
            current_balance=(req.loan.current_balance
                             if req.loan.current_balance
                             else req.loan.orig_balance),
        )

        hw = HullWhiteParams(a=req.hull_white.a, sigma=req.hull_white.sigma)

        curve = YieldCurve(
            tenors=np.array(req.yield_curve.tenors),
            rates=np.array(req.yield_curve.rates),
        )

        # Choose PSA or ML prepayment model
        if req.ml_prepayment is not None:
            # ML mode: load sklearn model and use ML-predicted SMM
            from models.inference_pipeline_v3 import PrepaymentModelInference

            ml = req.ml_prepayment
            inference = PrepaymentModelInference(
                model_dir=ml.model_dir, model_name=ml.model_name)

            loan_ml_params = {
                "fico": ml.fico,
                "orig_interest_rate": loan.coupon,
                "orig_upb": loan.orig_balance,
                "current_upb": loan.current_balance,
                "loan_age": loan.loan_age,
                "orig_loan_term": loan.orig_term,
                "orig_ltv": ml.orig_ltv,
                "dti": ml.dti,
                "channel": ml.channel,
                "loan_purpose": ml.loan_purpose,
                "property_type": ml.property_type,
                "occupancy_status": ml.occupancy_status,
                "property_state": ml.property_state,
                "origination_year": ml.origination_year,
                "first_time_buyer": ml.first_time_buyer,
                "modified": ml.modified,
                "in_forbearance": ml.in_forbearance,
                "has_deferral": ml.has_deferral,
                "has_ppm": ml.has_ppm,
                "is_io": ml.is_io,
                "is_high_bal": ml.is_high_bal,
                "hltv_refi_option": ml.hltv_refi_option,
                "ph_delinq_count": ml.ph_delinq_count,
                "excess_principal": ml.excess_principal,
            }

            result = compute_oas_ml(
                loan=loan, inference=inference,
                loan_ml_params=loan_ml_params,
                curve=curve, hw_params=hw,
                market_price=req.market_price,
                n_paths=req.n_paths, seed=req.seed,
            )
            log.info("OAS-ML computed: %.1f bp  (price=%.2f, WAL=%.1fy, CPR=%.2f%%)",
                     result.oas_bps, result.model_price,
                     result.avg_life, result.avg_cpr * 100)
        else:
            # PSA mode (original behaviour)
            prepay = PrepaymentParams(
                psa_speed=req.prepayment.psa_speed,
                refi_sensitivity=req.prepayment.refi_sensitivity,
                risk_premium=req.prepayment.risk_premium,
                use_rate_model=req.prepayment.use_rate_model,
            )

            result = compute_oas(
                loan=loan, prepay=prepay, curve=curve,
                hw_params=hw, market_price=req.market_price,
                n_paths=req.n_paths, seed=req.seed,
            )
            log.info("OAS-PSA computed: %.1f bp  (price=%.2f, WAL=%.1fy, CPR=%.2f%%)",
                     result.oas_bps, result.model_price,
                     result.avg_life, result.avg_cpr * 100)

        return OASResponse(
            oas_bps=result.oas_bps,
            model_price=result.model_price,
            market_price=result.market_price,
            avg_life=result.avg_life,
            avg_smm=result.avg_smm,
            avg_cpr=result.avg_cpr,
            n_paths=result.n_paths,
            converged=result.converged,
        )

    except Exception as e:
        log.exception("OAS computation failed")
        raise HTTPException(status_code=500, detail=str(e))


# Direct execution

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("oas_service.api:app", host="0.0.0.0", port=8000, reload=True)
