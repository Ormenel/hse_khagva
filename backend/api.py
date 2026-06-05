import logging
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    OASRequest, OASResponse,
    LoanInput, LoanFeaturesInput, YieldCurveInput,
    AVAILABLE_MODELS,
)
from .hull_white import HullWhiteParams, YieldCurve
from .cashflow_engine import LoanParams
from .oas_calculator import compute_oas_ml
from .treasury_curve import fetch_latest_par_curve

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Нужно добавить HUll White model
HW_A = 0.1273
HW_SIGMA = 0.04
#HW_SIGMA = 0.00836

MODEL_DIR = os.getenv("MODEL_DIR", "/app/models/saved_models")

app = FastAPI(
    title="OAS Spread Calculator",
    description=(
        "Compute the Option-Adjusted Spread for fixed-rate mortgages"
    )
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_inference_registry: dict = {}


@app.on_event("startup")
def _load_models():
    # loading models
    from models.inference_pipeline import PrepaymentModelInference

    for name in AVAILABLE_MODELS:
        try:
            _inference_registry[name] = PrepaymentModelInference(
                model_dir=MODEL_DIR, model_name=name)
            log.info("Loaded ML model '%s' from %s", name, MODEL_DIR)
        except Exception:
            log.exception("Could not load model '%s'", name)

    if not _inference_registry:
        log.error("No ML models loaded from %s — /oas/compute. Check dir",
                  MODEL_DIR)


# ENDPOINTS

@app.get("/oas/health")
def health():
    return {"status": "ok", "loaded_models": sorted(_inference_registry.keys())}


@app.get("/oas/models")
def models():
    return {
        "available": AVAILABLE_MODELS,
        "loaded": sorted(_inference_registry.keys()),
        "default": "xgb",
    }


@app.get("/oas/defaults")
def defaults():
    return {
        "loan": LoanInput().model_dump(),
        "loan_features": LoanFeaturesInput().model_dump(),
        "yield_curve": YieldCurveInput().model_dump(),
        "model_name": "xgb",
        "market_price": 100.0,
        "n_paths": 500,
        "seed": 42,
    }


@app.get("/oas/par_curve/latest")
def par_curve_latest():
    try:
        curve = fetch_latest_par_curve()
    except Exception as exc:
        log.warning("Treasury par curve unavailable: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Treasury par curve unavailable: {exc}",
        )
    return {
        "date": curve.date,
        "tenors": curve.tenors,
        "rates": curve.rates,
        "source": "home.treasury.gov",
    }


@app.post("/oas/compute", response_model=OASResponse)
def compute(req: OASRequest):

    try:
        inference = _inference_registry.get(req.model_name)
        if inference is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{req.model_name}' is not loaded. "
                    f"Available: {sorted(_inference_registry.keys())}"
                ),
            )

        loan = LoanParams(
            coupon=req.loan.coupon,
            orig_term=req.loan.orig_term,
            orig_balance=req.loan.orig_balance,
            loan_age=req.loan.loan_age,
            current_balance=(req.loan.current_balance
                             if req.loan.current_balance
                             else req.loan.orig_balance),
            )

        hw = HullWhiteParams(a=HW_A, sigma=HW_SIGMA)

        # Input rates are PAR yields bootstrapping is inside
        par_curve = YieldCurve(
            tenors=np.array(req.yield_curve.tenors),
            rates=np.array(req.yield_curve.rates),
        )

        lf = req.loan_features
        loan_ml_params = {
            "fico": lf.fico,
            "orig_interest_rate": loan.coupon,
            "current_interest_rate": loan.coupon,
            "orig_upb": loan.orig_balance,
            "current_upb": loan.current_balance,
            "loan_age": loan.loan_age,
            "orig_loan_term": loan.orig_term,
            "orig_ltv": lf.orig_ltv,
            "dti": lf.dti,
            "channel": lf.channel,
            "loan_purpose": lf.loan_purpose,
            "property_type": lf.property_type,
            "occupancy_status": lf.occupancy_status,
            "property_state": lf.property_state,
            "origination_year": lf.origination_year,
            "first_time_buyer": lf.first_time_buyer,
            "modified": lf.modified,
            "in_forbearance": lf.in_forbearance,
            "has_deferral": lf.has_deferral,
            "has_ppm": lf.has_ppm,
            "is_io": lf.is_io,
            "is_high_bal": lf.is_high_bal,
            "hltv_refi_option": lf.hltv_refi_option,
            "ph_delinq_count": lf.ph_delinq_count,
            "excess_principal": lf.excess_principal,
        }

        result = compute_oas_ml(
            loan=loan, inference=inference,
            loan_ml_params=loan_ml_params,
            par_curve=par_curve, hw_params=hw,
            market_price=req.market_price,
            n_paths=req.n_paths, seed=req.seed,
        )
        log.info("OAS (%s) computed: %.1f bp  (price=%.2f, WAL=%.1fy, CPR=%.2f%%)",
                 req.model_name, result.oas_bps, result.model_price,
                 result.avg_life, result.avg_cpr * 100)

        return OASResponse(
            oas_bps=result.oas_bps,
            oas_expected_bps=result.oas_expected_bps,
            oas_unexpected_bps=result.oas_unexpected_bps,
            model_price=result.model_price,
            market_price=result.market_price,
            avg_life=result.avg_life,
            avg_smm=result.avg_smm,
            avg_cpr=result.avg_cpr,
            n_paths=result.n_paths,
            converged=result.converged,
            path_times=result.path_times or [],
            rate_paths=result.rate_paths or [],
            rate_mean=result.rate_mean or [],
            rate_p05=result.rate_p05 or [],
            rate_p95=result.rate_p95 or [],
            cpr_months=result.cpr_months or [],
            cpr_curve_monthly=result.cpr_curve_monthly or [],
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("OAS computation failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
