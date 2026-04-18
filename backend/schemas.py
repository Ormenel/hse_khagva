
from pydantic import BaseModel, Field
from typing import List


class YieldCurveInput(BaseModel):
    tenors: List[float] = Field(
        default=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
        description="Maturities in years",
    )
    rates: List[float] = Field(
        default=[0.045, 0.046, 0.047, 0.048, 0.048, 0.047, 0.046, 0.045, 0.043, 0.042],
        description="Continuously-compounded zero rates",
    )


class LoanInput(BaseModel):
    coupon: float = Field(
        default=0.065,
        description="Annual coupon rate (decimal, e.g. 0.065 = 6.5%)",
        ge=0.001, le=0.30,
    )
    orig_term: int = Field(
        default=360,
        description="Original term in months",
        ge=12, le=600,
    )
    orig_balance: float = Field(
        default=300_000.0,
        description="Original UPB",
        ge=1_000,
    )
    loan_age: int = Field(
        default=0,
        description="Current loan age in months",
        ge=0,
    )
    current_balance: float | None = Field(
        default=None,
        description="Current UPB (defaults to orig_balance)",
    )


AVAILABLE_MODELS = ["xgb", "lgb", "rf", "lr", "sgd", "stacking", "calibrated"]


class LoanFeaturesInput(BaseModel):

    fico: float = Field(default=700.0, description="Borrower FICO score", ge=300, le=850)
    orig_ltv: float = Field(default=80.0, description="Original LTV (%)", ge=1, le=200)
    dti: float = Field(default=35.0, description="Debt-to-income ratio (%)", ge=0, le=100)
    channel: str = Field(default="R", description="Origination channel: R=Retail, B=Broker, C=Correspondent")
    loan_purpose: str = Field(default="P", description="P=Purchase, C=Cash-out, R=Refi, U=Unknown")
    property_type: str = Field(default="SF", description="SF, CO, PU, MH, CP")
    occupancy_status: str = Field(default="P", description="P=Primary, S=Second, I=Investor")
    property_state: str = Field(default="CA", description="US state code")
    origination_year: int = Field(default=2020, description="Year loan was originated")
    first_time_buyer: int = Field(default=0, description="1 if first-time buyer", ge=0, le=1)
    modified: int = Field(default=0, description="1 if loan was modified", ge=0, le=1)
    in_forbearance: int = Field(default=0, ge=0, le=1)
    has_deferral: int = Field(default=0, ge=0, le=1)
    has_ppm: int = Field(default=0, description="1 if prepayment penalty mortgage", ge=0, le=1)
    is_io: int = Field(default=0, description="1 if interest-only", ge=0, le=1)
    is_high_bal: int = Field(default=0, description="1 if high-balance conforming", ge=0, le=1)
    hltv_refi_option: str = Field(default="N", description="Y/N for high-LTV refi option")
    ph_delinq_count: int = Field(default=0, description="Prior delinquency count", ge=0)
    excess_principal: float = Field(default=0.0, description="Excess principal payments ($)")


class OASRequest(BaseModel):

    loan: LoanInput = Field(default_factory=LoanInput)
    loan_features: LoanFeaturesInput = Field(default_factory=LoanFeaturesInput)
    yield_curve: YieldCurveInput = Field(default_factory=YieldCurveInput)
    model_name: str = Field(
        default="xgb",
        description=f"Prepayment model to use. One of: {AVAILABLE_MODELS}",
    )
    market_price: float = Field(
        default=100.0,
        description="Clean market price per $100 of face",
        ge=50.0, le=150.0,
    )
    n_paths: int = Field(
        default=500,
        description="Number of Monte Carlo simulation paths",
        ge=50, le=10_000,
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")


class OASResponse(BaseModel):

    oas_bps: float = Field(description="Option-Adjusted Spread in basis points")
    model_price: float = Field(description="Model-implied price at solved OAS")
    market_price: float = Field(description="Target market price")
    avg_life: float = Field(description="Weighted-average life in years")
    avg_smm: float = Field(description="Average SMM across all paths")
    avg_cpr: float = Field(description="Annualised CPR = 1 - (1-SMM)^12")
    n_paths: int = Field(description="Monte Carlo paths used")
    converged: bool = Field(description="Solver converged within tolerance")
    path_times: List[float] = Field(
        default_factory=list,
        description="Time grid (years) for sampled rate paths",
    )
    rate_paths: List[List[float]] = Field(
        default_factory=list,
        description="Subset of simulated Hull-White short-rate paths (decimal).",
    )
    rate_mean: List[float] = Field(
        default_factory=list,
        description="Mean short rate across ALL simulated paths, per month (decimal).",
    )
    rate_p05: List[float] = Field(
        default_factory=list,
        description="5th-percentile short rate across ALL paths, per month.",
    )
    rate_p95: List[float] = Field(
        default_factory=list,
        description="95th-percentile short rate across ALL paths, per month.",
    )
