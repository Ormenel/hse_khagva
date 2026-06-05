
from pydantic import BaseModel, Field
from typing import List


class YieldCurveInput(BaseModel):
    tenors: List[float] = Field(
        default=[0.5, 1, 2, 3, 5, 7, 10, 20, 30],
        description="Maturities in years",
    )
    rates: List[float] = Field(
        default=[0.046, 0.047, 0.048, 0.048, 0.047, 0.046, 0.045, 0.043, 0.042],
        description="Par yields on US Treasury",
    )


class LoanInput(BaseModel):
    coupon: float = Field(
        default=0.065,
        description="Annual coupon/loan rate",
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
        description="Current UPB",
    )


AVAILABLE_MODELS = ["xgb", "lgb", "rf", "lr", "sgd", "stacking", "calibrated"]


class LoanFeaturesInput(BaseModel):

    fico: float = Field(default=700.0, description="FICO score", ge=300, le=850)
    orig_ltv: float = Field(default=80.0, description="Orig LTV", ge=1, le=200)
    dti: float = Field(default=35.0, description="Debt-to-income ratio", ge=0, le=100)
    channel: str = Field(default="R", description="Origination channel")
    loan_purpose: str = Field(default="P", description="P=Purchase, C=Cash-out, R=Refi, U=Unknown")
    property_type: str = Field(default="SF", description="SF=Single-Family, CO=Condominium, PU=Urban Dev, MH= Manufactured, CP=Co-operative")
    occupancy_status: str = Field(default="P", description="P=Primary, S=Second, I=Investor")
    property_state: str = Field(default="CA", description="US state code")
    origination_year: int = Field(default=2020, description="Origination Year")
    first_time_buyer: int = Field(default=0, description="First-time buyer", ge=0, le=1)
    modified: int = Field(default=0, description="Loan was modified", ge=0, le=1)
    in_forbearance: int = Field(default=0, description="Client in forbearance", ge=0, le=1)
    has_deferral: int = Field(default=0, description="Client has deferral", ge=0, le=1)
    has_ppm: int = Field(default=0, description="Has prepayment penalty", ge=0, le=1)
    is_io: int = Field(default=0, description="Interest only", ge=0, le=1)
    is_high_bal: int = Field(default=0, description=" IS High balance", ge=0, le=1)
    hltv_refi_option: str = Field(default="N", description="High LTV refi")
    ph_delinq_count: int = Field(default=0, description="Prior delinquency count", ge=0)
    excess_principal: float = Field(default=0.0, description="Excess principal payments")


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
        description="Clean market price",
        ge=50.0, le=150.0,
    )
    n_paths: int = Field(
        default=500,
        description="Number of Monte Carlo simulation paths",
        ge=50, le=10_000,
    )
    seed: int = Field(default=42, description="Random seed")


class OASResponse(BaseModel):

    oas_bps: float = Field(description="Option-Adjusted Spread (OAS) in bp")
    oas_expected_bps: float = Field(description="Expected OAS component in bp")
    oas_unexpected_bps: float = Field(description="Unexpected OAS component in bp")
    model_price: float = Field(description="Model-implied price at solved OAS")
    market_price: float = Field(description="Target market price")
    avg_life: float = Field(description="Weighted average life in years")
    avg_smm: float = Field(description="Weighted average SMM")
    avg_cpr: float = Field(description="Weighted average CPR")
    n_paths: int = Field(description="Monte Carlo paths used")
    converged: bool = Field(description="Converged")
    path_times: List[float] = Field(default_factory=list, description="Time grid")
    rate_paths: List[List[float]] = Field(
        default_factory=list,
        description="Rate paths for chart",)
    rate_mean: List[float] = Field(
        default_factory=list,
        description="Rate mean for chart",
    )
    rate_p05: List[float] = Field(
        default_factory=list,
        description="Rate percentile for chart",
    )
    rate_p95: List[float] = Field(
        default_factory=list,
        description="Rate percentile for chart",
    )
    cpr_months: List[int] = Field(
        default_factory=list,
        description="CPR for chart",
    )
    cpr_curve_monthly: List[float] = Field(
        default_factory=list,
        description="CPR for chart",
    )
