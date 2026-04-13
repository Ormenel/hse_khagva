from pydantic import BaseModel, Field
from typing import List, Optional


class YieldCurveInput(BaseModel):
    tenors: List[float] = Field(
        default=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
        description="Maturities in years",
    )
    rates: List[float] = Field(
        default=[0.045, 0.046, 0.047, 0.048, 0.048, 0.047, 0.046, 0.045, 0.043, 0.042],
        description="Continuously-compounded zero rates (decimal, e.g. 0.05 = 5%)",
    )


class LoanInput(BaseModel):
    coupon: float = Field(
        default=0.065,
        description="Annual coupon rate (decimal, e.g. 0.065 = 6.5%)",
        ge=0.001, le=0.30,
    )
    orig_term: int = Field(
        default=360,
        description="Original term in months (180 or 360)",
        ge=12, le=600,
    )
    orig_balance: float = Field(
        default=300_000.0,
        description="Original UPB ($)",
        ge=1_000,
    )
    loan_age: int = Field(
        default=0,
        description="Current loan age in months",
        ge=0,
    )
    current_balance: Optional[float] = Field(
        default=None,
        description="Current UPB ($); defaults to orig_balance if omitted",
    )


class HullWhiteInput(BaseModel):
    a: float = Field(
        default=0.1,
        description="Mean-reversion speed (typical: 0.01 – 0.5)",
        ge=0.001, le=2.0,
    )
    sigma: float = Field(
        default=0.01,
        description="Short-rate volatility (typical: 0.005 – 0.03)",
        ge=0.0001, le=0.10,
    )


class PrepaymentInput(BaseModel):
    psa_speed: float = Field(
        default=150.0,
        description="PSA speed (100 = standard benchmark)",
        ge=0, le=1000,
    )
    refi_sensitivity: float = Field(
        default=5.0,
        description="Multiplier on refinancing incentive for rate-sensitive SMM",
        ge=0, le=50,
    )
    risk_premium: float = Field(
        default=0.02,
        description="Spread above short rate before refi incentive kicks in (decimal)",
        ge=0, le=0.10,
    )
    use_rate_model: bool = Field(
        default=True,
        description="Apply rate-sensitive prepayment adjustment on top of PSA",
    )


class MLPrepaymentInput(BaseModel):
    model_dir: str = Field(
        description="Path to saved_models/ directory with preprocessor.pkl and model pickles",
    )
    model_name: str = Field(
        default="xgb",
        description="Model to load: lr, rf, gbt, xgb, lgb, stacking, calibrated",
    )
    fico: float = Field(default=700.0, description="Borrower FICO score", ge=300, le=850)
    orig_ltv: float = Field(default=80.0, description="Original LTV (%)", ge=1, le=200)
    dti: float = Field(default=35.0, description="Debt-to-income ratio (%)", ge=0, le=100)
    channel: str = Field(default="R", description="Origination channel: R(etail), B(roker), C(orrespondent)")
    loan_purpose: str = Field(default="P", description="P(urchase), C(ash-out refi), R(efi), U(nknown)")
    property_type: str = Field(default="SF", description="SF, CO, PU, MH, CP")
    occupancy_status: str = Field(default="P", description="P(rimary), S(econd), I(nvestor)")
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
    hull_white: HullWhiteInput = Field(default_factory=HullWhiteInput)
    prepayment: PrepaymentInput = Field(default_factory=PrepaymentInput)
    ml_prepayment: Optional[MLPrepaymentInput] = Field(
        default=None,
        description="If provided, use sklearn ML model instead of PSA for prepayment",
    )
    yield_curve: YieldCurveInput = Field(default_factory=YieldCurveInput)
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
