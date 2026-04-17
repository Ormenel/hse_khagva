"""
oas_calculator.py  –  Option-Adjusted Spread Calculator
========================================================
Computes the OAS (Option-Adjusted Spread) for a fixed-rate mortgage
or MBS (Mortgage-Backed Security) using Monte Carlo simulation.

Algorithm
---------
1.  Simulate *N* interest-rate paths with the Hull-White model.
2.  For each path, project mortgage cash flows using the prepayment engine.
3.  Discount each path's cash flows at the simulated short rate PLUS
    a constant spread *s* (the OAS candidate).
4.  Compute the average present value across all paths:

        PV(s)  =  (1/N) * SUM_{paths} SUM_{months} CF(t) * exp(-(R(t)+s)*t)

5.  Find *s* such that PV(s) = market price (or par if price = 100).
    Uses the secant method (derivative-free root finding).

Interpretation
--------------
OAS is the constant spread over the risk-free term structure that makes
the model price match the market price, AFTER accounting for the embedded
prepayment option.

    OAS > 0 : security trades cheap (yield above fair value)
    OAS = 0 : fairly priced by the model
    OAS < 0 : security trades rich (yield below fair value)

References
----------
Fabozzi (2006), "Fixed Income Analysis", CFA Institute Investment Series.
Tuckman & Serrat (2012), "Fixed Income Securities", 3rd Edition, Wiley.
"""

import logging
import numpy as np
from dataclasses import dataclass

from .hull_white import HullWhiteModel, HullWhiteParams, YieldCurve
from .cashflow_engine import LoanParams, PrepaymentParams, project_cashflows

log = logging.getLogger(__name__)


@dataclass
class OASResult:
    """
    Result container for an OAS calculation.

    Attributes
    ----------
    oas_bps        : OAS in basis points (1 bp = 0.0001)
    model_price    : model-implied price at the solved OAS
    market_price   : target market price used in the solve
    avg_life       : weighted-average life in years
    avg_smm        : average single-monthly mortality across paths
    avg_cpr        : annualised CPR  =  1 - (1 - avg_smm)^12
    n_paths        : number of Monte Carlo paths used
    converged      : True if the solver converged within tolerance
    """
    oas_bps: float
    model_price: float
    market_price: float
    avg_life: float
    avg_smm: float
    avg_cpr: float
    n_paths: int
    converged: bool


def compute_oas(
    loan: LoanParams,
    prepay: PrepaymentParams,
    curve: YieldCurve,
    hw_params: HullWhiteParams,
    market_price: float = 100.0,
    n_paths: int = 500,
    seed: int = 42,
    tol: float = 0.005,
    max_iter: int = 50,
) -> OASResult:
    """
    Compute OAS for a fixed-rate mortgage / MBS.

    Parameters
    ----------
    loan         : mortgage parameters (coupon, term, balance, age)
    prepay       : prepayment model parameters (PSA speed, refi sensitivity)
    curve        : initial zero-coupon yield curve
    hw_params    : Hull-White model parameters (a, sigma)
    market_price : target clean price (100 = par)
    n_paths      : number of Monte Carlo paths
    seed         : random seed
    tol          : convergence tolerance for price difference ($)
    max_iter     : max iterations for the secant solver

    Returns
    -------
    OASResult with the solved spread and diagnostics.
    """
    remaining_months = loan.orig_term - loan.loan_age
    T = remaining_months / 12.0   # horizon in years
    n_steps = remaining_months     # monthly steps

    if n_steps <= 0:
        return OASResult(
            oas_bps=0.0, model_price=market_price,
            market_price=market_price, avg_life=0.0,
            avg_smm=0.0, avg_cpr=0.0, n_paths=n_paths, converged=True)

    # ── 1. Simulate interest-rate paths ───────────────────────────────────
    hw = HullWhiteModel(hw_params, curve)
    rates = hw.simulate(n_paths, n_steps, T, seed=seed)   # (n_paths, n_steps+1)

    dt = T / n_steps  # ≈ 1/12

    # Time in years for each cashflow month: [dt, 2*dt, ..., n_steps*dt]
    t_months = np.arange(1, n_steps + 1) * dt

    # ── 2. Project cash flows for each path ───────────────────────────────
    # Collect per-path: total_cf array and smm array
    all_cf = np.zeros((n_paths, n_steps))
    all_smm = np.zeros((n_paths, n_steps))
    all_upb = np.zeros((n_paths, n_steps))

    for p in range(n_paths):
        path_rates = rates[p, 1:]  # skip t=0, take rates at [dt, 2dt, ...]
        cf = project_cashflows(loan, prepay, path_rates, dt)
        all_cf[p] = cf["total_cf"]
        all_smm[p] = cf["smm"]
        all_upb[p] = cf["upb"]

    # ── 3. Cumulative short-rate integral for discounting ─────────────────
    # cum_r[p, m] = sum_{k=1..m} r(t_k) * dt   (path-wise)
    cum_r = np.cumsum(rates[:, 1:] * dt, axis=1)   # (n_paths, n_steps)

    # ── 4. Price as a function of spread s ────────────────────────────────
    # Normalise to per-$100 of face
    face = loan.current_balance

    def _price(s: float) -> float:
        """Average present value across paths at spread s, per $100."""
        # Discount factor per path per month: exp(-(cum_r + s*t))
        df = np.exp(-(cum_r + s * t_months[np.newaxis, :]))
        pv_paths = np.sum(all_cf * df, axis=1)
        return float(np.mean(pv_paths)) / face * 100.0

    # ── 5. Solve for OAS using secant method ──────────────────────────────
    # Initial bracket: try s=0 and s=0.02 (200 bp)
    s0, s1 = 0.0, 0.02
    p0, p1 = _price(s0) - market_price, _price(s1) - market_price

    converged = False
    s_star = 0.0

    for _ in range(max_iter):
        if abs(p1) < tol:
            s_star = s1
            converged = True
            break
        if abs(p1 - p0) < 1e-14:
            # Flat – can't compute secant step; use midpoint
            s_star = (s0 + s1) / 2.0
            break
        # Secant update
        s_new = s1 - p1 * (s1 - s0) / (p1 - p0)
        # Clamp to reasonable range (-500 bp to +2000 bp)
        s_new = max(-0.05, min(0.20, s_new))
        s0, p0 = s1, p1
        s1 = s_new
        p1 = _price(s1) - market_price

    if not converged and abs(p1) < tol:
        s_star = s1
        converged = True

    # ── 6. Diagnostics ────────────────────────────────────────────────────
    model_price = _price(s_star) + market_price  # undo subtraction

    # Weighted-average life (WAL) across all paths
    # WAL = sum(t * principal) / total_principal
    all_principal = np.zeros_like(all_cf)
    for p in range(n_paths):
        path_rates = rates[p, 1:]
        cf = project_cashflows(loan, prepay, path_rates, dt)
        all_principal[p] = cf["scheduled"] + cf["prepayment"]

    total_principal = np.sum(all_principal, axis=1)  # per path
    wal_per_path = np.where(
        total_principal > 0,
        np.sum(all_principal * t_months[np.newaxis, :], axis=1) / total_principal,
        0.0,
    )
    avg_life = float(np.mean(wal_per_path))

    avg_smm = float(np.mean(all_smm[all_smm > 0])) if np.any(all_smm > 0) else 0.0
    avg_cpr = 1.0 - (1.0 - avg_smm) ** 12

    return OASResult(
        oas_bps=round(s_star * 10_000, 2),
        model_price=round(_price(s_star) + market_price, 4),
        market_price=market_price,
        avg_life=round(avg_life, 2),
        avg_smm=round(avg_smm, 6),
        avg_cpr=round(avg_cpr, 4),
        n_paths=n_paths,
        converged=converged,
    )


def compute_oas_ml(
    loan: LoanParams,
    inference,
    loan_ml_params: dict,
    curve: YieldCurve,
    hw_params: HullWhiteParams,
    market_price: float = 100.0,
    n_paths: int = 500,
    seed: int = 42,
    tol: float = 0.005,
    max_iter: int = 50,
) -> OASResult:
    """
    Compute OAS using the sklearn ML prepayment model instead of PSA.

    Parameters
    ----------
    loan           : mortgage parameters (coupon, term, balance, age)
    inference      : fitted PrepaymentModelInference instance
    loan_ml_params : dict of loan-level features for the ML model
                     (fico, orig_ltv, dti, channel, etc.)
    curve          : initial zero-coupon yield curve
    hw_params      : Hull-White model parameters (a, sigma)
    market_price   : target clean price (100 = par)
    n_paths        : number of Monte Carlo paths
    seed           : random seed
    tol            : convergence tolerance for price difference ($)
    max_iter       : max iterations for the secant solver

    Returns
    -------
    OASResult with the solved spread and diagnostics.
    """
    from models.inference_pipeline_v3 import project_cashflows_ml

    remaining_months = loan.orig_term - loan.loan_age
    T = remaining_months / 12.0
    n_steps = remaining_months

    if n_steps <= 0:
        return OASResult(
            oas_bps=0.0, model_price=market_price,
            market_price=market_price, avg_life=0.0,
            avg_smm=0.0, avg_cpr=0.0, n_paths=n_paths, converged=True)

    # ── 1. Simulate interest-rate paths ───────────────────────────────────
    hw = HullWhiteModel(hw_params, curve)
    rates = hw.simulate(n_paths, n_steps, T, seed=seed)

    dt = T / n_steps

    t_months = np.arange(1, n_steps + 1) * dt

    # ── 2. Build base ML params from loan + user-supplied features ────────
    ml_params = dict(loan_ml_params)
    ml_params.setdefault("orig_interest_rate", loan.coupon)
    ml_params.setdefault("orig_upb", loan.orig_balance)
    ml_params.setdefault("current_upb", loan.current_balance)
    ml_params.setdefault("loan_age", loan.loan_age)
    ml_params.setdefault("orig_loan_term", loan.orig_term)

    # ── 3. Project cash flows for each path using ML model ────────────────
    all_cf = np.zeros((n_paths, n_steps))
    all_smm = np.zeros((n_paths, n_steps))
    all_upb = np.zeros((n_paths, n_steps))

    for p in range(n_paths):
        path_rates = rates[p, 1:]
        cf = project_cashflows_ml(ml_params, inference, path_rates, dt)
        n = min(n_steps, len(cf["total_cf"]))
        all_cf[p, :n] = cf["total_cf"][:n]
        all_smm[p, :n] = cf["smm"][:n]
        all_upb[p, :n] = cf["upb"][:n]

    # ── 4. Cumulative short-rate integral for discounting ─────────────────
    cum_r = np.cumsum(rates[:, 1:] * dt, axis=1)

    # ── 5. Price as a function of spread s ────────────────────────────────
    face = loan.current_balance

    def _price(s: float) -> float:
        df = np.exp(-(cum_r + s * t_months[np.newaxis, :]))
        pv_paths = np.sum(all_cf * df, axis=1)
        return float(np.mean(pv_paths)) / face * 100.0

    # ── 6. Solve for OAS using secant method ──────────────────────────────
    s0, s1 = 0.0, 0.02
    p0, p1 = _price(s0) - market_price, _price(s1) - market_price

    converged = False
    s_star = 0.0

    for _ in range(max_iter):
        if abs(p1) < tol:
            s_star = s1
            converged = True
            break
        if abs(p1 - p0) < 1e-14:
            s_star = (s0 + s1) / 2.0
            break
        s_new = s1 - p1 * (s1 - s0) / (p1 - p0)
        s_new = max(-0.05, min(0.20, s_new))
        s0, p0 = s1, p1
        s1 = s_new
        p1 = _price(s1) - market_price

    if not converged and abs(p1) < tol:
        s_star = s1
        converged = True

    # ── 7. Diagnostics ────────────────────────────────────────────────────
    all_principal = np.zeros_like(all_cf)
    for p in range(n_paths):
        path_rates = rates[p, 1:]
        cf = project_cashflows_ml(ml_params, inference, path_rates, dt)
        n = min(n_steps, len(cf["scheduled"]))
        all_principal[p, :n] = cf["scheduled"][:n] + cf["prepayment"][:n]

    total_principal = np.sum(all_principal, axis=1)
    wal_per_path = np.where(
        total_principal > 0,
        np.sum(all_principal * t_months[np.newaxis, :], axis=1) / total_principal,
        0.0,
    )
    avg_life = float(np.mean(wal_per_path))

    avg_smm = float(np.mean(all_smm[all_smm > 0])) if np.any(all_smm > 0) else 0.0
    avg_cpr = 1.0 - (1.0 - avg_smm) ** 12

    log.info("OAS-ML solved: %.1f bp (converged=%s, WAL=%.1fy, CPR=%.2f%%)",
             s_star * 10_000, converged, avg_life, avg_cpr * 100)

    return OASResult(
        oas_bps=round(s_star * 10_000, 2),
        model_price=round(_price(s_star) + market_price, 4),
        market_price=market_price,
        avg_life=round(avg_life, 2),
        avg_smm=round(avg_smm, 6),
        avg_cpr=round(avg_cpr, 4),
        n_paths=n_paths,
        converged=converged,
    )
