import logging
from dataclasses import dataclass

import numpy as np

from .hull_white import (
    HullWhiteModel, HullWhiteParams, YieldCurve, ParCurveBootstrapper,
)
from .cashflow_engine import LoanParams

log = logging.getLogger(__name__)


@dataclass
class OASResult:
    oas_bps: float
    oas_expected_bps: float
    oas_unexpected_bps: float
    model_price: float
    market_price: float
    avg_life: float
    avg_smm: float
    avg_cpr: float
    n_paths: int
    converged: bool
    path_times: list = None
    rate_paths: list = None
    rate_mean: list = None
    rate_p05: list = None
    rate_p95: list = None
    cpr_months: list = None
    cpr_curve_monthly: list = None


def _project_cashflows_no_prepayment(loan: LoanParams, n_steps: int) -> np.ndarray:
    monthly_rate = loan.coupon / 12.0
    balance = float(loan.current_balance)
    base_age = loan.loan_age
    orig_term = loan.orig_term

    total_cf = np.zeros(n_steps, dtype=np.float64)

    for m in range(n_steps):
        if balance <= 0.01:
            break

        interest = balance * monthly_rate
        months_left = orig_term - base_age - m
        if months_left > 0 and monthly_rate > 0:
            x = (1 + monthly_rate) ** months_left
            pmt = balance * monthly_rate * x / (x - 1)
        else:
            pmt = balance

        sched = min(max(pmt - interest, 0.0), balance)
        total_cf[m] = interest + sched
        balance = max(balance - sched, 0.0)

    return total_cf


def _solve_implied_market_rate(
    cashflows: np.ndarray,
    cum_r: np.ndarray,
    t_months: np.ndarray,
    face: float,
    target_price: float,
    tol: float,
    max_iter: int,
) -> tuple[float, bool, float]:

    if cashflows.ndim == 1:
        cf = cashflows[np.newaxis, :]
    else:
        cf = cashflows

    def _price(s: float) -> float:
        df = np.exp(-(cum_r + s * t_months[np.newaxis, :]))
        return float(np.mean(np.sum(cf * df, axis=1))) / face * 100.0

    s0, s1 = 0.0, 0.02
    p0 = _price(s0) - target_price
    p1 = _price(s1) - target_price
    converged = abs(p1) < tol
    s_star = s1 if converged else 0.0

    for _ in range(max_iter):
        if converged:
            break
        if abs(p1 - p0) < 1e-14:
            s_star = (s0 + s1) / 2.0
            break
        s_new = max(-0.05, min(0.20, s1 - p1 * (s1 - s0) / (p1 - p0)))
        s0, p0 = s1, p1
        s1, p1 = s_new, _price(s_new) - target_price
        if abs(p1) < tol:
            s_star = s1
            converged = True

    return float(s_star), converged, float(_price(s_star))


def _project_cashflows(
    loan: LoanParams,
    inference,
    loan_ml_params: dict,
    gs10_par: np.ndarray,          # (n_paths, n_steps) — 10Y CMT par yield
    n_steps: int,
) -> dict:

    n_paths = gs10_par.shape[0]
    base_age = loan.loan_age
    orig_term = loan.orig_term
    monthly_rate = loan.coupon / 12.0

    base_params = dict(loan_ml_params)
    base_params["orig_interest_rate"] = loan.coupon
    base_params["current_interest_rate"] = loan.coupon   # loan rate = coupon
    base_params["orig_upb"] = loan.orig_balance
    base_params["orig_loan_term"] = orig_term

    balance = np.full(n_paths, loan.current_balance, dtype=np.float64)

    cf = np.zeros((n_paths, n_steps))
    smm = np.zeros((n_paths, n_steps))
    upb = np.zeros((n_paths, n_steps))
    scheduled = np.zeros((n_paths, n_steps))
    prepayment = np.zeros((n_paths, n_steps))

    for m in range(n_steps):
        upb[:, m] = balance
        active = balance > 0.01
        if not np.any(active):
            break

        age_m = base_age + m
        report_month = (age_m % 12) + 1
        safe_bal = np.where(active, balance, 1.0)

        batch = [
            {**base_params,
             "loan_age": age_m,
             "gs10_rate": float(gs10_par[p, m]),
             "current_upb": float(safe_bal[p]),
             "reporting_month": report_month}
            for p in range(n_paths)
        ]
        smm_m = np.asarray(inference.predict_smm(batch), dtype=np.float64)

        # Scheduled amortisation from current balance
        interest = balance * monthly_rate
        months_left = orig_term - age_m
        if months_left > 0 and monthly_rate > 0:
            x = (1 + monthly_rate) ** months_left
            pmt = balance * monthly_rate * x / (x - 1)
        else:
            pmt = balance.copy()
        sched = np.minimum(np.maximum(pmt - interest, 0.0), balance)

        interest = np.where(active, interest, 0.0)
        sched = np.where(active, sched, 0.0)
        smm_m = np.where(active, smm_m, 0.0)
        prepay = np.maximum(balance - sched, 0.0) * smm_m

        smm[:, m] = smm_m
        scheduled[:, m] = sched
        prepayment[:, m] = prepay
        cf[:, m] = interest + sched + prepay
        balance = np.maximum(balance - sched - prepay, 0.0)

    return {
        "cf": cf, "smm": smm, "upb": upb,
        "scheduled": scheduled, "prepayment": prepayment,
    }


def compute_oas_ml(
    loan: LoanParams,
    inference,
    loan_ml_params: dict,
    par_curve: YieldCurve,
    hw_params: HullWhiteParams,
    market_price: float = 100.0,
    n_paths: int = 500,
    seed: int = 42,
    tol: float = 0.005,
    max_iter: int = 50,
) -> OASResult:
    remaining_months = loan.orig_term - loan.loan_age
    if remaining_months <= 0:
        return OASResult(
            oas_bps=0.0,
            oas_expected_bps=0.0,
            oas_unexpected_bps=0.0,
            model_price=market_price,
            market_price=market_price, avg_life=0.0,
            avg_smm=0.0, avg_cpr=0.0, n_paths=n_paths, converged=True,
            cpr_months=[], cpr_curve_monthly=[])

    T = remaining_months / 12.0
    n_steps = remaining_months
    dt = T / n_steps
    t_months = np.arange(1, n_steps + 1) * dt

    # Bootstrap par to zero curve
    zero_curve = ParCurveBootstrapper(
        tenors=par_curve.tenors, par_rates=par_curve.rates, coupon_freq=2,
    ).bootstrap_zero_curve()
    hw = HullWhiteModel(hw_params, zero_curve)

    # Get 10Y rate full pack path
    short_rates = hw.simulate(n_paths, n_steps, T, seed=seed)
    gs10_full = hw.par_yield_along_path(short_rates, T, tau=10.0, coupon_freq=2)
    gs10_par = gs10_full[:, :n_steps]

    proj = _project_cashflows(loan, inference, loan_ml_params, gs10_par, n_steps)
    cf = proj["cf"]
    smm = proj["smm"]
    principal = proj["scheduled"] + proj["prepayment"]

    # Risk-neutral discount using the simulated short rate
    cum_r = np.cumsum(short_rates[:, 1:] * dt, axis=1)
    face = loan.current_balance

    no_prepayment_cf = _project_cashflows_no_prepayment(loan, n_steps)

    # OAS = implied rate for no-prepayment cashflows minus implied rate
    # for cashflows with prepayments across all simulated paths.
    rate_no_pp, conv_no_pp, _ = _solve_implied_market_rate(
        cashflows=no_prepayment_cf,
        cum_r=cum_r,
        t_months=t_months,
        face=face,
        target_price=market_price,
        tol=tol,
        max_iter=max_iter,
    )
    rate_with_pp, conv_with_pp, model_price = _solve_implied_market_rate(
        cashflows=cf,
        cum_r=cum_r,
        t_months=t_months,
        face=face,
        target_price=market_price,
        tol=tol,
        max_iter=max_iter,
    )
    oas_rate = rate_no_pp - rate_with_pp
    converged = conv_no_pp and conv_with_pp

    hw_expected = HullWhiteModel(
        HullWhiteParams(a=hw_params.a, sigma=0.0),
        zero_curve,
    )
    short_rates_expected = hw_expected.simulate(
        n_paths=1, n_steps=n_steps, T=T, seed=seed
    )
    gs10_expected = hw_expected.par_yield_along_path(
        short_rates_expected, T, tau=10.0, coupon_freq=2
    )[:, :n_steps]
    proj_expected = _project_cashflows(
        loan=loan,
        inference=inference,
        loan_ml_params=loan_ml_params,
        gs10_par=gs10_expected,
        n_steps=n_steps,
    )
    cum_r_expected = np.cumsum(short_rates_expected[:, 1:] * dt, axis=1)
    rate_no_pp_expected, _, _ = _solve_implied_market_rate(
        cashflows=no_prepayment_cf,
        cum_r=cum_r_expected,
        t_months=t_months,
        face=face,
        target_price=market_price,
        tol=tol,
        max_iter=max_iter,
    )
    rate_with_pp_expected, _, _ = _solve_implied_market_rate(
        cashflows=proj_expected["cf"],
        cum_r=cum_r_expected,
        t_months=t_months,
        face=face,
        target_price=market_price,
        tol=tol,
        max_iter=max_iter,
    )
    oas_expected_rate = rate_no_pp_expected - rate_with_pp_expected
    oas_unexpected_rate = oas_rate - oas_expected_rate
    oas_bps = round(oas_rate * 10_000, 2)
    oas_expected_bps = round(oas_expected_rate * 10_000, 2)
    oas_unexpected_bps = round(oas_bps - oas_expected_bps, 2)

    # WAL / CPR aggregates
    total_principal = np.sum(principal, axis=1)
    wal_per_path = np.where(
        total_principal > 0,
        np.sum(principal * t_months[np.newaxis, :], axis=1) / total_principal,
        0.0,
    )
    avg_life = float(np.mean(wal_per_path))

    smm_per_path = np.where(
        total_principal > 0,
        np.sum(principal * smm, axis=1) / total_principal,
        0.0,
    )

    avg_smm = float(np.mean(smm_per_path))
    cpr_per_path = 1.0 - (1.0 - smm_per_path) ** 12
    avg_cpr = float(np.mean(cpr_per_path))

    # Path sampling for the UI
    path_rates = short_rates[:, 1:]
    time_stride = max(1, n_steps // 120)
    t_idx = np.arange(0, n_steps, time_stride)
    n_sample = int(min(50, n_paths))
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n_paths, size=n_sample, replace=False)

    mean_short = short_rates.mean(axis=0, keepdims=True)
    mean_gs10 = hw.par_yield_along_path(
        mean_short, T, tau=10.0, coupon_freq=2
    )[:, :n_steps]
    mean_proj = _project_cashflows(
        loan=loan,
        inference=inference,
        loan_ml_params=loan_ml_params,
        gs10_par=mean_gs10,
        n_steps=n_steps,
    )
    smm_mean_path = mean_proj["smm"][0]
    cpr_curve_monthly = 1.0 - (1.0 - smm_mean_path) ** 12
    cpr_curve_monthly = np.clip(cpr_curve_monthly, 0.0, 1.0)
    cpr_months = np.arange(1, n_steps + 1, dtype=int).tolist()

    log.info(
        "OAS solved: total=%.1f bp (expected=%.1f bp, unexpected=%.1f bp; "
        "no-pp rate=%.1f bp, with-pp rate=%.1f bp, conv=%s, WAL=%.1fy, CPR=%.2f%%, paths=%d)",
        oas_bps,
        oas_expected_bps,
        oas_unexpected_bps,
        rate_no_pp * 10_000,
        rate_with_pp * 10_000,
        converged,
        avg_life,
        avg_cpr * 100,
        n_paths,
    )

    return OASResult(
        oas_bps=oas_bps,
        oas_expected_bps=oas_expected_bps,
        oas_unexpected_bps=oas_unexpected_bps,
        model_price=round(model_price, 4),
        market_price=market_price,
        avg_life=round(avg_life, 2),
        avg_smm=round(avg_smm, 6),
        avg_cpr=round(avg_cpr, 4),
        n_paths=n_paths,
        converged=converged,
        path_times=t_months[t_idx].tolist(),
        rate_paths=path_rates[np.ix_(sample_idx, t_idx)].tolist(),
        rate_mean=path_rates.mean(axis=0)[t_idx].tolist(),
        rate_p05=np.percentile(path_rates, 5, axis=0)[t_idx].tolist(),
        rate_p95=np.percentile(path_rates, 95, axis=0)[t_idx].tolist(),
        cpr_months=cpr_months,
        cpr_curve_monthly=cpr_curve_monthly.tolist(),
    )
