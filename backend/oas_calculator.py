
import logging
import numpy as np
from dataclasses import dataclass

from .hull_white import HullWhiteModel, HullWhiteParams, YieldCurve
from .cashflow_engine import LoanParams

log = logging.getLogger(__name__)


@dataclass
class OASResult:
    oas_bps: float
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


def _project_cashflows_batched(
    loan: LoanParams,
    inference,
    loan_ml_params: dict,
    rates: np.ndarray, # (n_paths, n_steps+1)
    gs10_rates: np.ndarray, # (n_paths, n_steps+1)
    dt: float,
    n_steps: int,
    ) -> dict:

    n_paths = rates.shape[0]
    base_age = loan.loan_age
    orig_term = loan.orig_term
    monthly_rate = loan.coupon / 12.0

    base_params = dict(loan_ml_params)
    base_params.setdefault("orig_interest_rate", loan.coupon)
    base_params.setdefault("orig_upb", loan.orig_balance)
    base_params.setdefault("orig_loan_term", orig_term)

    balance = np.full(n_paths, loan.current_balance, dtype=np.float64)

    all_cf = np.zeros((n_paths, n_steps))
    all_smm = np.zeros((n_paths, n_steps))
    all_upb = np.zeros((n_paths, n_steps))
    all_scheduled = np.zeros((n_paths, n_steps))
    all_prepayment = np.zeros((n_paths, n_steps))

    for m in range(n_steps):
        all_upb[:, m] = balance
        active = balance > 0.01

        if not np.any(active):
            break

        age_m = base_age + m
        report_month = ((base_age + m) % 12) + 1
        gs10_m = gs10_rates[:, m]   # rate at end of month m, all paths

        safe_balance = np.where(active, balance, 1.0)
        batch_params = []
        for p in range(n_paths):
            params = dict(base_params)
            params["loan_age"] = age_m
            params["current_interest_rate"] = float(loan.coupon)
            params["gs10_rate"] = float(gs10_m[p])
            params["current_upb"] = float(safe_balance[p])
            params["reporting_month"] = report_month
            batch_params.append(params)

        smm_arr = np.asarray(
            inference.predict_smm_batch(batch_params), dtype=np.float64)

        # Vectorised cashflow
        interest = balance * monthly_rate
        months_left = orig_term - base_age - m
        if months_left > 0 and monthly_rate > 0:
            x = (1 + monthly_rate) ** months_left
            pmt = balance * monthly_rate * x / (x - 1)
        else:
            pmt = balance.copy()

        sched = np.minimum(np.maximum(pmt - interest, 0.0), balance)

        # Zeroes for inactive paths
        interest = np.where(active, interest, 0.0)
        sched = np.where(active, sched, 0.0)
        smm_arr = np.where(active, smm_arr, 0.0)

        prepay = np.maximum(balance - sched, 0.0) * smm_arr

        all_smm[:, m] = smm_arr
        all_scheduled[:, m] = sched
        all_prepayment[:, m] = prepay
        all_cf[:, m] = interest + sched + prepay

        balance = np.maximum(balance - sched - prepay, 0.0)

    return {
        "total_cf": all_cf,
        "smm": all_smm,
        "upb": all_upb,
        "scheduled": all_scheduled,
        "prepayment": all_prepayment,
    }


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
    ):

    remaining_months = loan.orig_term - loan.loan_age
    T = remaining_months / 12.0
    n_steps = remaining_months

    if n_steps <= 0:
        return OASResult(
            oas_bps=0.0, model_price=market_price,
            market_price=market_price, avg_life=0.0,
            avg_smm=0.0, avg_cpr=0.0, n_paths=n_paths, converged=True)

    # HW simulation for all possible paths
    hw = HullWhiteModel(hw_params, curve)
    rates = hw.simulate(n_paths, n_steps, T, seed=seed)

    dt = T / n_steps
    t_months = np.arange(1, n_steps + 1) * dt

    # 10-year zero rate along each path
    gs10_rates = hw.zero_rate_along_path(rates, T, tau=10.0)

    # Cash flow engine
    cf = _project_cashflows_batched(
        loan, inference, loan_ml_params, rates, gs10_rates, dt, n_steps)

    all_cf = cf["total_cf"]
    all_smm = cf["smm"]
    all_scheduled = cf["scheduled"]
    all_prepayment = cf["prepayment"]

    # Discounting for CF
    cum_r = np.cumsum(rates[:, 1:] * dt, axis=1)
    face = loan.current_balance

    def _price(s: float) -> float:
        df = np.exp(-(cum_r + s * t_months[np.newaxis, :]))
        pv_paths = np.sum(all_cf * df, axis=1)
        return float(np.mean(pv_paths)) / face * 100.0

    # Solver for OAS
    s0, s1 = 0.0, 0.02
    p0 = _price(s0) - market_price
    p1 = _price(s1) - market_price

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

    # Statistics for front
    all_principal = all_scheduled + all_prepayment
    total_principal = np.sum(all_principal, axis=1)
    wal_per_path = np.where(
        total_principal > 0,
        np.sum(all_principal * t_months[np.newaxis, :], axis=1) / total_principal,
        0.0,
    )
    avg_life = float(np.mean(wal_per_path))

    avg_smm = float(np.mean(all_smm[all_smm > 0])) if np.any(all_smm > 0) else 0.0
    avg_cpr = 1.0 - (1.0 - avg_smm) ** 12

    # Paths for front
    path_rates = rates[:, 1:]
    times_all = t_months.astype(float)

    # Limit up to 120 months and 50 paths
    time_stride = max(1, n_steps // 120)
    t_idx = np.arange(0, n_steps, time_stride)
    n_sample = int(min(50, n_paths))
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n_paths, size=n_sample, replace=False)

    sampled = path_rates[np.ix_(sample_idx, t_idx)]

    mean_rates = path_rates.mean(axis=0)[t_idx]
    p05_rates = np.percentile(path_rates, 5, axis=0)[t_idx]
    p95_rates = np.percentile(path_rates, 95, axis=0)[t_idx]

    log.info("OAS solved: %.1f bp (converged=%s, WAL=%.1fy, CPR=%.2f%%, paths=%d)",
             s_star * 10_000, converged, avg_life, avg_cpr * 100, n_paths)

    return OASResult(
        oas_bps=round(s_star * 10_000, 2),
        model_price=round(_price(s_star), 4),
        market_price=market_price,
        avg_life=round(avg_life, 2),
        avg_smm=round(avg_smm, 6),
        avg_cpr=round(avg_cpr, 4),
        n_paths=n_paths,
        converged=converged,
        path_times=times_all[t_idx].tolist(),
        rate_paths=sampled.tolist(),
        rate_mean=mean_rates.tolist(),
        rate_p05=p05_rates.tolist(),
        rate_p95=p95_rates.tolist(),
    )
