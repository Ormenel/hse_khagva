"""
cashflow_engine.py  –  Mortgage Cash-Flow Projection with Prepayment
=====================================================================
Projects monthly cash flows for a fixed-rate mortgage under each
simulated interest-rate path from the Hull-White model.

Prepayment model
----------------
The engine supports two prepayment models:

1. **PSA ramp** (default, no ML required):
   Single Monthly Mortality (SMM) follows the PSA (Public Securities
   Association) standard prepayment convention scaled by a "PSA speed":

       CPR(t) = min(6%, 0.2% * t) * psa_speed / 100
       SMM(t) = 1 - (1 - CPR(t))^(1/12)

   Where t = loan age in months and psa_speed = 100 is the baseline.

2. **Rate-sensitive SMM** (built-in model):
   Adjusts the PSA SMM based on the spread between the mortgage coupon
   and the simulated short rate, capturing the rational refinancing
   incentive that drives the "option" in OAS:

       refi_incentive = coupon - r(t) - risk_premium
       multiplier     = 1 + max(refi_incentive, 0) * refi_sensitivity
       SMM_adj(t)     = min(SMM_psa(t) * multiplier, 1.0)

   Higher refi_sensitivity → more aggressive prepayment when rates drop.

Cash-flow components (per month)
---------------------------------
  Interest     = upb * monthly_rate
  Scheduled    = level payment - interest  (standard amortisation)
  Prepayment   = (upb - scheduled_principal) * SMM
  Total CF     = Interest + Scheduled + Prepayment
  Ending UPB   = Beginning UPB - Scheduled - Prepayment
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class LoanParams:
    """
    Parameters describing a single fixed-rate mortgage.

    Parameters
    ----------
    coupon          : annual mortgage coupon rate (e.g. 0.065 for 6.5%)
    orig_term       : original term in months (typically 180 or 360)
    orig_balance    : original unpaid principal balance ($)
    loan_age        : current age of the loan in months (0 = brand new)
    current_balance : current UPB (defaults to orig_balance if not supplied)
    """
    coupon: float = 0.065
    orig_term: int = 360
    orig_balance: float = 300_000.0
    loan_age: int = 0
    current_balance: float = 0.0

    def __post_init__(self):
        if self.current_balance <= 0:
            self.current_balance = self.orig_balance


@dataclass
class PrepaymentParams:
    """
    Knobs controlling the prepayment model.

    Parameters
    ----------
    psa_speed         : PSA speed as % of the standard model (100 = baseline)
    refi_sensitivity  : multiplier on refi_incentive for rate-sensitive SMM
    risk_premium      : spread between mortgage rate and short rate below
                        which prepayment incentive starts  (e.g. 0.02 = 200 bp)
    use_rate_model    : if True, apply rate-sensitive adjustment on top of PSA
    """
    psa_speed: float = 150.0
    refi_sensitivity: float = 5.0
    risk_premium: float = 0.02
    use_rate_model: bool = True


def _level_payment(balance: float, monthly_rate: float,
                   remaining_months: int) -> float:
    """Standard fixed-payment amortisation formula."""
    if monthly_rate <= 0 or remaining_months <= 0:
        return balance / max(remaining_months, 1)
    x = (1 + monthly_rate) ** remaining_months
    return balance * monthly_rate * x / (x - 1)


def psa_smm(loan_age: int, psa_speed: float = 100.0) -> float:
    """
    Single Monthly Mortality under the PSA benchmark.

    CPR(t) = min(6%, 0.2% * t) * psa_speed / 100
    SMM    = 1 - (1 - CPR)^(1/12)
    """
    cpr = min(0.06, 0.002 * loan_age) * psa_speed / 100.0
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def project_cashflows(loan: LoanParams,
                      prepay: PrepaymentParams,
                      rate_path: np.ndarray,
                      dt_years: float) -> dict:
    """
    Project monthly mortgage cash flows along a single rate path.

    Parameters
    ----------
    loan      : LoanParams describing the mortgage
    prepay    : PrepaymentParams for the prepayment model
    rate_path : 1-D array of simulated short rates (one per month)
    dt_years  : time step in years (1/12 for monthly)

    Returns
    -------
    dict with keys:
        interest   : ndarray of monthly interest payments
        scheduled  : ndarray of scheduled principal payments
        prepayment : ndarray of prepayment amounts
        total_cf   : ndarray of total cash flows (interest + principal + prepay)
        upb        : ndarray of outstanding balance at start of each month
        smm        : ndarray of applied SMM values
    """
    remaining = loan.orig_term - loan.loan_age
    n_months = min(remaining, len(rate_path))

    monthly_rate = loan.coupon / 12.0

    interest = np.zeros(n_months)
    scheduled = np.zeros(n_months)
    prepayment_arr = np.zeros(n_months)
    total_cf = np.zeros(n_months)
    upb = np.zeros(n_months)
    smm_arr = np.zeros(n_months)

    balance = loan.current_balance

    for m in range(n_months):
        if balance <= 0.01:
            break

        age = loan.loan_age + m + 1
        months_left = loan.orig_term - loan.loan_age - m

        upb[m] = balance

        # Interest
        int_pmt = balance * monthly_rate
        interest[m] = int_pmt

        # Scheduled principal (level-payment amortisation)
        pmt = _level_payment(balance, monthly_rate, months_left)
        sched_prin = pmt - int_pmt
        sched_prin = min(sched_prin, balance)
        scheduled[m] = sched_prin

        # SMM: PSA base ± rate-sensitive adjustment
        smm_base = psa_smm(age, prepay.psa_speed)

        if prepay.use_rate_model and m < len(rate_path):
            refi_incentive = loan.coupon - rate_path[m] - prepay.risk_premium
            multiplier = 1.0 + max(refi_incentive, 0.0) * prepay.refi_sensitivity
            smm_val = min(smm_base * multiplier, 1.0)
        else:
            smm_val = smm_base

        smm_arr[m] = smm_val

        # Prepayment on remaining balance after scheduled principal
        prepay_amt = (balance - sched_prin) * smm_val
        prepayment_arr[m] = prepay_amt

        # Total cash flow this month
        total_cf[m] = int_pmt + sched_prin + prepay_amt

        # Update balance
        balance = balance - sched_prin - prepay_amt
        balance = max(balance, 0.0)

    return {
        "interest": interest,
        "scheduled": scheduled,
        "prepayment": prepayment_arr,
        "total_cf": total_cf,
        "upb": upb,
        "smm": smm_arr,
    }
