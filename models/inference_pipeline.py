
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

from models.features import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES,
    _JUDICIAL_STATES,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class PrepaymentModelInference:

    def __init__(self, model_dir: str, model_name: str = "xgb"):
        # Load metadata
        with open(os.path.join(model_dir, "metadata.json")) as fh:
            self.meta = json.load(fh)

        # Load sklearn preprocessor
        with open(os.path.join(model_dir, "preprocessor.pkl"), "rb") as fh:
            self.preprocessor = pickle.load(fh)

        # Load model
        with open(os.path.join(model_dir, f"{model_name}.pkl"), "rb") as fh:
            self.model = pickle.load(fh)

        self.num_feats = self.meta.get("numeric_features", NUMERIC_FEATURES)
        self.cat_feats = self.meta.get("categorical_features", CATEGORICAL_FEATURES)
        self.bin_feats = self.meta.get("binary_features", BINARY_FEATURES)
        self.optimal_threshold = self.meta.get("optimal_threshold", 0.5)

        # Load training weight
        self.weight_ratio = float(self.meta.get("weight_ratio", 1.0))
        self.apply_prior_correction = (
            model_name != "calibrated" and self.weight_ratio > 1.0
        )
        self.model_name = model_name

        log.info("Loaded %s model + preprocessor from %s (weight_ratio=%.2f)",
                 model_name, model_dir, self.weight_ratio)

    # Функция для аджаста вероятности
    def _correct(self, p):
        if not self.apply_prior_correction:
            return p
        w = self.weight_ratio
        return p / (w * (1.0 - p) + p)

    def _build_raw_row(self, **kwargs) -> pd.DataFrame:

        # Input from interface
        fico = kwargs.get("fico", 700.0)
        orig_rate = kwargs.get("orig_interest_rate", 0.065) * 100.0
        curr_rate = kwargs.get("current_interest_rate", 0.065) * 100.0
        orig_ltv = kwargs.get("orig_ltv", 80.0)
        dti = kwargs.get("dti", 35.0)
        orig_upb = kwargs.get("orig_upb", 300000.0)
        current_upb = kwargs.get("current_upb", orig_upb)
        loan_age = kwargs.get("loan_age", 0)
        orig_loan_term = kwargs.get("orig_loan_term", 360)
        gs10_rate = kwargs.get("gs10_rate", 0.04) * 100.0
        property_state = kwargs.get("property_state", "CA")

        # Derived features
        refi_incentive = orig_rate - curr_rate
        refi_incentive_pos = max(refi_incentive, 0.0)
        burnout = loan_age * refi_incentive_pos
        upb_fraction = current_upb / orig_upb if orig_upb > 0 else 1.0
        pct_term_elapsed = loan_age / orig_loan_term if orig_loan_term > 0 else 0.0
        equity_proxy = 1.0 - orig_ltv / 100.0
        remaining = max(orig_loan_term - loan_age, 0)
        age_sq = loan_age * loan_age / 100.0
        rate_spread = curr_rate - gs10_rate
        spread_pos = max(rate_spread, 0.0)
        logit_rate_spread = 1.0 / (1.0 + np.exp(-rate_spread))
        rate_duration = curr_rate * remaining / 1200.0 if remaining > 0 else 0.0

        # FICO bucket
        if fico < 620:
            fico_bucket = "SubPrime"
        elif fico < 680:
            fico_bucket = "NearPrime"
        elif fico < 740:
            fico_bucket = "Prime"
        else:
            fico_bucket = "SuperPrime"

        # Seasoning bucket
        if loan_age <= 12:
            seasoning_bucket = "0-12m"
        elif loan_age <= 36:
            seasoning_bucket = "13-36m"
        elif loan_age <= 60:
            seasoning_bucket = "37-60m"
        elif loan_age <= 120:
            seasoning_bucket = "61-120m"
        else:
            seasoning_bucket = "120m+"

        loan_purpose = kwargs.get("loan_purpose", "P")

        row = {
            # Numeric
            "fico": fico,
            "orig_interest_rate": orig_rate,
            "current_interest_rate": curr_rate,
            "refi_incentive": refi_incentive,
            "refi_incentive_pos": refi_incentive_pos,
            "rate_spread_to_10y": rate_spread,
            "spread_pos": spread_pos,
            "orig_ltv": orig_ltv,
            "dti": dti,
            "orig_upb": orig_upb,
            "upb_fraction": upb_fraction,
            "equity_proxy": equity_proxy,
            "loan_age": float(loan_age),
            "age_sq": age_sq,
            "burnout": burnout,
            "pct_term_elapsed": pct_term_elapsed,
            "orig_loan_term": float(orig_loan_term),
            "remaining_months_to_mat": float(remaining),
            "rate_duration": rate_duration,
            "burnout_x_refi": burnout * refi_incentive_pos,
            "fico_x_refi": fico / 100.0 * refi_incentive_pos,
            "ltv_x_refi": orig_ltv * refi_incentive_pos,
            "ph_delinq_count": float(kwargs.get("ph_delinq_count", 0)),
            "excess_principal": float(kwargs.get("excess_principal", 0.0)),
            "gs10_monthly": gs10_rate,
            "logit_rate_spread_to_10y": logit_rate_spread,

            # Categorical
            "channel": kwargs.get("channel", "R"),
            "loan_purpose": loan_purpose,
            "property_type": kwargs.get("property_type", "SF"),
            "occupancy_status": kwargs.get("occupancy_status", "P"),
            "fico_bucket": fico_bucket,
            "seasoning_bucket": seasoning_bucket,
            "month_of_year": str(kwargs.get("reporting_month", 6)),
            "vintage_year": str(kwargs.get("origination_year", 2020)),

            # Binary
            "high_ltv": 1 if orig_ltv > 80 else 0,
            "term_15y": 1 if orig_loan_term <= 180 else 0,
            "is_refi": 1 if loan_purpose in ("C", "R", "U") else 0,
            "is_cashout": 1 if loan_purpose == "C" else 0,
            "is_io": kwargs.get("is_io", 0),
            "has_ppm": kwargs.get("has_ppm", 0),
            "modified": kwargs.get("modified", 0),
            "is_investor": 1 if kwargs.get("occupancy_status") == "I" else 0,
            "is_high_bal": kwargs.get("is_high_bal", 0),
            "first_time_buyer": kwargs.get("first_time_buyer", 0),
            "in_forbearance": kwargs.get("in_forbearance", 0),
            "has_deferral": kwargs.get("has_deferral", 0),
            "is_judicial_state": 1 if property_state in _JUDICIAL_STATES else 0,
            "is_hltv_refi": 1 if kwargs.get("hltv_refi_option") == "Y" else 0,
        }
        return pd.DataFrame([row])

    def _transform_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:

        for col in self.num_feats + self.bin_feats:
            if col in df.columns:
                df[col] = df[col].astype("float64")
        return df

    def predict_smm(self, **kwargs) -> float:

        row_df = self._transform_dtypes(self._build_raw_row(**kwargs))
        X = self.preprocessor.transform(row_df)
        prob = self.model.predict_proba(X)[0, 1]
        return float(self._correct(prob))

    def predict_smm_batch(self, loan_params_list: list) -> np.ndarray:

        rows = pd.concat([self._build_raw_row(**lp) for lp in loan_params_list],
                         ignore_index=True)
        rows = self._transform_dtypes(rows)
        X = self.preprocessor.transform(rows)
        probs = self.model.predict_proba(X)[:, 1]
        return self._correct(probs)

    def predict_smm_along_path(self,
                               loan_params: dict,
                               rate_path: np.ndarray) -> np.ndarray:

        n = len(rate_path)
        smm_path = np.zeros(n)

        params = dict(loan_params)
        base_age = params.get("loan_age", 0)
        balance = params.get("current_upb", params.get("orig_upb", 300000))
        coupon = params.get("orig_interest_rate", 0.065)
        monthly_rate = coupon / 12.0

        for m in range(n):
            if balance <= 0.01:
                break

            # обновляем переменные которые меняются со временем
            params["loan_age"] = base_age + m
            params["current_interest_rate"] = coupon
            params["gs10_rate"] = rate_path[m]
            params["current_upb"] = balance
            params["reporting_month"] = ((base_age + m) % 12) + 1

            smm_val = self.predict_smm(**params)
            smm_path[m] = smm_val

            # Amortise
            rem = params.get("orig_loan_term", 360) - (base_age + m)
            if rem > 0 and monthly_rate > 0:
                x = (1 + monthly_rate) ** rem
                pmt = balance * monthly_rate * x / (x - 1)
            else:
                pmt = balance
            sched = min(pmt - balance * monthly_rate, balance)
            prepay = (balance - sched) * smm_val
            balance = max(balance - sched - prepay, 0.0)

        return smm_path

# Cash flow projection
def project_cashflows_ml(loan_params: dict,
                         inference: PrepaymentModelInference,
                         rate_path: np.ndarray,
                         dt_years: float) -> dict:

    coupon = loan_params.get("orig_interest_rate", 0.065)
    orig_term = loan_params.get("orig_loan_term", 360)
    base_age = loan_params.get("loan_age", 0)
    balance = loan_params.get("current_upb",
                               loan_params.get("orig_upb", 300000))
    monthly_rate = coupon / 12.0

    remaining = orig_term - base_age
    n_months = min(remaining, len(rate_path))

    interest = np.zeros(n_months)
    scheduled = np.zeros(n_months)
    prepayment_arr = np.zeros(n_months)
    total_cf = np.zeros(n_months)
    upb = np.zeros(n_months)
    smm_arr = np.zeros(n_months)

    params = dict(loan_params)

    for m in range(n_months):
        if balance <= 0.01:
            break

        upb[m] = balance

        # обновляем переменные которые меняются со временем
        params["loan_age"] = base_age + m + 1
        params["current_interest_rate"] = coupon
        params["gs10_rate"] = rate_path[m]
        params["current_upb"] = balance
        params["reporting_month"] = ((base_age + m) % 12) + 1

        # Interest
        int_pmt = balance * monthly_rate
        interest[m] = int_pmt

        # Scheduled principal
        months_left = orig_term - base_age - m
        if months_left > 0 and monthly_rate > 0:
            x = (1 + monthly_rate) ** months_left
            pmt = balance * monthly_rate * x / (x - 1)
        else:
            pmt = balance
        sched = min(pmt - int_pmt, balance)
        scheduled[m] = sched

        # ML-predicted SMM
        smm_val = inference.predict_smm(**params)
        smm_arr[m] = smm_val

        # Prepayment
        prepay = (balance - sched) * smm_val
        prepayment_arr[m] = prepay

        total_cf[m] = int_pmt + sched + prepay
        balance = max(balance - sched - prepay, 0.0)

    return {
        "interest": interest,
        "scheduled": scheduled,
        "prepayment": prepayment_arr,
        "total_cf": total_cf,
        "upb": upb,
        "smm": smm_arr,
    }
