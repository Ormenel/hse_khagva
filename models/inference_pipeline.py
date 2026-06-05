import warnings
import json
import logging
import os
import pickle
from typing import Iterable

import numpy as np
import pandas as pd

from models.features import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES,
    _JUDICIAL_STATES,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _fico_bucket(fico: float) -> str:

    if fico < 620:
        return "SubPrime"
    if fico < 680:
        return "NearPrime"
    if fico < 740:
        return "Prime"
    return "SuperPrime"


def _seasoning_bucket(loan_age: int) -> str:

    if loan_age <= 12:
        return "0-12m"
    if loan_age <= 36:
        return "13-36m"
    if loan_age <= 60:
        return "37-60m"
    if loan_age <= 120:
        return "61-120m"
    return "120m+"


def build_feature_row(**kw) -> dict:

    fico = float(kw.get("fico", 700.0))
    orig_rate = float(kw.get("orig_interest_rate", 0.065)) * 100.0
    # current_interest_rate = loan rate
    curr_rate = float(kw.get("current_interest_rate", orig_rate / 100.0)) * 100.0
    gs10_rate = float(kw.get("gs10_rate", 0.04)) * 100.0
    orig_ltv = float(kw.get("orig_ltv", 80.0))
    dti = float(kw.get("dti", 35.0))
    orig_upb = float(kw.get("orig_upb", 300_000.0))
    current_upb = float(kw.get("current_upb", orig_upb))
    loan_age = int(kw.get("loan_age", 0))
    orig_term = int(kw.get("orig_loan_term", 360))
    loan_purpose = kw.get("loan_purpose", "P")
    occupancy = kw.get("occupancy_status", "P")
    property_state = kw.get("property_state", "CA")

    refi_incentive = orig_rate - curr_rate
    refi_pos = max(refi_incentive, 0.0)
    burnout = loan_age * refi_pos
    remaining = max(orig_term - loan_age, 0)
    rate_spread = curr_rate - gs10_rate

    return {
        # Numeric
        "fico": fico,
        "orig_interest_rate": orig_rate,
        "current_interest_rate": curr_rate,
        "refi_incentive": refi_incentive,
        "refi_incentive_pos": refi_pos,
        "rate_spread_to_10y": rate_spread,
        "spread_pos": max(rate_spread, 0.0),
        "orig_ltv": orig_ltv,
        "dti": dti,
        "orig_upb": orig_upb,
        "upb_fraction": current_upb / orig_upb if orig_upb > 0 else 1.0,
        "equity_proxy": 1.0 - orig_ltv / 100.0,
        "loan_age": float(loan_age),
        "age_sq": loan_age * loan_age / 100.0,
        "burnout": burnout,
        "pct_term_elapsed": loan_age / orig_term if orig_term > 0 else 0.0,
        "orig_loan_term": float(orig_term),
        "remaining_months_to_mat": float(remaining),
        "rate_duration": curr_rate * remaining / 1200.0 if remaining > 0 else 0.0,
        "burnout_x_refi": burnout * refi_pos,
        "fico_x_refi": fico / 100.0 * refi_pos,
        "ltv_x_refi": orig_ltv * refi_pos,
        "ph_delinq_count": float(kw.get("ph_delinq_count", 0)),
        "excess_principal": float(kw.get("excess_principal", 0.0)),
        "gs10_monthly": gs10_rate,
        "logit_rate_spread_to_10y": 1.0 / (1.0 + np.exp(-rate_spread)),

        # Categorical (training stored these as strings)
        "channel": kw.get("channel", "R"),
        "loan_purpose": loan_purpose,
        "property_type": kw.get("property_type", "SF"),
        "occupancy_status": occupancy,
        "fico_bucket": _fico_bucket(fico),
        "seasoning_bucket": _seasoning_bucket(loan_age),
        "month_of_year": str(kw.get("reporting_month", 6)),
        "vintage_year": kw.get("origination_year", 2014),

        # Binary
        "high_ltv": 1 if orig_ltv > 80 else 0,
        "term_15y": 1 if orig_term <= 180 else 0,
        "is_refi": 1 if loan_purpose in ("C", "R", "U") else 0,
        "is_cashout": 1 if loan_purpose == "C" else 0,
        "is_io": int(kw.get("is_io", 0)),
        "has_ppm": int(kw.get("has_ppm", 0)),
        "modified": int(kw.get("modified", 0)),
        "is_investor": 1 if occupancy == "I" else 0,
        "is_high_bal": int(kw.get("is_high_bal", 0)),
        "first_time_buyer": int(kw.get("first_time_buyer", 0)),
        "in_forbearance": int(kw.get("in_forbearance", 0)),
        "has_deferral": int(kw.get("has_deferral", 0)),
        "is_judicial_state": 1 if property_state in _JUDICIAL_STATES else 0,
        "is_hltv_refi": 1 if kw.get("hltv_refi_option") == "Y" else 0,
    }


class PrepaymentModelInference:

    def __init__(self, model_dir: str, model_name: str = "xgb"):

        with open(os.path.join(model_dir, "metadata.json")) as fh:
            meta = json.load(fh)
        with open(os.path.join(model_dir, "preprocessor.pkl"), "rb") as fh:
            self.preprocessor = pickle.load(fh)
        with open(os.path.join(model_dir, f"{model_name}.pkl"), "rb") as fh:
            self.model = pickle.load(fh)

        self.num_feats = meta.get("numeric_features", NUMERIC_FEATURES)
        self.cat_feats = meta.get("categorical_features", CATEGORICAL_FEATURES)
        self.bin_feats = meta.get("binary_features", BINARY_FEATURES)

        self._weight = float(meta.get("weight_ratio", 1.0))
        self._correct_prior = model_name != "calibrated" and self._weight > 1.0
        self.model_name = model_name
        log.info("Loaded %s from %s (weight_ratio=%.2f)",
                 model_name, model_dir, self._weight)

    def _correct(self, p: np.ndarray) -> np.ndarray:

        if not self._correct_prior:
            return p
        w = self._weight
        return p / (w * (1.0 - p) + p)

    def predict_smm(self, loan_params_list: Iterable[dict]) -> np.ndarray:

        df = pd.DataFrame([build_feature_row(**lp) for lp in loan_params_list])
        for col in self.num_feats + self.bin_feats:
            if col in df.columns:
                df[col] = df[col].astype("float64")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Found unknown categories",
                category=UserWarning,
            )
            X = self.preprocessor.transform(df)
        probs = self.model.predict_proba(X)[:, 1]
        return self._correct(probs)
