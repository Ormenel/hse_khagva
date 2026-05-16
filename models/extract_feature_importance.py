from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


def load_pickle(path: Path):
    with open(path, "rb") as f:
            return pickle.load(f)


def load_feature_names(meta_path: Path) -> list[str]:
    with open(meta_path) as f:
        meta = json.load(f)
    return meta["feature_names"]


def linear_coefficients(model, feature_names: list[str]) -> pd.DataFrame:
    coef = np.ravel(model.coef_).astype(float)
    df = pd.DataFrame({
        "feature": feature_names[: len(coef)],
        "coef": coef,
        "abs_coef": np.abs(coef),
    })
    df["abs_coef_norm"] = df["abs_coef"] / max(df["abs_coef"].sum(), 1e-12)
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def tree_importance(model, feature_names: list[str]) -> pd.DataFrame:

    imp = np.asarray(model.feature_importances_, dtype=float)
    df = pd.DataFrame({
        "feature": feature_names[: len(imp)],
        "importance": imp,
        "importance_norm": imp / max(imp.sum(), 1e-12),
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def xgb_native_importance(model, feature_names: list[str]) -> pd.DataFrame:

    booster = model.get_booster()
    df = pd.DataFrame({"feature": feature_names})
    for itype in ("weight", "gain", "cover", "total_gain"):
        scores = booster.get_score(importance_type=itype)
        col = np.zeros(len(feature_names), dtype=float)
        for k, v in scores.items():
            try:
                idx = int(k.lstrip("f"))
            except ValueError:
                continue
            if 0 <= idx < len(col):
                col[idx] = v
        df[itype] = col
    df["gain_norm"] = df["gain"] / max(df["gain"].sum(), 1e-12)
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df

def build_combined(out_dir: Path, feature_names: list[str]) -> pd.DataFrame:

    sources = {
        "lr":  ("feature_importance_logistic_regression.csv", "abs_coef_norm"),
        "sgd": ("feature_importance_sgd.csv",                "abs_coef_norm"),
        "rf":  ("feature_importance_random_forest.csv",      "importance_norm"),
        "xgb": ("feature_importance_xgboost.csv",            "importance_norm"),
        "lgb": ("feature_importance_lightgbm.csv",           "importance_norm"),
    }

    df = pd.DataFrame({"feature": feature_names})
    for short, (fname, col) in sources.items():
        path = out_dir / fname
        if not path.exists():
            continue
        s = pd.read_csv(path).set_index("feature")[col]
        df[short] = df["feature"].map(s).fillna(0.0)

    cols = [c for c in sources if c in df.columns]
    df["sum_norm"] = df[cols].sum(axis=1)
    df = df.sort_values("sum_norm", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df
