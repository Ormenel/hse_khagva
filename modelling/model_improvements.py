import json
import logging
import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    roc_curve,
)

from modelling.train_models import (
    SEED, load_raw_data, prepare_arrays,
    compute_weight_ratio, evaluate, load_model,
    plot_roc_curves, plot_pr_curves, plot_calibration,
    plot_model_comparison,
)

import xgboost as xgb
import lightgbm as lgb

from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ============================================================================
#  SMOTE OVERSAMPLING
# ============================================================================

def apply_smote(X_train, y_train, random_state=SEED):

    log.info("Applying SMOTE …")
    n_pos_before = np.sum(y_train == 1)
    sm = SMOTE(random_state=random_state, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    log.info("SMOTE: %d → %d positives (total %d → %d rows)",
             n_pos_before, np.sum(y_res == 1), len(y_train), len(y_res))
    return X_res, y_res

# ============================================================================
#  STACKING ENSEMBLE
# ============================================================================

def train_stacking_ensemble(X_train, y_train, X_test, weight_ratio):
    log.info("Training stacking ensemble …")
    t0 = time.time()

    estimators = [
        ("sgd", SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-4,
            max_iter=1000, tol=1e-3, class_weight="balanced",
            random_state=SEED, n_jobs=-1, verbose=0)),
        ("xgb", xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=weight_ratio, tree_method="hist",
            random_state=SEED, n_jobs=-1, verbosity=0)),
        ("lgb", lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            boosting_type="gbdt", random_state=SEED, n_jobs=-1, verbosity=-1)),
    ]

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, solver="lbfgs",
                                           max_iter=500, random_state=SEED),
        cv=3, stack_method="predict_proba", passthrough=False, n_jobs=-1)
    stack.fit(X_train, y_train)
    log.info("Stacking done in %.1f s", time.time() - t0)
    return stack, stack.predict_proba(X_test)[:, 1]


def train_stacking_ensemble_v2(X_train, y_train, X_test, weight_ratio):
    log.info("Training stacking ensemble …")
    t0 = time.time()

    estimators = [
        ("sgd", SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-4,
            max_iter=1000, tol=1e-3, class_weight="balanced",
            random_state=SEED, n_jobs=-1, verbose=0)),
        ("xgb", xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=weight_ratio, tree_method="hist",
            random_state=SEED, n_jobs=-1, verbosity=0)),
        ("cat", CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED,
            verbose=0,
            thread_count=-1
        )),

    ]

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=2,
            num_leaves=3,
            learning_rate=0.05,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1
        ),
        cv=3,
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1
    )
    stack.fit(X_train, y_train)
    log.info("Stacking done in %.1f s", time.time() - t0)
    return stack, stack.predict_proba(X_test)[:, 1]

# ============================================================================
#  SOFT VOTING ENSEMBLE
# ============================================================================

def train_voting_ensemble(X_test, fitted_models):
    log.info("Building soft voting ensemble (%d models) …", len(fitted_models))
    probs = []
    for name, model in fitted_models.items():
        if model is None:
            continue
        try:
            p = model.predict_proba(X_test)[:, 1]
            probs.append(p)
            log.info("  %s → mean=%.4f", name, p.mean())
        except Exception as e:
            log.warning("  %s failed: %s", name, e)

    if not probs:
        return None
    return np.mean(probs, axis=0)


def train_voting_ensemble_v2(X_test, fitted_models, weights=None, use_ranks=False):
    log.info("Building voting ensemble (%d models) …", len(fitted_models))

    preds = {}
    for name, model in fitted_models.items():
        if model is None:
            continue
        try:
            p = model.predict_proba(X_test)[:, 1]
            preds[name] = p
            log.info("  %s → mean=%.4f", name, p.mean())
        except Exception as e:
            log.warning("  %s failed: %s", name, e)

    if not preds:
        return None

    if weights is None:
        weights = {name: 1.0 for name in preds}

    total_weight = sum(weights.get(name, 0.0) for name in preds)
    if total_weight <= 0:
        raise ValueError("Sum of weights must be positive.")

    if use_ranks:
        from scipy.stats import rankdata
        ensemble = sum(
            weights.get(name, 0.0) * (rankdata(preds[name]) / len(preds[name]))
            for name in preds
        ) / total_weight
    else:
        ensemble = sum(
            weights.get(name, 0.0) * preds[name]
            for name in preds
        ) / total_weight

    return ensemble

# ============================================================================
#  ISOTONIC CALIBRATION
# ============================================================================

def calibrate_model(model, X_train, y_train, X_test, method="isotonic"):
    log.info("Calibrating with %s regression …", method)
    cal = CalibratedClassifierCV(model, cv=5, method=method)
    cal.fit(X_train, y_train)
    return cal, cal.predict_proba(X_test)[:, 1]

# ============================================================================
#  RUN ALL IMPROVEMENTS
# ============================================================================

def run_improvements(out_path, sample_frac=0.1):

    model_dir = os.path.join(out_path, "saved_models")
    preprocessor = load_model(os.path.join(model_dir, "preprocessor.pkl"))
    train_pd, test_pd, col_meta = load_raw_data(out_path, sample_frac)
    X_train, y_train, X_test, y_test = prepare_arrays(train_pd, test_pd, preprocessor)
    weight_ratio = compute_weight_ratio(y_train)

    models = {}
    for name in ["lr", "rf", "sgd", "xgb", "lgb"]:
        p = os.path.join(model_dir, f"{name}.pkl")
        if os.path.exists(p):
            models[name] = load_model(p)

    base_metrics = []
    base_curves = []
    for name, model in models.items():
        if model is None:
            continue
        probs = model.predict_proba(X_test)[:, 1]
        m = evaluate(y_test, probs, name.upper())
        base_metrics.append(m)
        base_curves.append((name.upper(), y_test, probs))

    # Stacking
    stack_model, stack_p = train_stacking_ensemble(
        X_train, y_train, X_test, weight_ratio)
    m_stack = evaluate(y_test, stack_p, "Stacking Ensemble")

    # Voting
    vote_p = train_voting_ensemble(X_test, models)
    m_vote = evaluate(y_test, vote_p, "Voting Ensemble") if vote_p is not None else None

    # Isotonic calibration
    best_name = "xgb"
    cal_model, cal_p, m_cal = None, None, None
    if best_name:
        cal_model, cal_p = calibrate_model(
            models[best_name], X_train, y_train, X_test)
        m_cal = evaluate(y_test, cal_p, f"{best_name.upper()} + Isotonic")

    # Save ensemble models
    for name, model in [("stacking", stack_model), ("calibrated", cal_model)]:
        if model is not None:
            p = os.path.join(model_dir, f"{name}.pkl")
            with open(p, "wb") as fh:
                pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
            log.info("Saved %s → %s", name, p)

    log.info("Improvements complete. All artefacts saved to %s", out_path)
