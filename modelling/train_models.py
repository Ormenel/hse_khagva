import json
import logging
import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pyarrow.dataset as ds

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder as SkOHE, StandardScaler

import xgboost as xgb
import lightgbm as lgb

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.1)


NUMERIC_FEATURES = [
    "fico", "orig_interest_rate", "current_interest_rate",
    "refi_incentive", "refi_incentive_pos",
    "rate_spread_to_10y", "spread_pos",
    "orig_ltv", "dti", "orig_upb", "upb_fraction", "equity_proxy",
    "loan_age", "age_sq", "burnout", "pct_term_elapsed",
    "orig_loan_term", "remaining_months_to_mat", "rate_duration",
    "burnout_x_refi", "fico_x_refi", "ltv_x_refi",
    "ph_delinq_count", "excess_principal", "gs10_monthly",
]

CATEGORICAL_FEATURES = [
    "channel", "loan_purpose", "property_type", "occupancy_status",
    "fico_bucket", "seasoning_bucket", "month_of_year", "vintage_year",
]

BINARY_FEATURES = [
    "high_ltv", "term_15y", "is_refi", "is_cashout", "is_io",
    "has_ppm", "modified", "is_investor", "is_high_bal",
    "first_time_buyer", "in_forbearance", "has_deferral",
    "is_judicial_state", "is_hltv_refi",
]

TARGET_COL = "smm_target"
TRAIN_CUTOFF = "2014-02-28"
SEED = 42

_JUDICIAL_STATES = ["CT", "DE", "FL", "HI", "IL", "IN", "IA", "KS", "KY", "LA",
                    "ME", "MD", "MA", "MN", "MO", "MT", "NE", "NJ", "NM", "NY",
                    "ND", "OH", "OK", "PA", "RI", "SC", "SD", "VT", "WI",
]

# ============================================================================
#  1. SPARK PREPROCESSING
# ============================================================================

def create_spark(app="FannieMae_V3", driver_mem="24g"):
    return (
        SparkSession.builder.appName(app)
        .config("spark.driver.memory", driver_mem)
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled","true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", "400")
        .config("spark.sql.autoBroadcastJoinThreshold", "100m")
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "8g")
        .getOrCreate()
    )


def create_spark_server(app="FannieMae_V3", master="yarn"):
    return (
        SparkSession.builder.appName(app).master(master)
        .config("spark.submit.deployMode", "client")
        .config("spark.dynamicAllocation.enabled", "false")
        .config("spark.driver.memory", "20g")
        .config("spark.driver.cores", "4")
        .config("spark.executor.instances", "4")
        .config("spark.executor.cores", "12")
        .config("spark.executor.memory", "33g")
        .config("spark.executor.memoryOverhead", "4g")
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled","true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.autoBroadcastJoinThreshold", "200m")
        .config("spark.sql.parquet.filterPushdown", "true")
        .getOrCreate()
    )


def load_panel_spark(
        spark,
        panel_path: str,
        fred_path: str,
    ):
    log.info("Loading panel Parquet from %s", panel_path)
    panel = spark.read.parquet(panel_path)
    log.info("Panel rows: {:,}".format(panel.count()))

    w = Window.partitionBy("loan_id").orderBy("reporting_date")
    df = (
        panel
        .withColumn("next_zbc", F.lead("zero_balance_code", 1).over(w))
        .withColumn(TARGET_COL, F.when(F.col("next_zbc") == "01", 1).otherwise(0))
        .filter(F.col("zero_balance_code").isNull())
        .filter(~F.col("current_delinquency_status")
                .isin("06", "07", "08", "09", "12", "XX"))
        .withColumn("upb_fraction",
            F.when(F.col("orig_upb") > 0,
                   F.col("current_actual_upb") / F.col("orig_upb")).otherwise(1.0))
        .withColumn("pct_term_elapsed",
            F.when(F.col("orig_loan_term") > 0,
                   F.col("loan_age") / F.col("orig_loan_term")).otherwise(0.0))
        .withColumn("excess_principal",
            F.greatest(F.col("total_principal") - F.col("scheduled_principal"), F.lit(0.0)))
        .withColumn("in_forbearance",
            F.when(F.isnan("in_forbearance") | F.col("in_forbearance").isNull(), 0)
             .otherwise(F.col("in_forbearance")))
        .withColumn("has_deferral",
            F.when(F.isnan("has_deferral") | F.col("has_deferral").isNull(), 0)
             .otherwise(F.col("has_deferral")))
        .withColumn("has_ppm",
            F.when(F.isnan("has_ppm") | F.col("has_ppm").isNull(), 1)
             .otherwise(F.col("has_ppm")))
        .withColumn("remaining_months_to_mat",
            F.coalesce(F.col("remaining_months_to_mat"),
                F.greatest(F.floor(F.months_between(F.col("maturity_dt"),
                    F.col("reporting_date"))), F.lit(0)).cast("int")))
    )

    # Derived features
    df = (df
        .withColumn("refi_incentive",
            F.col("orig_interest_rate") - F.col("current_interest_rate"))
        .withColumn("refi_incentive_pos",
            F.greatest(F.col("orig_interest_rate") - F.col("current_interest_rate"), F.lit(0.0)))
        .withColumn("burnout",
            F.col("loan_age") * F.greatest(
                F.col("orig_interest_rate") - F.col("current_interest_rate"), F.lit(0.0)))
        .withColumn("month_of_year", F.month("reporting_date").cast("string"))
        .withColumn("is_judicial_state",
            F.col("property_state").isin(_JUDICIAL_STATES).cast("int"))
        .withColumn("is_hltv_refi",
            F.when(F.col("hltv_refi_option") == "Y", 1).otherwise(0))
    )

    # FRED GS10 join
    log.info("Loading FRED GS10 from %s", fred_path)
    gs10 = (
        spark.read.option("header", "true").option("inferSchema", "false").csv(fred_path)
        .withColumn("month_date", F.trunc(F.to_date("observation_date", "yyyy-MM-dd"), "month"))
        .withColumn("gs10_monthly", F.col("GS10").cast("double"))
        .select("month_date", "gs10_monthly")
        .filter(F.col("month_date").isNotNull())
    )
    df = (df
        .withColumn("month_date", F.trunc("reporting_date", "month"))
        .join(F.broadcast(gs10), on="month_date", how="left")
        .withColumn("rate_spread_to_10y",
            F.when(F.col("current_interest_rate").isNotNull() &
                   F.col("gs10_monthly").isNotNull(),
                   F.col("current_interest_rate") - F.col("gs10_monthly")))
    )

    # Additional derived features
    df = (df
        .withColumn("spread_pos", F.greatest(F.col("rate_spread_to_10y"), F.lit(0.0)))
        .withColumn("age_sq", F.col("loan_age") * F.col("loan_age") / F.lit(100.0))
        .withColumn("rate_duration",
            F.when(F.col("remaining_months_to_mat").isNotNull() &
                   F.col("current_interest_rate").isNotNull(),
                   F.col("current_interest_rate") * F.col("remaining_months_to_mat") / F.lit(1200.0))
             .otherwise(F.lit(0.0)))
        .withColumn("burnout_x_refi",
            F.col("burnout") * F.col("refi_incentive_pos"))
        .withColumn("fico_x_refi",
            F.when(F.col("fico").isNotNull(),
                   F.col("fico") / F.lit(100.0) * F.col("refi_incentive_pos"))
             .otherwise(F.lit(0.0)))
        .withColumn("ltv_x_refi",
            F.when(F.col("orig_ltv").isNotNull(),
                   F.col("orig_ltv") * F.col("refi_incentive_pos"))
             .otherwise(F.lit(0.0)))
    )

    df = df.withColumn("vintage_year", F.year("origination_dt").cast("string"))

    log.info("Panel with target rows: {:,}".format(df.count()))
    return df


def export_raw_data(df, out_path: str):

    num_feats = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    bin_feats = [c for c in BINARY_FEATURES if c in df.columns]

    select_cols = (
        [TARGET_COL, "reporting_date", "origination_dt", "loan_id"]
        + num_feats + cat_feats + bin_feats
    )
    # Extra columns for Cox
    for c in ["loan_age", "fico_bucket"]:
        if c in df.columns and c not in select_cols:
            select_cols.append(c)

    available = [c for c in select_cols if c in df.columns]
    flat = df.select(available)

    # Split
    export_dir = os.path.join(out_path, "exported")
    os.makedirs(export_dir, exist_ok=True)

    train_path = os.path.join(export_dir, "train_raw.parquet")
    test_path = os.path.join(export_dir, "test_raw.parquet")

    train_df = flat.filter(F.col("origination_dt") < F.lit(TRAIN_CUTOFF))
    test_df = flat.filter(F.col("origination_dt") >= F.lit(TRAIN_CUTOFF))

    log.info("Exporting raw train set → %s", train_path)
    train_df.write.mode("overwrite").parquet(train_path)
    log.info("Exporting raw test set → %s", test_path)
    test_df.write.mode("overwrite").parquet(test_path)

    # Save column
    meta = {
        "numeric_features": num_feats,
        "categorical_features": cat_feats,
        "binary_features": bin_feats,
        "target_col": TARGET_COL,
        "train_cutoff": TRAIN_CUTOFF,
        "seed": SEED,
    }
    meta_path = os.path.join(export_dir, "columns.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info("Saved column metadata → %s", meta_path)


# ============================================================================
#  2: SKLEARN PREPROCESSING + MODEL FITTING
# ============================================================================

def read_first_n_rows(path, n=100_000):
    dataset = ds.dataset(path, format="parquet")

    batches = []
    total = 0

    for batch in dataset.to_batches(batch_size=n):
        pdf = batch.to_pandas()
        batches.append(pdf)

        total += len(pdf)
        if total >= n:
            break

    return pd.concat(batches, ignore_index=True).head(n)


def load_raw_data(out_path: str, sample_frac: float = 0.1):
    export_dir = os.path.join(out_path, "exported")

    with open(os.path.join(export_dir, "columns.json")) as fh:
        col_meta = json.load(fh)

    log.info("Loading raw train Parquet …")
    train_pd = pd.read_parquet(os.path.join(export_dir, "train_raw.parquet"))
    # train_pd = read_first_n_rows(os.path.join(export_dir, "train_raw.parquet"), 50_000_000)
    log.info("Loading raw test Parquet …")
    test_pd = pd.read_parquet(os.path.join(export_dir, "test_raw.parquet"))
    # test_pd = read_first_n_rows(os.path.join(export_dir, "test_raw.parquet"), 5_000_000)

    log.info("Full sizes: train={:,}, test={:,}".format(len(train_pd), len(test_pd)))

    if sample_frac < 1.0:
        train_pd = train_pd.sample(frac=sample_frac, random_state=SEED)
        test_pd = test_pd.sample(frac=sample_frac, random_state=SEED)
        log.info("After sampling (%.0f%%): train={:,}, test={:,}".format(
            len(train_pd), len(test_pd)) % (sample_frac * 100))

    return train_pd, test_pd, col_meta


def build_sklearn_preprocessor(train_pd, col_meta):
    num_feats = [c for c in col_meta["numeric_features"] if c in train_pd.columns]
    cat_feats = [c for c in col_meta["categorical_features"] if c in train_pd.columns]
    bin_feats = [c for c in col_meta["binary_features"] if c in train_pd.columns]

    numeric_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", SkOHE(drop="first", sparse_output=False, handle_unknown="infrequent_if_exist", min_frequency=0.001)),
    ])

    binary_pipe = SkPipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_feats),
        ("cat", categorical_pipe, cat_feats),
        ("bin", binary_pipe, bin_feats),
    ], remainder="drop")

    log.info("Fitting sklearn preprocessor (imputer + OHE + scaler) …")
    train_pd[bin_feats] = train_pd[bin_feats].astype(np.float32)
    preprocessor.fit(train_pd)

    # Features
    feature_names = list(num_feats)
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        cat_names = ohe.get_feature_names_out(cat_feats).tolist()
        feature_names += cat_names
    except:
        feature_names += [f"{c}_enc" for c in cat_feats]
    feature_names += list(bin_feats)

    log.info("Preprocessor fitted: %d features total", len(feature_names))
    return preprocessor, feature_names


def prepare_arrays(train_pd, test_pd, preprocessor):
    log.info("Transforming train set …")
    X_train = preprocessor.transform(train_pd).astype(np.float32)
    y_train = train_pd[TARGET_COL].values.astype(np.int32)

    log.info("Transforming test set …")
    test_pd[BINARY_FEATURES] = test_pd[BINARY_FEATURES].astype(np.float32)
    X_test = preprocessor.transform(test_pd).astype(np.float32)
    y_test = test_pd[TARGET_COL].values.astype(np.int32)

    log.info("Feature matrix: train=%s, test=%s (%.1f GB total)",
             X_train.shape, X_test.shape,
             (X_train.nbytes + X_test.nbytes) / 1e9)
    return X_train, y_train, X_test, y_test


def compute_weight_ratio(y_train):
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    ratio = n_neg / max(n_pos, 1)
    log.info("Class balance: n_neg=%d, n_pos=%d, weight_ratio=%.2f",n_neg, n_pos, ratio)
    return ratio

# ============================================================================
#  Logistic Regression
# ============================================================================

def train_logistic_regression(X_train, y_train, X_test, weight_ratio):
    log.info("Training Logistic Regression …")
    t0 = time.time()
    model = LogisticRegression(
        penalty="l2",
        C=1000.0,
        solver="saga",
        max_iter=200,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbose=1)
    model.fit(X_train, y_train)
    log.info("LR done in %.1f s", time.time() - t0)
    return model, model.predict_proba(X_test)[:, 1]

# ============================================================================
#  Random Forest
# ============================================================================

def train_random_forest(X_train, y_train, X_test, weight_ratio, n_sample=None):
    log.info("Training Random Forest …")
    t0 = time.time()

    # sampling
    if n_sample is not None:
        X_fit, y_fit = X_train[:n_sample], y_train[:n_sample]
        log.info("Using first %d rows for training", n_sample)
    else:
        X_fit, y_fit = X_train, y_train

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=50,
        max_features="sqrt",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbose=1)
    model.fit(X_fit, y_fit)
    log.info("RF done in %.1f s", time.time() - t0)
    return model, model.predict_proba(X_test)[:, 1]

# ============================================================================
#  SGD Model
# ============================================================================

def train_sgd(X_train, y_train, X_test, weight_ratio):
    log.info("Training SGD model …")
    t0 = time.time()
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbose=1)
    model.fit(X_train, y_train)
    log.info("SGD done in %.1f s", time.time() - t0)
    return model, model.predict_proba(X_test)[:, 1]

# ============================================================================
#  XGBoost
# ============================================================================

def train_xgboost(X_train, y_train, X_test, y_test, weight_ratio):
    log.info("Training XGBoost (n=%d) …", len(X_train))
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=weight_ratio,
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    log.info("XGBoost done in %.1f s", time.time() - t0)
    return model, model.predict_proba(X_test)[:, 1]

# ============================================================================
#  LightGBM
# ============================================================================

def train_lightgbm(X_train, y_train, X_test, weight_ratio):
    log.info("Training LightGBM DART (n=%d) …", len(X_train))
    t0 = time.time()
    sw = np.where(y_train == 1, weight_ratio, 1.0)
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        boosting_type="dart",
        random_state=SEED,
        n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=sw, callbacks=[lgb.log_evaluation(period=-1)])
    log.info("LightGBM done in %.1f s", time.time() - t0)
    return model, model.predict_proba(X_test)[:, 1]

# ============================================================================
#  GRID SEARCH
# ============================================================================

def _save_sklearn_grid_results(model_name: str, gs, out_path: str):
    df = pd.DataFrame(gs.cv_results_)
    cols = [c for c in df.columns
            if c.startswith("param_") or c in ("mean_test_score", "std_test_score", "rank_test_score")]
    df = df[cols].sort_values("rank_test_score")
    fname = f"grid_search_{model_name.lower().replace(' ', '_')}.csv"
    path = os.path.join(out_path, fname)
    df.to_csv(path, index=False)
    log.info("Saved %s grid results → %s", model_name, path)


def grid_search_xgboost(X_train, y_train, weight_ratio, out_path, sample_frac=0.05, sample_n=None):
    n_total = len(y_train)
    if n_total == 0:
        raise ValueError("X_train / y_train are empty")

    y_train = np.asarray(y_train).astype(np.int32)

    if sample_n is not None:
        n_sample = min(sample_n, n_total)
    else:
        n_sample = max(1, int(n_total * sample_frac))

    if n_sample < n_total:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=n_sample,
            random_state=SEED
        )
        sample_idx, _ = next(sss.split(np.zeros(n_total), y_train))
        X_gs = X_train[sample_idx]
        y_gs = y_train[sample_idx]
    else:
        X_gs = X_train
        y_gs = y_train

    if isinstance(X_gs, np.ndarray):
        X_gs = X_gs.astype(np.float32, copy=False)

    log.info("XGB grid sample: %s rows out of %s (%.2f%%)",
             len(y_gs), n_total, 100 * len(y_gs) / n_total)

    base = xgb.XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=weight_ratio,
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )

    param_grid = {
        "n_estimators": [400, 600, 800],
        "max_depth": [4, 6],
        "learning_rate": [0.01, 0.05],
    }

    n_cfg = (
        len(param_grid["n_estimators"])
        * len(param_grid["max_depth"])
        * len(param_grid["learning_rate"])
    )
    log.info("Grid search: XGBoost (%d configs × 3 folds) …", n_cfg)

    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=1,
        verbose=1,
        refit=True,
    )

    gs.fit(X_gs, y_gs)
    log.info("Best XGB: %s  AUC=%.4f", gs.best_params_, gs.best_score_)
    _save_sklearn_grid_results("XGBoost", gs, out_path)

    return gs


def grid_search_xgboost_random_ver2(X_train, y_train, weight_ratio, out_path,
                        sample_frac=0.1, sample_n=None, n_iter=40):
    from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, StratifiedKFold
    import numpy as np

    n_total = len(y_train)
    if n_total == 0:
        raise ValueError("X_train / y_train are empty")

    y_train = np.asarray(y_train).astype(np.int32)

    if sample_n is not None:
        n_sample = min(sample_n, n_total)
    else:
        n_sample = max(1, int(n_total * sample_frac))

    if n_sample < n_total:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=n_sample,
            random_state=SEED
        )
        sample_idx, _ = next(sss.split(np.zeros(n_total), y_train))
        X_gs = X_train[sample_idx]
        y_gs = y_train[sample_idx]
    else:
        X_gs = X_train
        y_gs = y_train

    if isinstance(X_gs, np.ndarray):
        X_gs = X_gs.astype(np.float32, copy=False)

    log.info("XGB search sample: %s rows out of %s (%.2f%%)",
             len(y_gs), n_total, 100 * len(y_gs) / n_total)
    log.info("Positive rate in sample: %.4f", y_gs.mean())

    base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=weight_ratio,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )

    param_dist = {
        "n_estimators": [700, 900, 1100, 1400, 1800],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "max_depth": [2, 3, 4],
        "min_child_weight": [2, 3, 4, 5, 6],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha": [0.001, 0.01, 0.05, 0.1],
        "reg_lambda": [7, 10, 15, 20],
        "gamma": [0, 0.05, 0.1, 0.2],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        random_state=SEED,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_gs, y_gs)
    log.info("Best XGB: %s  AUC=%.4f", search.best_params_, search.best_score_)
    _save_sklearn_grid_results("XGBoost_random_2", search, out_path)
    return search


def grid_search_lightgbm(X_train, y_train, weight_ratio, out_path, sample_frac=0.05, sample_n=None):
    n_total = len(y_train)
    if n_total == 0:
        raise ValueError("X_train / y_train are empty")

    y_train = np.asarray(y_train).astype(np.int32)

    if sample_n is not None:
        n_sample = min(sample_n, n_total)
    else:
        n_sample = max(1, int(n_total * sample_frac))

    if n_sample < n_total:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=n_sample,
            random_state=SEED
        )
        sample_idx, _ = next(sss.split(np.zeros(n_total), y_train))
        X_gs = X_train[sample_idx]
        y_gs = y_train[sample_idx]
    else:
        X_gs = X_train
        y_gs = y_train

    if isinstance(X_gs, np.ndarray):
        X_gs = X_gs.astype(np.float32, copy=False)

    sample_weights = np.where(y_gs == 1, weight_ratio, 1.0).astype(np.float32)

    log.info("LGB grid sample: %s rows out of %s (%.2f%%)",
             len(y_gs), n_total, 100 * len(y_gs) / n_total)

    base = lgb.LGBMClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        boosting_type="dart",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )

    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [4, 6],
        "num_leaves": [31, 63],
        "learning_rate": [0.01, 0.05, 0.1],
    }

    n_cfg = (
        len(param_grid["n_estimators"])
        * len(param_grid["max_depth"])
        * len(param_grid["num_leaves"])
        * len(param_grid["learning_rate"])
    )
    log.info("Grid search: LightGBM %d configs", n_cfg)

    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=1,
        verbose=1,
        refit=True,
    )

    gs.fit(
        X_gs,
        y_gs,
        sample_weight=sample_weights,
        callbacks=[lgb.log_evaluation(period=0)]
    )

    log.info("Best LGB: %s  AUC=%.4f", gs.best_params_, gs.best_score_)
    _save_sklearn_grid_results("LightGBM", gs, out_path)

    return gs

# ============================================================================
#  EVALUATION
# ============================================================================

def evaluate(y_true, y_prob, model_name: str) -> dict:
    auc = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    log.info("%-22s  AUC-ROC=%.4f  PR-AUC=%.4f  Brier=%.5f", model_name, auc, prauc, brier)
    return {"Model": model_name, "AUC-ROC": round(auc, 4),
            "PR-AUC": round(prauc, 4), "Brier": round(brier, 5)}

# ============================================================================
#  3. MODEL SAVING
# ============================================================================

def save_all(out_path, preprocessor=None, feature_names=None,
             lr_model=None, rf_model=None, sgd_model=None,
             xgb_model=None, lgb_model=None, cox_model=None,
             weight_ratio=1.0, col_meta=None):
    model_dir = os.path.join(out_path, "saved_models")
    os.makedirs(model_dir, exist_ok=True)

    if preprocessor is not None:
        p = os.path.join(model_dir, "preprocessor.pkl")
        with open(p, "wb") as fh:
            pickle.dump(preprocessor, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Saved sklearn preprocessor → %s", p)

    for name, model in [("lr", lr_model), ("rf", rf_model), ("sgd", sgd_model),
                        ("xgb", xgb_model), ("lgb", lgb_model), ("cox", cox_model)]:
        if model is not None:
            p = os.path.join(model_dir, f"{name}.pkl")
            with open(p, "wb") as fh:
                pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
            log.info("Saved %s → %s", name.upper(), p)

    meta = {
        "feature_names": feature_names or [],
        "weight_ratio": weight_ratio,
        "numeric_features": col_meta.get("numeric_features", []) if col_meta else [],
        "categorical_features": col_meta.get("categorical_features", []) if col_meta else [],
        "binary_features": col_meta.get("binary_features", []) if col_meta else [],
        "target_col": TARGET_COL,
        "train_cutoff": TRAIN_CUTOFF,
        "seed": SEED,
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info("Saved metadata.json")


def save_model(model, model_dir, name):
    if model is None:
        return
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{name}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Saved %s → %s", name.upper(), path)


def load_model(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)

# ============================================================================
#  VISUALISATIONS
# ============================================================================

def plot_roc_curves(results_list, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, yt, yp in results_list:
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(yt,yp):.4f})")
    ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curves – Prepayment Models V3")
    ax.legend(loc="lower right", fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(out_path, "roc_curves.png"), dpi=150); plt.close(fig)

def plot_pr_curves(results_list, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, yt, yp in results_list:
        prec, rec, _ = precision_recall_curve(yt, yp)
        ax.plot(rec, prec, lw=2, label=f"{name} (AP={average_precision_score(yt,yp):.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR Curves – Prepayment Models V3")
    ax.legend(loc="upper right", fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(out_path, "pr_curves.png"), dpi=150); plt.close(fig)

def plot_calibration(results_list, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0,1],[0,1],"k--",label="Perfect")
    for name, yt, yp in results_list:
        frac, mean = calibration_curve(yt, yp, n_bins=20, strategy="uniform")
        ax.plot(mean, frac, "s-", lw=1.5, ms=4, label=name)
    ax.set_xlabel("Mean Predicted"); ax.set_ylabel("Fraction Positive")
    ax.set_title("Calibration – V3"); ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(out_path, "calibration.png"), dpi=150); plt.close(fig)

def plot_feature_importance(model, feat_names, model_name, out_path, top_n=25):
    if model is None: return
    imp = model.feature_importances_
    n = min(len(imp), len(feat_names))
    fi = pd.Series(imp[:n], index=feat_names[:n]).nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(5, top_n*0.35)))
    fi.plot(kind="barh", ax=ax, edgecolor="black", lw=0.4)
    ax.set_title(f"{model_name} – Feature Importances (Top {top_n})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, f"fi_{model_name.lower().replace(' ','_')}.png"), dpi=150)
    plt.close(fig)

def plot_lr_coefficients(model, feat_names, out_path, top_n=25):
    if model is None: return
    coefs = model.coef_[0]
    n = min(len(coefs), len(feat_names))
    s = pd.Series(coefs[:n], index=feat_names[:n])
    top = s.abs().nlargest(top_n).index
    sel = s[top].sort_values()
    fig, ax = plt.subplots(figsize=(9, max(5, top_n*0.35)))
    colors = ["#e74c3c" if v<0 else "#2ecc71" for v in sel]
    sel.plot(kind="barh", ax=ax, color=colors, edgecolor="black", lw=0.4)
    ax.set_title(f"LR – Top {top_n} Coefficients"); fig.tight_layout()
    fig.savefig(os.path.join(out_path, "fi_logistic_regression.png"), dpi=150)
    plt.close(fig)

def plot_kaplan_meier(test_pd, out_path):
    if "fico_bucket" not in test_pd.columns or "loan_age" not in test_pd.columns:
        return
    pdf = test_pd[["loan_age", TARGET_COL, "fico_bucket"]].dropna().copy()
    pdf = pdf[pdf["loan_age"] > 0]
    pdf.rename(columns={TARGET_COL: "prepaid"}, inplace=True)
    palette = {"SubPrime":"#e74c3c","NearPrime":"#e67e22","Prime":"#3498db","SuperPrime":"#27ae60"}
    fig, ax = plt.subplots(figsize=(11, 6))
    for bucket, color in palette.items():
        grp = pdf[(pdf["fico_bucket"]==bucket) & pdf["prepaid"].isin([0,1])]
        if len(grp) < 50: continue
        kmf = KaplanMeierFitter()
        kmf.fit(grp["loan_age"].clip(lower=1), grp["prepaid"], label=f"{bucket} (n={len(grp):,})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, lw=2)
    ax.set_title("KM: Survival by FICO Tier"); ax.set_xlabel("Loan Age")
    ax.set_ylabel("P(Not Prepaid)"); ax.set_ylim(0,1)
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(out_path, "kaplan_meier.png"), dpi=150); plt.close(fig)


def plot_smm_cpr(test_pd, results_list, out_path):
    if "reporting_date" not in test_pd.columns: return
    agg = test_pd.groupby("reporting_date").agg(
        n=(TARGET_COL,"count"), n_prepaid=(TARGET_COL,"sum")).reset_index()
    agg["smm"] = agg["n_prepaid"]/agg["n"]
    agg["cpr"] = (1-(1-agg["smm"])**12)*100
    agg = agg.sort_values("reporting_date")
    fig,(a1,a2) = plt.subplots(2,1,figsize=(13,10),sharex=True)
    a1.plot(agg["reporting_date"],agg["smm"]*100,color="black",lw=2.5,label="Actual SMM")
    a2.plot(agg["reporting_date"],agg["cpr"],color="black",lw=2.5,label="Actual CPR")
    colors = ["#e74c3c","#3498db","#2ecc71","#9b59b6","#e67e22","#1abc9c","#f39c12","#8e44ad"]
    for (name,yt,yp),c in zip(results_list, colors):
        m = yp.mean()
        a1.axhline(m*100,ls="--",lw=1.2,color=c,label=f"{name} ({m*100:.2f}%)")
        a2.axhline((1-(1-m)**12)*100,ls="--",lw=1.2,color=c,label=f"{name}")
    a1.set_ylabel("SMM (%)"); a1.legend(fontsize=7)
    a2.set_ylabel("CPR (%)"); a2.set_xlabel("Date"); a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_path,"smm_cpr.png"),dpi=150); plt.close(fig)


def plot_model_comparison(results_df, out_path):
    metrics = ["AUC-ROC","PR-AUC","Brier"]; x = np.arange(len(results_df)); w=0.25
    fig, ax = plt.subplots(figsize=(12,6))
    for i,(m,c) in enumerate(zip(metrics,["#3498db","#2ecc71","#e74c3c"])):
        v = results_df[m].values
        bars = ax.bar(x+i*w, v, w, label=m, color=c, alpha=0.85)
        for b,val in zip(bars,v):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x+w); ax.set_xticklabels(results_df["Model"],rotation=20,ha="right")
    ax.set_title("Model Comparison V3"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_path,"model_comparison.png"),dpi=150); plt.close(fig)


def main(panel_path,
         abt_path,
         fred_path,
         out_path,
         skip_spark=False,
         sample_frac=0.1):
    os.makedirs(out_path, exist_ok=True)
    if not skip_spark:
        spark = create_spark()
        df = load_panel_spark(spark, panel_path, fred_path)
        export_raw_data(df, out_path)
        spark.stop()
        log.info("Spark stopped. All further processing is sklearn-only.")
    else:
        log.info("Skipping Spark")

    log.info("2. SKlearn preprocessing + base model fitting")

    train_pd, test_pd, col_meta = load_raw_data(out_path, sample_frac)
    preprocessor, feature_names = build_sklearn_preprocessor(train_pd, col_meta)
    X_train, y_train, X_test, y_test = prepare_arrays(train_pd, test_pd, preprocessor)
    weight_ratio = compute_weight_ratio(y_train)

    # MOdel fit
    lr_model, lr_p = train_logistic_regression(X_train, y_train, X_test, weight_ratio)
    rf_model, rf_p = train_random_forest(X_train, y_train, X_test, weight_ratio)
    sgd_model, sgd_p = train_sgd(X_train, y_train, X_test, weight_ratio)
    xgb_model, xgb_p = train_xgboost(X_train, y_train, X_test, y_test, weight_ratio)
    lgb_model, lgb_p = train_lightgbm(X_train, y_train, X_test, weight_ratio)

    # Evaluate
    m_lr  = evaluate(y_test, lr_p, "Logistic Regression")
    m_rf  = evaluate(y_test, rf_p, "Random Forest")
    m_sgd = evaluate(y_test, sgd_p, "SGD Classifier")
    m_xgb = evaluate(y_test, xgb_p, "XGBoost")
    m_lgb = evaluate(y_test, lgb_p, "LightGBM")
    all_m = [m_lr, m_rf, m_sgd, m_xgb, m_lgb]
    results_df = pd.DataFrame(all_m).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)

    log.info("3: Saving models")

    save_all(out_path, preprocessor=preprocessor, feature_names=feature_names,
             lr_model=lr_model, rf_model=rf_model, sgd_model=sgd_model,
             xgb_model=xgb_model, lgb_model=lgb_model, cox_model=cox_model,
             weight_ratio=weight_ratio, col_meta=col_meta)

    return results_df
