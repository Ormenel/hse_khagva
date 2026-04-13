
import argparse
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

# ── sklearn ──────────────────────────────────────────────────────────────────
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder as SkOHE, StandardScaler

# ── Survival analysis ────────────────────────────────────────────────────────
from lifelines import CoxPHFitter, KaplanMeierFitter

# ── Optional: XGBoost ────────────────────────────────────────────────────────
import xgboost as xgb

# ── Optional: LightGBM ──────────────────────────────────────────────────────
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
_JUDICIAL_STATES = [
    "CT", "DE", "FL", "HI", "IL", "IN", "IA", "KS", "KY", "LA",
    "ME", "MD", "MA", "MN", "MO", "MT", "NE", "NJ", "NM", "NY",
    "ND", "OH", "OK", "PA", "RI", "SC", "SD", "VT", "WI",
]

# ============================================================================
#  1. SPARK PREPROCESSING
# ============================================================================

def create_spark(app="FannieMae_V3", driver_mem="24g"):
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder.appName(app)
        .config("spark.driver.memory",                          driver_mem)
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max",              "512m")
        .config("spark.sql.adaptive.enabled",                   "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled","true")
        .config("spark.sql.adaptive.skewJoin.enabled",          "true")
        .config("spark.sql.shuffle.partitions",                 "400")
        .config("spark.sql.autoBroadcastJoinThreshold",         "100m")
        .config("spark.sql.parquet.filterPushdown",             "true")
        .config("spark.memory.offHeap.enabled",                 "true")
        .config("spark.memory.offHeap.size",                    "8g")
        .getOrCreate()
    )


def create_spark_server(app="FannieMae_V3", master="yarn"):
    from pyspark.sql import SparkSession
    return (
        SparkSession.builder.appName(app).master(master)
        .config("spark.submit.deployMode",                      "client")
        .config("spark.dynamicAllocation.enabled",              "false")
        .config("spark.driver.memory",                          "20g")
        .config("spark.driver.cores",                           "4")
        .config("spark.executor.instances",                     "2")
        .config("spark.executor.cores",                         "12")
        .config("spark.executor.memory",                        "33g")
        .config("spark.executor.memoryOverhead",                "4g")
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max",              "512m")
        .config("spark.sql.adaptive.enabled",                   "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled","true")
        .config("spark.sql.adaptive.skewJoin.enabled",          "true")
        .config("spark.sql.shuffle.partitions",                 "200")
        .config("spark.sql.autoBroadcastJoinThreshold",         "200m")
        .config("spark.sql.parquet.filterPushdown",             "true")
        .getOrCreate()
    )


def load_panel_spark(
        spark,
        panel_path: str,
        fred_path: str,
    ):
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

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

