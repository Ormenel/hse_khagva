import argparse, logging, os, json, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve, brier_score_loss)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.1)


# Adding next month prepayment id
def build_panel_dataset(panel):
    w = Window.partitionBy("loan_id").orderBy("reporting_date")
    df = (panel
          .withColumn("next_zbc", F.lead("zero_balance_code",1).over(w))

          # TARGET: Погасится ли досрочно кредит в следующем месяце
          .withColumn("smm_target",
                      (F.col("next_zbc") == "01").cast("integer"))

          # Очистим от дефолтов, досрочки и т.д.
          .filter(F.col("zero_balance_code").isNull())
          .filter(F.col("next_zbc").isNotNull())

          # Исключим все нарушения
          .filter(~F.col("current_delinquency_status").isin("06","07","08","09","12","XX"))

          # UPB amortisation progress
          .withColumn("upb_fraction",
                      F.when(F.col("orig_upb")>0,
                             F.col("current_actual_upb") / F.col("orig_upb"))
                       .otherwise(1.0))

          .withColumn("excess_principal",
                      F.greatest(F.col("total_principal")-F.col("scheduled_principal"),
                                 F.lit(0.0)))

          .withColumn("pct_term_elapsed",
                      F.when(F.col("orig_loan_term")>0,
                             F.col("loan_age")/F.col("orig_loan_term"))
                       .otherwise(0.0))
    )
    log.info("Panel dataset rows: {:,}".format(df.count()))
    return df


def load_fred_monthly_csv(spark, path: str):
    log.info("Loading monthly FRED series from %s", path)

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv(path)
    )

    df = (
        df
        .withColumn("date", F.to_date("observation_date", "yyyy-MM-dd"))
        .withColumn(
            "gs10_monthly",
            F.col("GS10").cast("double")
        )
        .withColumn("month_date", F.trunc("date", "month"))
        .select("month_date", "gs10_monthly")
        .filter(F.col("month_date").isNotNull())
    )

    log.info("Loaded monthly rows: {:,}".format(df.count()))
    return df


def prepare_dataset(spark, panel, path_to_market_data):
    log.info("Preparing dataset")
    panel = build_panel_dataset(panel)
    panel = panel.withColumn("month_date", F.trunc("reporting_date", "month"))
    market_data = load_fred_monthly_csv(spark, path_to_market_data)

    out = (
        panel
        .join(F.broadcast(market_data), on="month_date", how="left")
        .withColumn(
            "rate_spread",
            F.when(
                F.col("current_interest_rate").isNotNull() &
                F.col("gs10_monthly").isNotNull(),
                F.col("current_interest_rate") - F.col("gs10_monthly")
            )
        )
    )

    return out
