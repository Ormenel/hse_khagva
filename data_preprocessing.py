import argparse
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, DoubleType
)
from pyspark.sql.window import Window

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FANNIE_SCHEMA = StructType([
    StructField("reference_pool_id", StringType(), True),
    StructField("loan_id", StringType(), True),
    StructField("monthly_reporting_period", StringType(), True),
    StructField("channel", StringType(), True),
    StructField("seller_name", StringType(), True),
    StructField("servicer_name", StringType(), True),
    StructField("master_servicer", StringType(), True),
    StructField("orig_interest_rate", DoubleType(), True),
    StructField("current_interest_rate", DoubleType(), True),
    StructField("orig_upb", DoubleType(), True),
    StructField("upb_at_issuance", DoubleType(), True),
    StructField("current_actual_upb", DoubleType(), True),
    StructField("orig_loan_term", IntegerType(), True),
    StructField("origination_date", StringType(), True),
    StructField("first_payment_date", StringType(), True),
    StructField("loan_age", IntegerType(), True),
    StructField("remaining_months_legal_mat", IntegerType(), True),
    StructField("remaining_months_to_mat", IntegerType(), True),
    StructField("maturity_date", StringType(), True),
    StructField("orig_ltv", FloatType(), True),
    StructField("orig_cltv", FloatType(), True),
    StructField("num_borrowers", IntegerType(), True),
    StructField("dti", FloatType(), True),
    StructField("borrower_credit_score_orig", IntegerType(), True),
    StructField("coborrower_credit_score_orig", IntegerType(), True),
    StructField("first_time_homebuyer", StringType(), True),
    StructField("loan_purpose", StringType(), True),
    StructField("property_type", StringType(), True),
    StructField("num_units", IntegerType(), True),
    StructField("occupancy_status", StringType(), True),
    StructField("property_state", StringType(), True),
    StructField("msa", StringType(), True),
    StructField("zip_short", StringType(), True),
    StructField("mi_percent", FloatType(), True),
    StructField("amortization_type", StringType(), True),
    StructField("prepayment_penalty", StringType(), True),
    StructField("io_flag", StringType(), True),
    StructField("io_first_pi_date", StringType(), True),
    StructField("months_to_amortization", IntegerType(), True),
    StructField("current_delinquency_status", StringType(), True),
    StructField("loan_payment_history", StringType(), True),
    StructField("modification_flag", StringType(), True),
    StructField("mi_cancellation_indicator", StringType(), True),
    StructField("zero_balance_code", StringType(), True),
    StructField("zero_balance_effective_date", StringType(), True),
    StructField("upb_at_removal", DoubleType(), True),
    StructField("repurchase_date", StringType(), True),
    StructField("scheduled_principal", DoubleType(), True),
    StructField("total_principal", DoubleType(), True),
    StructField("unscheduled_principal", DoubleType(), True),
    StructField("last_paid_installment_date", StringType(), True),
    StructField("foreclosure_date", StringType(), True),
    StructField("disposition_date", StringType(), True),
    StructField("foreclosure_costs", DoubleType(), True),
    StructField("property_preservation_costs", DoubleType(), True),
    StructField("asset_recovery_costs", DoubleType(), True),
    StructField("misc_holding_expenses", DoubleType(), True),
    StructField("taxes_holding", DoubleType(), True),
    StructField("net_sales_proceeds", DoubleType(), True),
    StructField("credit_enhancement_proceeds", DoubleType(), True),
    StructField("repurchase_make_whole_proceeds", DoubleType(), True),
    StructField("other_foreclosure_proceeds", DoubleType(), True),
    StructField("mod_non_interest_bearing_upb", DoubleType(), True),
    StructField("principal_forgiveness", DoubleType(), True),
    StructField("orig_list_start_date", StringType(), True),
    StructField("orig_list_price", DoubleType(), True),
    StructField("current_list_start_date", StringType(), True),
    StructField("current_list_price", DoubleType(), True),
    StructField("borrower_credit_score_issuance", IntegerType(), True),
    StructField("coborrower_credit_score_issuance", IntegerType(), True),
    StructField("borrower_credit_score_current", IntegerType(), True),
    StructField("coborrower_credit_score_current", IntegerType(), True),
    StructField("mi_type", StringType(), True),
    StructField("servicing_activity_indicator", StringType(), True),
    StructField("current_period_mod_loss", DoubleType(), True),
    StructField("cumulative_mod_loss", DoubleType(), True),
    StructField("current_period_credit_event_net", DoubleType(), True),
    StructField("cumulative_credit_event_net", DoubleType(), True),
    StructField("special_eligibility_program", StringType(), True),
    StructField("foreclosure_principal_writeoff", DoubleType(), True),
    StructField("relocation_mortgage", StringType(), True),
    StructField("zbc_change_date", StringType(), True),
    StructField("loan_holdback_indicator", StringType(), True),
    StructField("loan_holdback_effective_date", StringType(), True),
    StructField("delinquent_accrued_interest", DoubleType(), True),
    StructField("property_valuation_method", StringType(), True),
    StructField("high_balance_loan", StringType(), True),
    StructField("arm_init_fixed_le5y", StringType(), True),
    StructField("arm_product_type", StringType(), True),
    StructField("initial_fixed_rate_period", IntegerType(), True),
    StructField("interest_rate_adj_freq", IntegerType(), True),
    StructField("next_rate_adj_date", StringType(), True),
    StructField("next_payment_change_date", StringType(), True),
    StructField("index", StringType(), True),
    StructField("arm_cap_structure", StringType(), True),
    StructField("init_rate_cap_up", FloatType(), True),
    StructField("periodic_rate_cap_up", FloatType(), True),
    StructField("lifetime_rate_cap_up", FloatType(), True),
    StructField("mortgage_margin", FloatType(), True),
    StructField("arm_balloon", StringType(), True),
    StructField("arm_plan_number", IntegerType(), True),
    StructField("borrower_assistance_plan", StringType(), True),
    StructField("hltv_refi_option", StringType(), True),
    StructField("deal_name", StringType(), True),
    StructField("repurchase_make_whole_flag", StringType(), True),
    StructField("alt_delinquency_resolution", StringType(), True),
    StructField("alt_delinquency_res_count", IntegerType(), True),
    StructField("total_deferral_amount", DoubleType(), True),
    StructField("payment_deferral_mod_event", StringType(), True),
    StructField("interest_bearing_upb", DoubleType(), True),
    StructField("orig_classic_fico", IntegerType(), True),
    StructField("issuance_classic_fico", IntegerType(), True),
    StructField("current_classic_fico", IntegerType(), True),
])

# ===========================================================================
# SPARK SESSION
# ===========================================================================

def create_spark(app: str = "FannieMae_CPR",
                 driver_mem: str = "24g",
                 shuffle: int = 400) -> SparkSession:
    return (
        SparkSession.builder.appName(app)
        .config("spark.driver.memory",                         driver_mem)
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max",             "512m")
        .config("spark.sql.adaptive.enabled",                  "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled","true")
        .config("spark.sql.adaptive.skewJoin.enabled",         "true")
        .config("spark.sql.shuffle.partitions",        str(shuffle))
        .config("spark.sql.autoBroadcastJoinThreshold",        "100m")
        .config("spark.sql.parquet.filterPushdown",            "true")
        .config("spark.memory.offHeap.enabled",                "true")
        .config("spark.memory.offHeap.size",                   "8g")
        .getOrCreate()
    )

# ===========================================================================
# LOAD CSV
# ===========================================================================
def load_raw(spark, path: str):
    log.info("Loading raw data: %s", path)
    df = (spark.read
          .option("sep", "|")
          .option("header", "false")
          .option("nullValue", "").schema(FANNIE_SCHEMA).csv(path))
    log.info("Raw rows in csv: {:,}".format(df.count()))

    return df

# ===========================================================================
# CLEANING ANDF LABELING
# ===========================================================================

def clean_and_label(df):
    # Просто лучший FICO
    df = df.withColumn(
        "fico",
        F.coalesce(
            F.col("orig_classic_fico").cast("float"),
            F.col("borrower_credit_score_orig").cast("float")
        )
    )

    df = (
        df
        # Date parsing
        .withColumn("reporting_date", F.to_date("monthly_reporting_period", "MMyyyy"))
        .withColumn("origination_dt", F.to_date("origination_date", "MMyyyy"))
        .withColumn("first_payment_dt", F.to_date("first_payment_date", "MMyyyy"))
        .withColumn(
            "zbc_effective_dt",
            F.when(
                F.col("zero_balance_effective_date").isNotNull(),
                F.to_date("zero_balance_effective_date", "MMyyyy")
            )
        )

        .withColumn(
            "maturity_dt",
            F.when(
                F.col("maturity_date").isNull(),
                F.when(
                    F.col("first_payment_dt").isNotNull() &
                    F.col("orig_loan_term").isNotNull(),
                    F.add_months(
                        F.col("first_payment_dt"),
                        F.col("orig_loan_term") - 1
                    )
                )
            ).otherwise(
                F.to_date("maturity_date", "MMyyyy")
            )
        )

        .withColumn(
            "loan_age",
            F.coalesce(
                F.col("loan_age"),
                F.when(
                    F.col("first_payment_dt").isNotNull() & F.col("reporting_date").isNotNull(),
                    F.floor(F.months_between(F.col("reporting_date"), F.col("first_payment_dt"))) + 1
                ).cast("int")
            )
        )

        .withColumn(
            "current_actual_upb",
            F.greatest(
                F.coalesce(F.col("current_actual_upb"), F.lit(0.0)),
                F.coalesce(F.col("upb_at_removal"), F.lit(0.0))
            )
        )

        .withColumn(
            "current_actual_upb",
            F.when(
                F.col("current_actual_upb") <= 0,
                F.col("orig_upb")
            ).otherwise(F.col("current_actual_upb"))
        )

        # Винтажи
        .withColumn("vintage_year", F.year("origination_dt"))
        .withColumn(
            "vintage_quarter",
            F.concat(
                F.year("origination_dt").cast("string"),
                F.lit("Q"),
                F.quarter("origination_dt").cast("string")
            )
        )

        # Только fixed rate
        .filter(F.col("amortization_type") == "FRM")
        .filter(F.col("orig_loan_term").isin(180, 360))
        .filter(F.col("fico").isNotNull() & F.col("fico").between(300, 850))
        .filter(F.col("orig_ltv").isNotNull() & F.col("orig_ltv").between(1, 200))
        .filter(F.col("dti").isNotNull() & F.col("dti").between(1, 65))
        .filter(F.col("orig_interest_rate").between(1.0, 20.0))
        .filter(F.col("orig_upb") > 0)
        .filter(F.col("loan_age") >= 0)

        # Prepay marking
        .withColumn("prepaid", (F.col("zero_balance_code") == "01").cast("int"))
        .withColumn(
            "default",
            F.col("zero_balance_code")
             .isin("02", "03", "09", "15", "97", "98")
             .cast("int")
        )
        .withColumn(
            "removed",
            F.col("zero_balance_code")
             .isin("06", "16", "96")
             .cast("int")
        )
        .withColumn("active", F.col("zero_balance_code").isNull().cast("int"))

        # FICO int -> segment
        .withColumn(
            "fico_bucket",
            F.when(F.col("fico") < 620, "SubPrime")
             .when(F.col("fico") < 680, "NearPrime")
             .when(F.col("fico") < 740, "Prime")
             .otherwise("SuperPrime")
        )

        # High LTV
        .withColumn("high_ltv", (F.col("orig_ltv") > 80).cast("int"))
        .withColumn("term_15y", (F.col("orig_loan_term") <= 180).cast("int"))
        .withColumn(
            "is_refi",
            F.col("loan_purpose").isin("C", "R", "U").cast("int")
        )
        .withColumn("is_cashout", (F.col("loan_purpose") == "C").cast("int"))
        .withColumn("is_io", (F.col("io_flag") == "Y").cast("int"))
        .withColumn("has_ppm", (F.col("prepayment_penalty") == "Y").cast("int"))
        .withColumn("modified", (F.col("modification_flag") == "Y").cast("int"))
        .withColumn("is_investor", (F.col("occupancy_status") == "I").cast("int"))
        .withColumn("is_high_bal", (F.col("high_balance_loan") == "Y").cast("int"))
        .withColumn(
            "first_time_buyer",
            (F.col("first_time_homebuyer") == "Y").cast("int")
        )
        .withColumn(
            "in_forbearance",
            F.col("borrower_assistance_plan")
             .isin("F", "R", "T", "O")
             .cast("int")
        )
        .withColumn("has_deferral", (F.col("total_deferral_amount") > 0).cast("int"))
        .withColumn(
            "rate_spread",
            F.col("orig_interest_rate") - F.col("current_interest_rate")
        )
        .withColumn(
            "equity_proxy",
            F.lit(1.0) - F.col("orig_ltv") / F.lit(100.0)
        )
        .withColumn(
            "delinquency_months",
            F.when(F.col("current_delinquency_status") == "XX", None)
             .otherwise(F.col("current_delinquency_status").cast("int"))
        )
        .withColumn(
            "seasoning_bucket",
            F.when(F.col("loan_age") <= 12, "0-12m")
             .when(F.col("loan_age") <= 36, "13-36m")
             .when(F.col("loan_age") <= 60, "37-60m")
             .when(F.col("loan_age") <= 120, "61-120m")
             .otherwise("120m+")
        )

        # Payment history score
        .withColumn(
            "ph_delinq_count",
            F.when(
                F.col("loan_payment_history").isNotNull(),
                (
                    F.length("loan_payment_history") / 2
                    - F.length(
                        F.regexp_replace("loan_payment_history", "00", "")
                    ) / 2
                ).cast("integer")
            ).otherwise(0)
        )
        .withColumn(
            "ph_recent_delinq",
            F.when(
                F.length("loan_payment_history") >= 6,
                (
                    F.regexp_replace(
                        F.substring("loan_payment_history", -6, 6),
                        "00",
                        ""
                    ) != ""
                ).cast("int")
            ).otherwise(0)
        )

        .withColumn(
            "upb_fraction",
            F.when(
                F.col("orig_upb") > 0,
                F.col("current_actual_upb") / F.col("orig_upb")
            )
        )
        # Excess principal for current month
        .withColumn(
            "excess_principal",
            F.greatest(
                F.col("total_principal") - F.col("scheduled_principal"),
                F.lit(0.0)
            )
        )
        # standard burnout
        .withColumn(
            "burnout",
            F.col("loan_age") * F.greatest(F.col("rate_spread"), F.lit(0.0))
        )
    )

    df = df.drop(
        "seller_name",
        "servicer_name",
        "master_servicer",
        "upb_at_issuance",
        "months_to_amortization",
        "io_first_pi_date",
        "arm_init_fixed_le5y",
        "arm_product_type",
        "initial_fixed_rate_period",
        "interest_rate_adj_freq",
        "next_rate_adj_date",
        "next_payment_change_date",
        "index",
        "arm_cap_structure",
        "init_rate_cap_up",
        "periodic_rate_cap_up",
        "lifetime_rate_cap_up",
        "mortgage_margin",
        "arm_balloon",
        "arm_plan_number",
        "deal_name",
        "current_period_mod_loss",
        "cumulative_mod_loss",
        "current_period_credit_event_net",
        "cumulative_credit_event_net",
        "orig_list_start_date",
        "orig_list_price",
        "current_list_start_date",
        "current_list_price",
        "borrower_credit_score_issuance",
        "coborrower_credit_score_issuance",
        "borrower_credit_score_current",
        "coborrower_credit_score_current"
    )

    w_rate = (
        Window.partitionBy("loan_id")
        .orderBy("reporting_date")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    df = df.withColumn(
        "current_interest_rate",
        F.coalesce(
            F.last("current_interest_rate", ignorenulls=True).over(w_rate),
            F.col("orig_interest_rate")
        )
    )

    log.info("Clean rows: {:,}".format(df.count()))
    return df


# Could be useful later
def build_abt(df):
    w = Window.partitionBy("loan_id").orderBy(F.desc("reporting_date"))
    abt = (df.withColumn("_rn", F.row_number().over(w))
             .filter(F.col("_rn") == 1).drop("_rn"))
    log.info("ABT rows (one per loan): {:,}".format(abt.count()))
    return abt


def save(df, path: str):
    log.info("Writing Parquet → %s", path)
    (df.write.mode("overwrite")
       .option("compression","snappy")
       .parquet(path))
    log.info("Done.")


def main(data_path, out_path):
    spark = create_spark()
    raw = load_raw(spark, data_path)
    cleaned = clean_and_label(raw)
    save(cleaned, out_path + "/panel")
    abt = build_abt(cleaned)
    save(abt, out_path + "/abt")
    cleaned.groupBy("zero_balance_code").count().orderBy("zero_balance_code").show()
    spark.stop()


DEFAULT_RAW = "D:/HSE/Diplom/raw_data"
DEFAULT_OUT = "D:/HSE/Diplom/processed_data"
#  C:\Users\paine\.conda\envs\thesis\python.exe data_preprocessing.py --data_path D:/HSE/Diplom/raw_data --out_path D:/HSE/Diplom/processed_data

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DEFAULT_RAW)
    p.add_argument("--out_path",  default=DEFAULT_OUT)
    args = p.parse_args()
    main(args.data_path, args.out_path)
