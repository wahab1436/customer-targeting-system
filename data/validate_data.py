"""
validate_data.py
----------------
Schema integrity checks, missing value audit, outlier detection,
and churn rate validation on the generated dataset.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "customer_id",
    "tenure",
    "monthly_usage_mb",
    "recharge_amount",
    "complaints_count",
    "last_activity_days",
    "region",
    "package_type",
    "usage_trend_30d",
    "recharge_freq_30d",
    "usage_per_rupee",
    "complaints_x_tenure",
    "received_offer",
    "churn",
]

NUMERIC_BOUNDS = {
    "tenure": (1, 60),
    "monthly_usage_mb": (0, 15000),
    "recharge_amount": (0, 6000),
    "complaints_count": (0, 30),
    "last_activity_days": (0, 90),
    "recharge_freq_30d": (0, 20),
}

VALID_REGIONS = {"Urban", "Semi-urban", "Rural"}
VALID_PACKAGES = {"Basic", "Standard", "Premium"}


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_schema(df: pd.DataFrame) -> bool:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Missing columns: %s", missing)
        return False
    logger.info("Schema check passed: all %d columns present.", len(EXPECTED_COLUMNS))
    return True


def check_missing_values(df: pd.DataFrame) -> bool:
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        logger.error("Missing values detected:\n%s", missing[missing > 0])
        return False
    logger.info("Missing value check passed: 0 nulls found.")
    return True


def check_numeric_bounds(df: pd.DataFrame) -> bool:
    all_ok = True
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range > 0:
            logger.warning(
                "Column '%s': %d values outside [%s, %s].",
                col, out_of_range, lo, hi,
            )
            all_ok = False
    if all_ok:
        logger.info("Numeric bounds check passed.")
    return all_ok


def check_categoricals(df: pd.DataFrame) -> bool:
    regions_ok = set(df["region"].unique()).issubset(VALID_REGIONS)
    packages_ok = set(df["package_type"].unique()).issubset(VALID_PACKAGES)
    if not regions_ok:
        logger.error("Unexpected region values: %s", set(df["region"].unique()))
    if not packages_ok:
        logger.error("Unexpected package_type values: %s", set(df["package_type"].unique()))
    if regions_ok and packages_ok:
        logger.info("Categorical check passed.")
    return regions_ok and packages_ok


def check_binary_columns(df: pd.DataFrame) -> bool:
    for col in ["received_offer", "churn"]:
        unique = set(df[col].unique())
        if not unique.issubset({0, 1}):
            logger.error("Column '%s' has non-binary values: %s", col, unique)
            return False
    logger.info("Binary column check passed.")
    return True


def check_churn_rate(df: pd.DataFrame) -> bool:
    churn_rate = df["churn"].mean()
    if not (0.10 <= churn_rate <= 0.60):
        logger.warning(
            "Churn rate %.2f%% is outside expected range [10%%, 60%%].",
            churn_rate * 100,
        )
        return False
    logger.info("Churn rate check passed: %.2f%%.", churn_rate * 100)
    return True


def check_duplicates(df: pd.DataFrame) -> bool:
    dupes = df["customer_id"].duplicated().sum()
    if dupes > 0:
        logger.error("%d duplicate customer_ids found.", dupes)
        return False
    logger.info("Duplicate check passed: all customer_ids unique.")
    return True


def outlier_summary(df: pd.DataFrame) -> None:
    numeric_cols = [
        "tenure", "monthly_usage_mb", "recharge_amount",
        "complaints_count", "last_activity_days",
    ]
    logger.info("Outlier summary (IQR method):")
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        logger.info("  %s: %d outliers (%.2f%%)", col, outliers, 100 * outliers / len(df))


def validate(data_path: str) -> bool:
    logger.info("Loading data from %s ...", data_path)
    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows x %d columns.", *df.shape)

    checks = [
        check_schema(df),
        check_missing_values(df),
        check_numeric_bounds(df),
        check_categoricals(df),
        check_binary_columns(df),
        check_churn_rate(df),
        check_duplicates(df),
    ]

    outlier_summary(df)

    if all(checks):
        logger.info("All validation checks passed.")
        return True
    else:
        logger.error("One or more validation checks failed.")
        return False


def main() -> None:
    config = load_config()
    ok = validate(config["data"]["output_path"])
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
