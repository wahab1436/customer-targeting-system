"""
test_data.py
------------
Unit tests for data generation and validation modules.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_data import generate_data, load_config


@pytest.fixture
def config():
    cfg = load_config()
    cfg["data"]["n_customers"] = 500  # small for testing
    return cfg


@pytest.fixture
def sample_df(config):
    return generate_data(config)


def test_generate_data_row_count(config, sample_df):
    assert len(sample_df) == config["data"]["n_customers"]


def test_generate_data_columns(sample_df):
    expected = [
        "customer_id", "tenure", "monthly_usage_mb", "recharge_amount",
        "complaints_count", "last_activity_days", "region", "package_type",
        "usage_trend_30d", "recharge_freq_30d", "usage_per_rupee",
        "complaints_x_tenure", "received_offer", "churn",
    ]
    for col in expected:
        assert col in sample_df.columns, f"Missing column: {col}"


def test_no_null_values(sample_df):
    assert sample_df.isnull().sum().sum() == 0


def test_churn_is_binary(sample_df):
    assert set(sample_df["churn"].unique()).issubset({0, 1})


def test_received_offer_is_binary(sample_df):
    assert set(sample_df["received_offer"].unique()).issubset({0, 1})


def test_tenure_range(sample_df):
    assert sample_df["tenure"].min() >= 1
    assert sample_df["tenure"].max() <= 60


def test_last_activity_days_capped(sample_df):
    assert sample_df["last_activity_days"].max() <= 90


def test_churn_rate_reasonable(sample_df):
    churn_rate = sample_df["churn"].mean()
    assert 0.05 <= churn_rate <= 0.70, f"Churn rate {churn_rate:.2%} out of expected range"


def test_treatment_rate(config, sample_df):
    treatment_rate = sample_df["received_offer"].mean()
    expected = config["data"]["treatment_rate"]
    assert abs(treatment_rate - expected) < 0.10, (
        f"Treatment rate {treatment_rate:.2%} far from expected {expected:.2%}"
    )


def test_region_values(sample_df):
    assert set(sample_df["region"].unique()).issubset({"Urban", "Semi-urban", "Rural"})


def test_package_type_values(sample_df):
    assert set(sample_df["package_type"].unique()).issubset({"Basic", "Standard", "Premium"})


def test_unique_customer_ids(sample_df):
    assert sample_df["customer_id"].nunique() == len(sample_df)


def test_usage_per_rupee_non_negative(sample_df):
    assert (sample_df["usage_per_rupee"] >= 0).all()


def test_complaints_x_tenure_non_negative(sample_df):
    assert (sample_df["complaints_x_tenure"] >= 0).all()


def test_reproducibility(config):
    df1 = generate_data(config)
    df2 = generate_data(config)
    pd.testing.assert_frame_equal(df1, df2)
