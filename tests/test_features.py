"""
test_features.py
----------------
Unit tests for feature engineering pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_data import generate_data, load_config
from features.engineering import (
    prepare_X_y,
    build_preprocessing_pipeline,
    get_feature_columns,
    apply_log_transforms,
)


@pytest.fixture
def config():
    cfg = load_config()
    cfg["data"]["n_customers"] = 300
    return cfg


@pytest.fixture
def sample_df(config):
    return generate_data(config)


def test_prepare_X_y_shapes(config, sample_df):
    X, y, treatment = prepare_X_y(sample_df, config)
    assert len(X) == len(sample_df)
    assert len(y) == len(sample_df)
    assert len(treatment) == len(sample_df)


def test_prepare_X_y_no_target_in_X(config, sample_df):
    X, y, treatment = prepare_X_y(sample_df, config)
    assert "churn" not in X.columns
    assert "received_offer" not in X.columns


def test_y_is_binary(config, sample_df):
    _, y, _ = prepare_X_y(sample_df, config)
    assert set(y.unique()).issubset({0, 1})


def test_treatment_is_binary(config, sample_df):
    _, _, treatment = prepare_X_y(sample_df, config)
    assert set(treatment.unique()).issubset({0, 1})


def test_get_feature_columns(config):
    numeric_cols, categorical_cols = get_feature_columns(config)
    assert isinstance(numeric_cols, list)
    assert isinstance(categorical_cols, list)
    assert len(numeric_cols) > 0
    assert len(categorical_cols) > 0


def test_build_preprocessing_pipeline(config, sample_df):
    X, y, _ = prepare_X_y(sample_df, config)
    numeric_cols, categorical_cols = get_feature_columns(config)
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] > len(numeric_cols)  # OHE expands categoricals


def test_no_nans_after_transform(config, sample_df):
    X, _, _ = prepare_X_y(sample_df, config)
    numeric_cols, categorical_cols = get_feature_columns(config)
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    preprocessor.fit(X)
    X_t = preprocessor.transform(X)
    assert not np.isnan(X_t).any()


def test_log_transform_adds_columns(sample_df):
    df_out = apply_log_transforms(sample_df)
    assert "tenure_log" in df_out.columns
    assert "recharge_amount_log" in df_out.columns
    assert (df_out["tenure_log"] >= 0).all()
    assert (df_out["recharge_amount_log"] >= 0).all()


def test_standard_scaler_applied(config, sample_df):
    X, _, _ = prepare_X_y(sample_df, config)
    numeric_cols, categorical_cols = get_feature_columns(config)
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    preprocessor.fit(X)
    X_t = preprocessor.transform(X)
    # Scaled numeric features should have mean near 0 and std near 1
    numeric_slice = X_t[:, :len(numeric_cols)]
    assert abs(numeric_slice.mean()) < 1.0
