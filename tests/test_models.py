"""
test_models.py
--------------
Unit tests for churn and uplift model output shapes, ranges, and basic sanity checks.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_data import generate_data, load_config
from features.engineering import (
    prepare_X_y,
    build_preprocessing_pipeline,
    get_feature_columns,
)
from models.churn_model import train_xgboost, evaluate_model, apply_smote
from models.uplift_model import (
    split_treatment_control,
    compute_uplift_scores,
    classify_segments,
    compute_qini_curve,
)
from models.evaluate import compute_classification_metrics


@pytest.fixture
def config():
    cfg = load_config()
    cfg["data"]["n_customers"] = 400
    cfg["churn_model"]["xgboost"]["n_estimators"] = 20
    cfg["uplift_model"]["xgboost"]["n_estimators"] = 20
    return cfg


@pytest.fixture
def prepared_data(config):
    df = generate_data(config)
    X, y, treatment = prepare_X_y(df, config)
    numeric_cols, categorical_cols = get_feature_columns(config)
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    preprocessor.fit(X)
    X_t = preprocessor.transform(X)
    return X_t, y, treatment, df


def test_xgboost_predict_proba_shape(config, prepared_data):
    X_t, y, treatment, _ = prepared_data
    model = train_xgboost(X_t, y.values, config)
    proba = model.predict_proba(X_t)
    assert proba.shape == (len(y), 2)


def test_xgboost_proba_range(config, prepared_data):
    X_t, y, treatment, _ = prepared_data
    model = train_xgboost(X_t, y.values, config)
    proba = model.predict_proba(X_t)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_evaluate_model_keys(config, prepared_data):
    X_t, y, treatment, _ = prepared_data
    model = train_xgboost(X_t, y.values, config)
    metrics = evaluate_model(model, X_t, y.values)
    for key in ["auc_roc", "f1", "precision", "recall", "pr_auc"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_smote_increases_minority(prepared_data):
    X_t, y, _, _ = prepared_data
    original_churn = y.sum()
    X_res, y_res = apply_smote(X_t, y)
    assert y_res.sum() >= original_churn
    assert y_res.mean() > 0.4  # SMOTE should balance classes


def test_split_treatment_control_sizes(prepared_data):
    X_t, y, treatment, _ = prepared_data
    X_T, y_T, X_C, y_C = split_treatment_control(X_t, y, treatment)
    assert len(y_T) + len(y_C) == len(y)
    assert len(X_T) == len(y_T)
    assert len(X_C) == len(y_C)


def test_uplift_scores_shape(config, prepared_data):
    X_t, y, treatment, _ = prepared_data
    X_T, y_T, X_C, y_C = split_treatment_control(X_t, y, treatment)
    model_T = XGBClassifier(**config["uplift_model"]["xgboost"])
    model_T.fit(X_T, y_T)
    model_C = XGBClassifier(**config["uplift_model"]["xgboost"])
    model_C.fit(X_C, y_C)
    scores = compute_uplift_scores(model_T, model_C, X_t)
    assert scores.shape == (len(y),)


def test_uplift_scores_range(config, prepared_data):
    X_t, y, treatment, _ = prepared_data
    X_T, y_T, X_C, y_C = split_treatment_control(X_t, y, treatment)
    model_T = XGBClassifier(**config["uplift_model"]["xgboost"])
    model_T.fit(X_T, y_T)
    model_C = XGBClassifier(**config["uplift_model"]["xgboost"])
    model_C.fit(X_C, y_C)
    scores = compute_uplift_scores(model_T, model_C, X_t)
    assert scores.min() >= -1.0 and scores.max() <= 1.0


def test_classify_segments_valid_labels(prepared_data):
    X_t, y, treatment, _ = prepared_data
    fake_uplift = np.random.uniform(-0.2, 0.3, size=len(y))
    fake_churn = np.random.uniform(0, 1, size=len(y))
    segments = classify_segments(fake_uplift, fake_churn)
    valid = {"Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"}
    assert set(segments).issubset(valid)


def test_qini_curve_output(prepared_data):
    X_t, y, treatment, _ = prepared_data
    fake_uplift = np.random.uniform(-0.2, 0.3, size=len(y))
    x_axis, qini, auuc = compute_qini_curve(fake_uplift, y.values, treatment.values)
    assert len(x_axis) == len(qini) == len(y)
    assert isinstance(auuc, float)


def test_classification_metrics_range(prepared_data, config):
    X_t, y, _, _ = prepared_data
    model = train_xgboost(X_t, y.values, config)
    proba = model.predict_proba(X_t)[:, 1]
    metrics = compute_classification_metrics(y.values, proba)
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
