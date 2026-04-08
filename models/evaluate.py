"""
evaluate.py
-----------
Evaluation utilities: AUC-ROC, Qini curve, SHAP plots.
Used by training scripts and Streamlit dashboard.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "AUC-ROC": round(roc_auc_score(y_true, y_proba), 4),
        "F1-Score": round(f1_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "PR-AUC": round(average_precision_score(y_true, y_proba), 4),
    }


def compute_shap_values(
    model,
    X: np.ndarray,
    sample_size: int = 500,
) -> Tuple[shap.TreeExplainer, np.ndarray]:
    explainer = shap.TreeExplainer(model)
    sample = X[:sample_size]
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return explainer, shap_values


def get_shap_importance_df(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    return df


def compute_qini_curve(
    uplift_scores: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    n = len(y)
    order = np.argsort(-uplift_scores)
    y_sorted = y[order]
    t_sorted = treatment[order]

    n_treated = t_sorted.cumsum()
    n_control = (1 - t_sorted).cumsum()
    treated_outcomes = (y_sorted * t_sorted).cumsum()
    control_outcomes = (y_sorted * (1 - t_sorted)).cumsum()

    with np.errstate(divide="ignore", invalid="ignore"):
        qini = np.where(
            n_treated > 0,
            treated_outcomes - control_outcomes * (n_treated / np.maximum(n_control, 1)),
            0.0,
        )

    x_axis = np.arange(1, n + 1) / n
    auuc = float(np.trapz(qini, x_axis))
    return x_axis, qini, auuc


def get_individual_shap(
    model,
    X_row: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_row.reshape(1, -1))
    if isinstance(sv, list):
        sv = sv[1]
    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv.flatten(),
    }).sort_values("shap_value", key=abs, ascending=False)
    return df
