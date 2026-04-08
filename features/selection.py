"""
selection.py
------------
Feature importance analysis and pruning using SHAP values
and built-in model importances.
"""

import logging
from typing import List

import numpy as np
import pandas as pd
import shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def compute_shap_importance(model, X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        importance = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame(
            {"feature": feature_names, "shap_importance": importance}
        ).sort_values("shap_importance", ascending=False)
        logger.info("SHAP importance computed for %d features.", len(feature_names))
        return df
    except Exception as e:
        logger.warning("SHAP computation failed: %s. Falling back to model importance.", e)
        return compute_model_importance(model, feature_names)


def compute_model_importance(model, feature_names: List[str]) -> pd.DataFrame:
    importances = model.feature_importances_
    df = pd.DataFrame(
        {"feature": feature_names, "shap_importance": importances}
    ).sort_values("shap_importance", ascending=False)
    logger.info("Model-based importance computed.")
    return df


def prune_features(
    importance_df: pd.DataFrame,
    threshold: float = 0.01,
) -> List[str]:
    max_imp = importance_df["shap_importance"].max()
    selected = importance_df[
        importance_df["shap_importance"] >= threshold * max_imp
    ]["feature"].tolist()
    logger.info(
        "Feature pruning: %d / %d features retained (threshold=%.2f%% of max).",
        len(selected),
        len(importance_df),
        threshold * 100,
    )
    return selected
