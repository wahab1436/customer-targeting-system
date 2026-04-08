"""
uplift_model.py
---------------
Two-model uplift approach:
  Model T: trained on treated group (received_offer=1)
  Model C: trained on control group (received_offer=0)
  Uplift Score = P(no churn | T) - P(no churn | C)
Classifies customers into 4 segments. Evaluates via Qini curve + AUUC.
"""

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from features.engineering import load_raw_data, prepare_X_y, get_feature_columns
from features.pipeline import load_preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def split_treatment_control(
    X: np.ndarray,
    y: pd.Series,
    treatment: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    treated_mask = treatment.values == 1
    control_mask = treatment.values == 0

    X_T = X[treated_mask]
    y_T = y.values[treated_mask]
    X_C = X[control_mask]
    y_C = y.values[control_mask]

    logger.info(
        "Split: treated=%d (churn=%.1f%%), control=%d (churn=%.1f%%).",
        len(y_T), y_T.mean() * 100,
        len(y_C), y_C.mean() * 100,
    )
    return X_T, y_T, X_C, y_C


def train_uplift_model(
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
    group_name: str,
) -> XGBClassifier:
    params = config["uplift_model"]["xgboost"]
    model = XGBClassifier(**params)
    model.fit(X, y)
    train_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    logger.info("Uplift Model %s trained. Train AUC-ROC: %.4f.", group_name, train_auc)
    return model


def compute_uplift_scores(
    model_T: XGBClassifier,
    model_C: XGBClassifier,
    X_all: np.ndarray,
) -> np.ndarray:
    # Uplift = P(retain | treated) - P(retain | control)
    # = P(no churn | T) - P(no churn | C)
    # = (1 - P(churn | T)) - (1 - P(churn | C))
    # = P(churn | C) - P(churn | T)
    p_churn_T = model_T.predict_proba(X_all)[:, 1]
    p_churn_C = model_C.predict_proba(X_all)[:, 1]
    uplift_scores = p_churn_C - p_churn_T
    logger.info(
        "Uplift scores computed. Min=%.4f, Max=%.4f, Mean=%.4f.",
        uplift_scores.min(), uplift_scores.max(), uplift_scores.mean(),
    )
    return uplift_scores


def classify_segments(
    uplift_scores: np.ndarray,
    churn_probs: np.ndarray,
    persuadable_threshold: float = 0.05,
    sleeping_dog_threshold: float = -0.02,
) -> np.ndarray:
    segments = np.empty(len(uplift_scores), dtype=object)

    high_churn = churn_probs > 0.5

    segments[uplift_scores >= persuadable_threshold] = "Persuadable"
    segments[
        (uplift_scores > sleeping_dog_threshold)
        & (uplift_scores < persuadable_threshold)
    ] = "Sure Thing"
    segments[
        (uplift_scores <= sleeping_dog_threshold) & high_churn
    ] = "Lost Cause"
    segments[
        (uplift_scores <= sleeping_dog_threshold) & ~high_churn
    ] = "Sleeping Dog"

    counts = pd.Series(segments).value_counts()
    logger.info("Segment distribution:\n%s", counts.to_string())
    return segments


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

    n_treated_total = t_sorted.sum()
    n_control_total = (1 - t_sorted).sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        qini = np.where(
            n_treated > 0,
            treated_outcomes - control_outcomes * (n_treated / np.maximum(n_control, 1)),
            0.0,
        )

    x_axis = np.arange(1, n + 1) / n
    auuc = float(np.trapz(qini, x_axis))

    logger.info("Qini AUUC: %.4f.", auuc)
    return x_axis, qini, auuc


def save_models(model_T, model_C, config: dict) -> None:
    for model, path_key in [(model_T, "artifact_path_T"), (model_C, "artifact_path_C")]:
        p = Path(config["uplift_model"][path_key])
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, p)
        logger.info("Uplift model saved to %s.", p)


def save_scores(df: pd.DataFrame, config: dict) -> None:
    path = Path(config["uplift_model"]["scores_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Uplift scores saved to %s.", path)


def main() -> None:
    config = load_config()
    df = load_raw_data(config["data"]["output_path"])
    X, y, treatment = prepare_X_y(df, config)

    preprocessor = load_preprocessor()
    if preprocessor is None:
        logger.error("Run features/engineering.py first.")
        return

    X_all = preprocessor.transform(X)
    X_T, y_T, X_C, y_C = split_treatment_control(X_all, y, treatment)

    model_T = train_uplift_model(X_T, y_T, config, group_name="T (Treated)")
    model_C = train_uplift_model(X_C, y_C, config, group_name="C (Control)")

    uplift_scores = compute_uplift_scores(model_T, model_C, X_all)

    churn_probs = df["churn_prob"].values if "churn_prob" in df.columns else (
        model_C.predict_proba(X_all)[:, 1]
    )

    segments = classify_segments(
        uplift_scores,
        churn_probs,
        config["uplift_model"]["persuadable_threshold"],
        config["uplift_model"]["sleeping_dog_threshold"],
    )

    x_axis, qini, auuc = compute_qini_curve(
        uplift_scores, y.values, treatment.values
    )

    save_models(model_T, model_C, config)

    df_out = df.copy()
    df_out["uplift_score"] = uplift_scores
    df_out["uplift_segment"] = segments
    save_scores(df_out, config)

    # Write back to main data file
    df_out.to_csv(config["data"]["output_path"], index=False)
    logger.info("Uplift scores and segments written to main dataset.")


if __name__ == "__main__":
    main()
