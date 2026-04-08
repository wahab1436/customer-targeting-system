"""
churn_model.py
--------------
Trains XGBoost (primary) and LightGBM (comparison) churn models.
Applies SMOTE for class imbalance. 5-fold stratified cross-validation.
Hyperparameter tuning with Optuna. SHAP explainability. Saves artifacts.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from features.engineering import load_raw_data, prepare_X_y, get_feature_columns
from features.pipeline import load_preprocessor, get_feature_names

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_smote(
    X: np.ndarray,
    y: pd.Series,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    logger.info(
        "SMOTE applied: %d -> %d samples | churn rate: %.2f%% -> %.2f%%.",
        len(y),
        len(y_res),
        y.mean() * 100,
        y_res.mean() * 100,
    )
    return X_res, y_res


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "auc_roc": roc_auc_score(y, proba),
        "f1": f1_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "pr_auc": average_precision_score(y, proba),
    }


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probas = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    preds = (probas >= 0.5).astype(int)
    metrics = {
        "cv_auc_roc": roc_auc_score(y, probas),
        "cv_f1": f1_score(y, preds),
        "cv_precision": precision_score(y, preds),
        "cv_recall": recall_score(y, preds),
        "cv_pr_auc": average_precision_score(y, probas),
    }
    logger.info("Cross-validation results: %s", {k: round(v, 4) for k, v in metrics.items()})
    return metrics


def tune_xgboost(X: np.ndarray, y: np.ndarray, n_trials: int = 30) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        model = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        probas = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        return roc_auc_score(y, probas)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Optuna best AUC-ROC: %.4f | params: %s", study.best_value, study.best_params)
    return study.best_params


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
    tune: bool = False,
) -> XGBClassifier:
    if tune:
        logger.info("Running Optuna hyperparameter tuning (30 trials)...")
        best_params = tune_xgboost(X, y, n_trials=30)
        params = {**config["churn_model"]["xgboost"], **best_params}
    else:
        params = config["churn_model"]["xgboost"]

    model = XGBClassifier(**params)
    model.fit(X, y)
    logger.info("XGBoost churn model trained.")
    return model


def train_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
) -> lgb.LGBMClassifier:
    params = config["churn_model"]["lightgbm"]
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    logger.info("LightGBM churn model trained.")
    return model


def compute_and_log_shap(
    model: XGBClassifier,
    X: np.ndarray,
    feature_names: list,
    sample_size: int = 500,
) -> None:
    sample = X[:sample_size]
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        importance = np.abs(shap_values).mean(axis=0)
        top = sorted(zip(feature_names, importance), key=lambda x: -x[1])[:5]
        logger.info("Top 5 SHAP features: %s", top)
    except Exception as e:
        logger.warning("SHAP computation error: %s", e)


def save_model(model, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    logger.info("Model saved to %s.", p)


def add_churn_scores_to_data(
    df: pd.DataFrame,
    model,
    preprocessor,
    config: dict,
) -> pd.DataFrame:
    numeric_cols = config["features"]["numeric_columns"]
    categorical_cols = config["features"]["categorical_columns"]
    X = df[numeric_cols + categorical_cols]
    X_t = preprocessor.transform(X)
    df = df.copy()
    df["churn_prob"] = model.predict_proba(X_t)[:, 1]
    return df


def main() -> None:
    config = load_config()
    df = load_raw_data(config["data"]["output_path"])
    X, y, treatment = prepare_X_y(df, config)

    preprocessor = load_preprocessor()
    if preprocessor is None:
        logger.error("Run features/engineering.py first.")
        return

    X_transformed = preprocessor.transform(X)
    feature_names = get_feature_names(preprocessor)

    # SMOTE
    X_res, y_res = apply_smote(
        X_transformed, y,
        random_state=config["churn_model"]["smote_random_state"],
    )

    # Train XGBoost
    xgb_model = train_xgboost(X_res, y_res, config, tune=False)
    xgb_metrics = evaluate_model(xgb_model, X_transformed, y.values)
    logger.info("XGBoost test metrics: %s", {k: round(v, 4) for k, v in xgb_metrics.items()})
    cross_validate_model(xgb_model, X_transformed, y.values)
    compute_and_log_shap(xgb_model, X_transformed, feature_names)
    save_model(xgb_model, config["churn_model"]["artifact_path"])

    # Train LightGBM
    lgb_model = train_lightgbm(X_res, y_res, config)
    lgb_metrics = evaluate_model(lgb_model, X_transformed, y.values)
    logger.info("LightGBM test metrics: %s", {k: round(v, 4) for k, v in lgb_metrics.items()})
    save_model(lgb_model, config["churn_model"]["lgbm_artifact_path"])

    # Save churn scores to data
    df_with_scores = add_churn_scores_to_data(df, xgb_model, preprocessor, config)
    df_with_scores.to_csv(config["data"]["output_path"], index=False)
    logger.info("Churn scores written back to dataset.")


if __name__ == "__main__":
    main()
