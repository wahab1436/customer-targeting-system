"""
engineering.py
--------------
Feature engineering pipeline. Applies encoding, scaling, and produces
13 ML-ready features. Saves sklearn Pipeline to disk.
"""

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s.", len(df), path)
    return df


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenure_log"] = np.log1p(df["tenure"])
    df["recharge_amount_log"] = np.log1p(df["recharge_amount"])
    logger.info("Log transforms applied: tenure_log, recharge_amount_log.")
    return df


def build_preprocessing_pipeline(
    numeric_cols: list,
    categorical_cols: list,
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_columns(config: dict) -> Tuple[list, list]:
    numeric_cols = config["features"]["numeric_columns"]
    categorical_cols = config["features"]["categorical_columns"]
    return numeric_cols, categorical_cols


def prepare_X_y(
    df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    target = config["features"]["target_column"]
    treatment = config["features"]["treatment_column"]

    y = df[target].copy()
    treatment_col = df[treatment].copy()

    feature_cols = (
        config["features"]["numeric_columns"]
        + config["features"]["categorical_columns"]
    )
    X = df[feature_cols].copy()

    logger.info(
        "Feature matrix shape: %s | Target: '%s' | Treatment col extracted.",
        X.shape,
        target,
    )
    return X, y, treatment_col


def fit_and_save_pipeline(
    X: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list,
    artifact_path: str = "models/artifacts/preprocessor.pkl",
) -> ColumnTransformer:
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    preprocessor.fit(X)
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info("Preprocessor fitted and saved to %s.", path)
    return preprocessor


def transform_features(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer,
    numeric_cols: list,
    categorical_cols: list,
) -> np.ndarray:
    X_transformed = preprocessor.transform(X)
    logger.info("Feature transformation complete. Shape: %s.", X_transformed.shape)
    return X_transformed


def run_engineering_pipeline(config: dict) -> Tuple[np.ndarray, pd.Series, pd.Series]:
    df = load_raw_data(config["data"]["output_path"])
    X, y, treatment = prepare_X_y(df, config)
    numeric_cols, categorical_cols = get_feature_columns(config)
    preprocessor = fit_and_save_pipeline(X, numeric_cols, categorical_cols)
    X_transformed = transform_features(X, preprocessor, numeric_cols, categorical_cols)
    return X_transformed, y, treatment


def main() -> None:
    config = load_config()
    X, y, treatment = run_engineering_pipeline(config)
    logger.info(
        "Engineering pipeline complete. X=%s, y=%s, treatment=%s.",
        X.shape, y.shape, treatment.shape,
    )


if __name__ == "__main__":
    main()
