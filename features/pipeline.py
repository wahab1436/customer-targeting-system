"""
pipeline.py
-----------
Exports the fitted sklearn ColumnTransformer pipeline for reuse
across training and inference.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import yaml
from sklearn.compose import ColumnTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

_PREPROCESSOR_PATH = "models/artifacts/preprocessor.pkl"


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_preprocessor(
    path: str = _PREPROCESSOR_PATH,
) -> Optional[ColumnTransformer]:
    p = Path(path)
    if not p.exists():
        logger.error(
            "Preprocessor not found at %s. Run features/engineering.py first.", p
        )
        return None
    preprocessor = joblib.load(p)
    logger.info("Preprocessor loaded from %s.", p)
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    config = load_config()
    numeric_names = config["features"]["numeric_columns"]
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(config["features"]["categorical_columns"]))
    return numeric_names + cat_names
