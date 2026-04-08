"""
generate_data.py
----------------
Generates 10,000 synthetic telecom customer records with logically driven
churn probability. Includes treatment/control split for uplift modeling.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_customer_ids(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [f"CUST_{rng.integers(100000, 999999)}" for _ in range(n)]
    )


def generate_data(config: dict) -> pd.DataFrame:
    n = config["data"]["n_customers"]
    seed = config["data"]["random_seed"]
    treatment_rate = config["data"]["treatment_rate"]

    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic telecom customers (seed=%d).", n, seed)

    # --- Base features ---
    tenure = rng.integers(1, 61, size=n).astype(int)

    region = rng.choice(
        ["Urban", "Semi-urban", "Rural"],
        size=n,
        p=[0.50, 0.30, 0.20],
    )

    package_type = rng.choice(
        ["Basic", "Standard", "Premium"],
        size=n,
        p=[0.40, 0.40, 0.20],
    )

    package_map = {"Basic": 1, "Standard": 2, "Premium": 3}
    package_numeric = np.array([package_map[p] for p in package_type])

    # Monthly usage correlated with package
    base_usage = package_numeric * 1500
    monthly_usage_mb = np.clip(
        rng.normal(base_usage, 500), 100, 10000
    ).astype(float)

    # Recharge amount correlated with package
    base_recharge = package_numeric * 400
    recharge_amount = np.clip(
        rng.lognormal(np.log(base_recharge), 0.4), 50, 5000
    ).astype(float)

    # Complaints: Poisson, higher for rural
    region_complaint_rate = np.where(
        region == "Rural", 2.5,
        np.where(region == "Semi-urban", 1.5, 0.8)
    )
    complaints_count = rng.poisson(region_complaint_rate).astype(int)

    # Last activity days: exponential, capped
    last_activity_days = np.clip(
        rng.exponential(20, size=n), 0, 90
    ).astype(int)

    # --- Engineered features ---
    usage_trend_30d = rng.normal(0, 150, size=n).astype(float)
    usage_trend_30d -= (complaints_count * 20)
    usage_trend_30d += (package_numeric * 30)

    recharge_freq_30d = np.clip(
        rng.poisson(package_numeric * 1.5), 0, 15
    ).astype(int)

    usage_per_rupee = np.where(
        recharge_amount > 0,
        monthly_usage_mb / recharge_amount,
        0.0,
    ).astype(float)

    complaints_x_tenure = (complaints_count * tenure).astype(float)

    # --- Churn probability (logistic function of features) ---
    log_odds = (
        -2.5
        + 0.04 * complaints_count
        + 0.03 * last_activity_days
        - 0.02 * tenure
        - 0.0003 * monthly_usage_mb
        - 0.001 * recharge_amount
        - 0.5 * (region == "Rural").astype(int)
        - 0.3 * (region == "Semi-urban").astype(int)
        + 0.5 * (package_type == "Basic").astype(int)
        - 0.3 * (package_type == "Premium").astype(int)
        - 0.002 * usage_trend_30d
        - 0.1 * recharge_freq_30d
        + 0.001 * complaints_x_tenure
        + rng.normal(0, 0.3, size=n)  # noise
    )
    churn_prob = 1.0 / (1.0 + np.exp(-log_odds))
    churn = (rng.uniform(size=n) < churn_prob).astype(int)

    # --- Treatment assignment (random, 20%) ---
    received_offer = (rng.uniform(size=n) < treatment_rate).astype(int)

    # Offer reduces churn for persuadables
    churn_with_offer = churn.copy()
    persuadable_mask = (received_offer == 1) & (churn_prob > 0.4) & (churn_prob < 0.85)
    flip_prob = rng.uniform(size=n)
    churn_with_offer[persuadable_mask & (flip_prob < 0.45)] = 0
    churn = np.where(received_offer == 1, churn_with_offer, churn)

    customer_ids = generate_customer_ids(n, seed + 1)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "tenure": tenure,
            "monthly_usage_mb": monthly_usage_mb.round(2),
            "recharge_amount": recharge_amount.round(2),
            "complaints_count": complaints_count,
            "last_activity_days": last_activity_days,
            "region": region,
            "package_type": package_type,
            "usage_trend_30d": usage_trend_30d.round(4),
            "recharge_freq_30d": recharge_freq_30d,
            "usage_per_rupee": usage_per_rupee.round(4),
            "complaints_x_tenure": complaints_x_tenure.round(2),
            "received_offer": received_offer,
            "churn": churn,
        }
    )

    logger.info(
        "Dataset created: %d rows | churn rate=%.2f%% | treatment rate=%.2f%%",
        len(df),
        df["churn"].mean() * 100,
        df["received_offer"].mean() * 100,
    )
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Data saved to %s", path)


def main() -> None:
    config = load_config()
    df = generate_data(config)
    save_data(df, config["data"]["output_path"])


if __name__ == "__main__":
    main()
