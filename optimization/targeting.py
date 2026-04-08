"""
targeting.py
------------
Budget-constrained customer selection.
Two methods:
  1. Greedy ranking (default): sort by uplift score, take top N.
  2. Linear programming (PuLP): maximize uplift under budget constraint.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def greedy_targeting(
    df: pd.DataFrame,
    budget: float,
    cost_per_contact: float,
    uplift_col: str = "uplift_score",
) -> Tuple[pd.DataFrame, dict]:
    if cost_per_contact <= 0:
        raise ValueError("cost_per_contact must be greater than 0.")
    if budget <= 0:
        raise ValueError("budget must be greater than 0.")

    n_to_contact = int(budget // cost_per_contact)
    n_to_contact = min(n_to_contact, len(df))

    targeted = (
        df.sort_values(uplift_col, ascending=False)
        .head(n_to_contact)
        .copy()
    )

    budget_used = n_to_contact * cost_per_contact
    expected_retained = targeted[uplift_col].clip(lower=0).sum()

    summary = {
        "method": "Greedy Ranking",
        "n_targeted": n_to_contact,
        "budget_used": budget_used,
        "budget_available": budget,
        "expected_retained": round(expected_retained, 2),
    }

    logger.info(
        "Greedy targeting: %d customers selected | Budget used: Rs. %.0f | "
        "Expected retained: %.2f.",
        n_to_contact, budget_used, expected_retained,
    )
    return targeted, summary


def lp_targeting(
    df: pd.DataFrame,
    budget: float,
    cost_per_contact: float,
    uplift_col: str = "uplift_score",
) -> Tuple[pd.DataFrame, dict]:
    try:
        import pulp
    except ImportError:
        logger.warning("PuLP not installed. Falling back to greedy method.")
        return greedy_targeting(df, budget, cost_per_contact, uplift_col)

    df_pos = df[df[uplift_col] > 0].copy().reset_index(drop=True)
    n = len(df_pos)
    scores = df_pos[uplift_col].values

    prob = pulp.LpProblem("customer_targeting", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objective: maximize sum of uplift scores
    prob += pulp.lpSum(scores[i] * x[i] for i in range(n))

    # Budget constraint
    prob += pulp.lpSum(cost_per_contact * x[i] for i in range(n)) <= budget

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected = [i for i in range(n) if pulp.value(x[i]) == 1]
    targeted = df_pos.iloc[selected].copy()

    budget_used = len(selected) * cost_per_contact
    expected_retained = targeted[uplift_col].sum()

    summary = {
        "method": "Linear Programming (PuLP)",
        "n_targeted": len(selected),
        "budget_used": budget_used,
        "budget_available": budget,
        "expected_retained": round(expected_retained, 2),
        "lp_status": pulp.LpStatus[prob.status],
    }

    logger.info(
        "LP targeting: %d customers selected | Budget used: Rs. %.0f | "
        "Expected retained: %.2f | Status: %s.",
        len(selected), budget_used, expected_retained, pulp.LpStatus[prob.status],
    )
    return targeted, summary


def select_targets(
    df: pd.DataFrame,
    budget: float,
    cost_per_contact: float,
    method: str = "greedy",
    uplift_col: str = "uplift_score",
) -> Tuple[pd.DataFrame, dict]:
    if method == "lp":
        return lp_targeting(df, budget, cost_per_contact, uplift_col)
    return greedy_targeting(df, budget, cost_per_contact, uplift_col)
