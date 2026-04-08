"""
test_optimization.py
--------------------
Unit tests for targeting and ROI calculation modules.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.targeting import greedy_targeting, select_targets
from optimization.roi_calculator import compute_roi, roi_by_segment


@pytest.fixture
def sample_targeting_df():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "customer_id": [f"CUST_{i}" for i in range(n)],
        "uplift_score": np.random.uniform(-0.1, 0.4, size=n),
        "churn_prob": np.random.uniform(0.1, 0.9, size=n),
        "uplift_segment": np.random.choice(
            ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"], size=n
        ),
        "tenure": np.random.randint(1, 60, size=n),
        "region": np.random.choice(["Urban", "Rural", "Semi-urban"], size=n),
        "package_type": np.random.choice(["Basic", "Standard", "Premium"], size=n),
    })


def test_greedy_targeting_n_customers(sample_targeting_df):
    budget = 50000
    cost = 100
    targeted, summary = greedy_targeting(sample_targeting_df, budget, cost)
    expected_n = int(budget // cost)
    expected_n = min(expected_n, len(sample_targeting_df))
    assert summary["n_targeted"] == expected_n


def test_greedy_targeting_top_uplift(sample_targeting_df):
    budget = 20000
    cost = 100
    targeted, _ = greedy_targeting(sample_targeting_df, budget, cost)
    # All targeted customers should have uplift >= min of untargeted
    all_scores = sample_targeting_df["uplift_score"].sort_values(ascending=False)
    n = len(targeted)
    threshold = all_scores.iloc[n - 1]
    assert (targeted["uplift_score"] >= threshold - 1e-6).all()


def test_greedy_targeting_budget_not_exceeded(sample_targeting_df):
    budget = 30000
    cost = 100
    _, summary = greedy_targeting(sample_targeting_df, budget, cost)
    assert summary["budget_used"] <= budget


def test_greedy_targeting_invalid_budget(sample_targeting_df):
    with pytest.raises(ValueError):
        greedy_targeting(sample_targeting_df, budget=0, cost_per_contact=100)


def test_greedy_targeting_invalid_cost(sample_targeting_df):
    with pytest.raises(ValueError):
        greedy_targeting(sample_targeting_df, budget=50000, cost_per_contact=0)


def test_select_targets_greedy(sample_targeting_df):
    targeted, summary = select_targets(
        sample_targeting_df, budget=50000, cost_per_contact=100, method="greedy"
    )
    assert "n_targeted" in summary
    assert summary["method"] == "Greedy Ranking"


def test_select_targets_lp(sample_targeting_df):
    targeted, summary = select_targets(
        sample_targeting_df, budget=10000, cost_per_contact=100, method="lp"
    )
    assert "n_targeted" in summary
    assert summary["budget_used"] <= 10000


def test_compute_roi_basic():
    result = compute_roi(budget=50000, expected_retained=100, avg_ltv=5000)
    assert result["revenue_saved"] == 500000
    assert result["roi_multiplier"] == 10.0
    assert result["net_benefit"] == 450000


def test_compute_roi_zero_retained():
    result = compute_roi(budget=10000, expected_retained=0, avg_ltv=5000)
    assert result["revenue_saved"] == 0
    assert result["roi_multiplier"] == 0.0


def test_compute_roi_invalid_budget():
    with pytest.raises(ValueError):
        compute_roi(budget=0, expected_retained=10, avg_ltv=5000)


def test_roi_by_segment_returns_list(sample_targeting_df):
    rows = roi_by_segment(sample_targeting_df, budget=50000, cost_per_contact=100, avg_ltv=5000)
    assert isinstance(rows, list)
    assert len(rows) > 0


def test_roi_by_segment_keys(sample_targeting_df):
    rows = roi_by_segment(sample_targeting_df, budget=50000, cost_per_contact=100, avg_ltv=5000)
    required_keys = {"Segment", "Customers", "Cost (Rs.)", "Expected Retained", "Revenue Saved (Rs.)", "ROI"}
    for row in rows:
        assert required_keys.issubset(set(row.keys()))


def test_greedy_targeting_summary_expected_retained(sample_targeting_df):
    targeted, summary = greedy_targeting(sample_targeting_df, budget=20000, cost_per_contact=100)
    assert summary["expected_retained"] >= 0


def test_full_budget_exhausted_when_enough_customers(sample_targeting_df):
    budget = 10000
    cost = 100
    _, summary = greedy_targeting(sample_targeting_df, budget, cost)
    assert summary["n_targeted"] == 100
    assert summary["budget_used"] == 10000
