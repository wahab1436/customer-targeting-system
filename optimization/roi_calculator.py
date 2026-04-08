"""
roi_calculator.py
-----------------
ROI calculation: LTV x retained customers / budget.
"""

import logging
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def compute_roi(
    budget: float,
    expected_retained: float,
    avg_ltv: float,
) -> Dict[str, float]:
    if budget <= 0:
        raise ValueError("budget must be positive.")

    revenue_saved = expected_retained * avg_ltv
    roi = revenue_saved / budget if budget > 0 else 0.0
    net_benefit = revenue_saved - budget

    result = {
        "total_budget": round(budget, 2),
        "expected_retained_customers": round(expected_retained, 2),
        "avg_customer_ltv": round(avg_ltv, 2),
        "revenue_saved": round(revenue_saved, 2),
        "net_benefit": round(net_benefit, 2),
        "roi_multiplier": round(roi, 2),
        "roi_percent": round((roi - 1) * 100, 1),
    }

    logger.info(
        "ROI: Budget=Rs.%.0f | Retained=%.1f | Revenue Saved=Rs.%.0f | ROI=%.2fx",
        budget, expected_retained, revenue_saved, roi,
    )
    return result


def roi_by_segment(
    segment_df,
    budget: float,
    cost_per_contact: float,
    avg_ltv: float,
) -> list:
    rows = []
    for seg, grp in segment_df.groupby("uplift_segment"):
        n = len(grp)
        cost = n * cost_per_contact
        retained = grp["uplift_score"].clip(lower=0).sum()
        revenue = retained * avg_ltv
        roi = revenue / cost if cost > 0 else 0.0
        rows.append({
            "Segment": seg,
            "Customers": n,
            "Cost (Rs.)": round(cost, 0),
            "Expected Retained": round(retained, 2),
            "Revenue Saved (Rs.)": round(revenue, 0),
            "ROI": round(roi, 2),
        })
    return rows
