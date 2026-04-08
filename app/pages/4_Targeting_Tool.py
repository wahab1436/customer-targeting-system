"""
4_Targeting_Tool.py
--------------------
Page 4: Budget slider, cost-per-contact input, live customer selection,
expected retention gain, ROI metrics, and downloadable target CSV.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimization.targeting import select_targets
from optimization.roi_calculator import compute_roi, roi_by_segment

st.set_page_config(page_title="Targeting Tool", layout="wide")


@st.cache_data
def load_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    st.title("Targeting Tool")
    st.markdown(
        "Adjust the budget and cost parameters to generate a live ranked target list."
    )

    config = load_config()

    try:
        df = load_data(config["data"]["output_path"])
    except FileNotFoundError:
        st.error("Data not found. Run `make data` first.")
        return

    if "uplift_score" not in df.columns:
        st.warning("Uplift scores missing. Run `python models/uplift_model.py` first.")
        return

    opt_cfg = config["optimization"]

    # --- Controls ---
    st.subheader("Budget Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        budget = st.slider(
            "Total Marketing Budget (Rs.)",
            min_value=int(opt_cfg["min_budget"]),
            max_value=int(opt_cfg["max_budget"]),
            value=int(opt_cfg["default_budget"]),
            step=5000,
            format="Rs. %d",
        )

    with col2:
        cost_per_contact = st.number_input(
            "Cost per Contact (Rs.)",
            min_value=10,
            max_value=5000,
            value=int(opt_cfg["default_cost_per_contact"]),
            step=10,
        )

    with col3:
        avg_ltv = st.number_input(
            "Average Customer LTV (Rs.)",
            min_value=500,
            max_value=50000,
            value=int(opt_cfg["avg_customer_ltv"]),
            step=500,
        )

    method = st.radio(
        "Optimization Method",
        options=["greedy", "lp"],
        format_func=lambda x: "Greedy Ranking (fast)" if x == "greedy" else "Linear Programming (PuLP)",
        horizontal=True,
    )

    st.markdown("---")

    # --- Run Targeting ---
    try:
        targeted_df, summary = select_targets(
            df,
            budget=float(budget),
            cost_per_contact=float(cost_per_contact),
            method=method,
        )
    except Exception as e:
        st.error(f"Targeting failed: {e}")
        return

    # --- ROI Computation ---
    roi_result = compute_roi(
        budget=float(budget),
        expected_retained=summary["expected_retained"],
        avg_ltv=float(avg_ltv),
    )

    # --- Output Metrics ---
    st.subheader("Targeting Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers Targeted", f"{summary['n_targeted']:,}")
    c2.metric("Budget Used", f"Rs. {summary['budget_used']:,.0f}")
    c3.metric("Expected Retained", f"{summary['expected_retained']:.1f}")
    c4.metric("Revenue Saved", f"Rs. {roi_result['revenue_saved']:,.0f}")
    c5.metric("ROI", f"{roi_result['roi_multiplier']:.2f}x")

    if method == "lp" and "lp_status" in summary:
        st.caption(f"LP Solver Status: {summary['lp_status']}")

    st.markdown("---")

    # --- Charts ---
    col_left, col_right = st.columns(2)

    with col_left:
        if "uplift_segment" in targeted_df.columns:
            st.subheader("Targeted Customers by Segment")
            seg_counts = targeted_df["uplift_segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            color_map = {
                "Persuadable": "#16a34a",
                "Sure Thing": "#2563eb",
                "Lost Cause": "#dc2626",
                "Sleeping Dog": "#d97706",
            }
            fig = px.bar(
                seg_counts, x="Segment", y="Count",
                color="Segment", color_discrete_map=color_map,
                text="Count", title="Targeted Customers by Segment",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Uplift Score Distribution — Targeted Customers")
        fig2 = px.histogram(
            targeted_df,
            x="uplift_score",
            nbins=40,
            color_discrete_sequence=["#16a34a"],
            title="Uplift Scores of Targeted Customers",
            labels={"uplift_score": "Uplift Score"},
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # --- ROI by Segment ---
    if "uplift_segment" in df.columns:
        st.subheader("ROI by Segment")
        roi_rows = roi_by_segment(df, float(budget), float(cost_per_contact), float(avg_ltv))
        roi_df = pd.DataFrame(roi_rows)
        st.dataframe(roi_df, use_container_width=True)

    # --- Target Table ---
    st.subheader("Target Customer List")

    display_cols = ["customer_id", "tenure", "region", "package_type", "uplift_score"]
    if "churn_prob" in targeted_df.columns:
        display_cols.append("churn_prob")
    if "uplift_segment" in targeted_df.columns:
        display_cols.append("uplift_segment")

    available_cols = [c for c in display_cols if c in targeted_df.columns]
    st.dataframe(targeted_df[available_cols].head(200), use_container_width=True)

    # --- Download ---
    csv_data = targeted_df[available_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Target List as CSV",
        data=csv_data,
        file_name="target_list.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
else:
    main()
