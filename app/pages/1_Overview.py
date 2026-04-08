"""
1_Overview.py
-------------
Page 1: Dataset overview, key metrics, churn distribution,
treatment/control split, and dataset snapshot.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Overview", layout="wide")


@st.cache_data
def load_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    st.title("Overview")
    st.markdown("Dataset summary, key metrics, and distribution analysis.")

    config = load_config()
    data_path = config["data"]["output_path"]

    try:
        df = load_data(data_path)
    except FileNotFoundError:
        st.error(f"Data file not found at `{data_path}`. Run `make data` first.")
        return

    # --- Metric Cards ---
    st.subheader("Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    churn_rate = df["churn"].mean() * 100
    avg_tenure = df["tenure"].mean()
    avg_ltv = config["optimization"]["avg_customer_ltv"]
    treatment_rate = df["received_offer"].mean() * 100
    n_persuadable = (
        (df["uplift_segment"] == "Persuadable").sum()
        if "uplift_segment" in df.columns
        else "N/A"
    )

    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")
    col4.metric("Avg Customer LTV", f"Rs. {avg_ltv:,}")
    col5.metric(
        "Persuadable Customers",
        f"{n_persuadable:,}" if isinstance(n_persuadable, int) else n_persuadable,
    )

    st.markdown("---")

    # --- Charts Row 1 ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Churn Distribution")
        churn_counts = df["churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Churn"] = churn_counts["Churn"].map({0: "Retained", 1: "Churned"})
        fig_churn = px.bar(
            churn_counts,
            x="Churn",
            y="Count",
            color="Churn",
            color_discrete_map={"Retained": "#2563eb", "Churned": "#dc2626"},
            text="Count",
            title="Churned vs Retained Customers",
        )
        fig_churn.update_traces(textposition="outside")
        fig_churn.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_churn, use_container_width=True)

    with col_right:
        st.subheader("Treatment vs Control Split")
        treat_counts = df["received_offer"].value_counts().reset_index()
        treat_counts.columns = ["Group", "Count"]
        treat_counts["Group"] = treat_counts["Group"].map({0: "Control", 1: "Treated"})
        fig_donut = px.pie(
            treat_counts,
            values="Count",
            names="Group",
            hole=0.5,
            color_discrete_sequence=["#2563eb", "#16a34a"],
            title="Treatment / Control Assignment",
        )
        fig_donut.update_layout(height=350)
        st.plotly_chart(fig_donut, use_container_width=True)

    # --- Charts Row 2 ---
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("Monthly Usage Distribution")
        fig_usage = px.histogram(
            df,
            x="monthly_usage_mb",
            nbins=50,
            color_discrete_sequence=["#2563eb"],
            title="Monthly Usage (MB)",
            labels={"monthly_usage_mb": "Monthly Usage (MB)"},
        )
        fig_usage.update_layout(height=320)
        st.plotly_chart(fig_usage, use_container_width=True)

    with col_right2:
        st.subheader("Churn Rate by Region and Package")
        pivot = (
            df.groupby(["region", "package_type"])["churn"]
            .mean()
            .mul(100)
            .round(1)
            .reset_index()
            .pivot(index="region", columns="package_type", values="churn")
        )
        fig_heat = px.imshow(
            pivot,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Churn Rate (%) by Region x Package",
            labels={"color": "Churn Rate (%)"},
        )
        fig_heat.update_layout(height=320)
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- Uplift Segment Breakdown ---
    if "uplift_segment" in df.columns:
        st.subheader("Uplift Segment Breakdown")
        seg_counts = df["uplift_segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        color_map = {
            "Persuadable": "#16a34a",
            "Sure Thing": "#2563eb",
            "Lost Cause": "#dc2626",
            "Sleeping Dog": "#d97706",
        }
        fig_seg = px.bar(
            seg_counts,
            x="Segment",
            y="Count",
            color="Segment",
            color_discrete_map=color_map,
            text="Count",
            title="Customer Segment Distribution",
        )
        fig_seg.update_traces(textposition="outside")
        fig_seg.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_seg, use_container_width=True)

    # --- Dataset Snapshot ---
    st.subheader("Dataset Snapshot (First 20 Rows)")
    st.dataframe(df.head(20), use_container_width=True)

    # --- Summary Statistics ---
    with st.expander("Descriptive Statistics"):
        st.dataframe(df.describe().round(3), use_container_width=True)


if __name__ == "__main__":
    main()
else:
    main()
