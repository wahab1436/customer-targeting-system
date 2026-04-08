"""
3_Uplift_Explorer.py
---------------------
Page 3: Qini curve, uplift score distribution, 4-quadrant scatter plot,
and segment size breakdown.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.evaluate import compute_qini_curve

st.set_page_config(page_title="Uplift Explorer", layout="wide")


@st.cache_data
def load_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    st.title("Uplift Explorer")
    st.markdown(
        "Qini curve validation, uplift score distributions, and 4-segment customer classification."
    )

    config = load_config()

    try:
        df = load_data(config["data"]["output_path"])
    except FileNotFoundError:
        st.error("Data not found. Run `make data` first.")
        return

    if "uplift_score" not in df.columns:
        st.warning(
            "Uplift scores not found in dataset. "
            "Run `python models/uplift_model.py` first."
        )
        return

    _show_qini_curve(df)
    _show_uplift_distribution(df)
    _show_quadrant_scatter(df)
    _show_segment_breakdown(df)


def _show_qini_curve(df: pd.DataFrame):
    st.subheader("Qini Curve — Uplift Model Quality")

    if "received_offer" not in df.columns:
        st.info("Treatment column not available for Qini curve computation.")
        return

    try:
        x_axis, qini, auuc = compute_qini_curve(
            df["uplift_score"].values,
            df["churn"].values,
            df["received_offer"].values,
        )

        # Random baseline
        random_baseline = np.linspace(0, qini[-1], len(x_axis))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=qini,
            mode="lines",
            name=f"Uplift Model (AUUC={auuc:.4f})",
            line=dict(color="#2563eb", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=random_baseline,
            mode="lines",
            name="Random Baseline",
            line=dict(color="#9ca3af", width=1, dash="dash"),
        ))
        fig.update_layout(
            title=f"Qini Curve | AUUC = {auuc:.4f}",
            xaxis_title="Fraction of Population Targeted",
            yaxis_title="Qini Coefficient",
            height=400,
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "A higher Qini curve above the random baseline indicates the uplift model "
            "is successfully identifying persuadable customers."
        )
    except Exception as e:
        st.error(f"Qini curve computation failed: {e}")


def _show_uplift_distribution(df: pd.DataFrame):
    st.subheader("Uplift Score Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="uplift_score",
            nbins=60,
            color_discrete_sequence=["#2563eb"],
            title="Distribution of Uplift Scores",
            labels={"uplift_score": "Uplift Score"},
        )
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="Zero uplift",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "uplift_segment" in df.columns:
            color_map = {
                "Persuadable": "#16a34a",
                "Sure Thing": "#2563eb",
                "Lost Cause": "#dc2626",
                "Sleeping Dog": "#d97706",
            }
            fig2 = px.box(
                df,
                x="uplift_segment",
                y="uplift_score",
                color="uplift_segment",
                color_discrete_map=color_map,
                title="Uplift Score by Segment",
                labels={
                    "uplift_segment": "Segment",
                    "uplift_score": "Uplift Score",
                },
            )
            fig2.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig2, use_container_width=True)


def _show_quadrant_scatter(df: pd.DataFrame):
    st.subheader("4-Quadrant Scatter: Uplift Score vs Churn Probability")

    if "churn_prob" not in df.columns:
        st.info("Churn probabilities not available. Train the churn model first.")
        return

    plot_df = df.copy()
    sample = plot_df.sample(min(2000, len(plot_df)), random_state=42)

    color_col = "uplift_segment" if "uplift_segment" in sample.columns else "churn"
    color_map = {
        "Persuadable": "#16a34a",
        "Sure Thing": "#2563eb",
        "Lost Cause": "#dc2626",
        "Sleeping Dog": "#d97706",
    }

    fig = px.scatter(
        sample,
        x="churn_prob",
        y="uplift_score",
        color=color_col,
        color_discrete_map=color_map if color_col == "uplift_segment" else None,
        opacity=0.5,
        title="Uplift Score vs Churn Probability (sampled 2,000 customers)",
        labels={
            "churn_prob": "Churn Probability",
            "uplift_score": "Uplift Score",
        },
        hover_data=["customer_id", "tenure", "region", "package_type"],
    )

    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af", annotation_text="Zero uplift")
    fig.add_vline(x=0.5, line_dash="dash", line_color="#9ca3af", annotation_text="50% churn prob")

    # Quadrant annotations
    fig.add_annotation(x=0.8, y=0.15, text="Persuadable", showarrow=False,
                       font=dict(color="#16a34a", size=11))
    fig.add_annotation(x=0.2, y=0.15, text="Sure Thing", showarrow=False,
                       font=dict(color="#2563eb", size=11))
    fig.add_annotation(x=0.8, y=-0.1, text="Lost Cause", showarrow=False,
                       font=dict(color="#dc2626", size=11))
    fig.add_annotation(x=0.2, y=-0.1, text="Sleeping Dog", showarrow=False,
                       font=dict(color="#d97706", size=11))

    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)


def _show_segment_breakdown(df: pd.DataFrame):
    st.subheader("Segment Size Breakdown")

    if "uplift_segment" not in df.columns:
        st.info("Segment data not available.")
        return

    seg_stats = (
        df.groupby("uplift_segment")
        .agg(
            Count=("customer_id", "count"),
            Avg_Uplift=("uplift_score", "mean"),
            Avg_Churn_Prob=("churn_prob", "mean") if "churn_prob" in df.columns
            else ("uplift_score", "count"),
        )
        .round(4)
        .reset_index()
    )
    seg_stats["Pct of Base"] = (seg_stats["Count"] / len(df) * 100).round(1)

    color_map = {
        "Persuadable": "#16a34a",
        "Sure Thing": "#2563eb",
        "Lost Cause": "#dc2626",
        "Sleeping Dog": "#d97706",
    }

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            seg_stats,
            names="uplift_segment",
            values="Count",
            color="uplift_segment",
            color_discrete_map=color_map,
            title="Segment Distribution",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(seg_stats, use_container_width=True)
        st.markdown(
            """
            **Segment definitions:**
            - **Persuadable**: High uplift — target these customers.
            - **Sure Thing**: Low positive uplift — will retain regardless.
            - **Lost Cause**: Negative uplift, high churn — offer will not help.
            - **Sleeping Dog**: Negative uplift, low churn — offer may accelerate churn.
            """
        )


if __name__ == "__main__":
    main()
else:
    main()
