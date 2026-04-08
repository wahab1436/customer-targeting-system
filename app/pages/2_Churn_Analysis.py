"""
2_Churn_Analysis.py
--------------------
Page 2: SHAP global feature importance, SHAP waterfall for individual
customers, churn rate heatmap, and high-risk customer table.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.pipeline import load_preprocessor, get_feature_names

st.set_page_config(page_title="Churn Analysis", layout="wide")


@st.cache_data
def load_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_churn_model(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_resource
def load_preprocessor_cached():
    return load_preprocessor()


@st.cache_data
def compute_shap_values(_model, X_sample: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(_model)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1]
    return sv


def main():
    st.title("Churn Analysis")
    st.markdown("SHAP explainability, high-risk segments, and churn heatmaps.")

    config = load_config()

    try:
        df = load_data(config["data"]["output_path"])
    except FileNotFoundError:
        st.error("Data not found. Run `make data` first.")
        return

    model = load_churn_model(config["churn_model"]["artifact_path"])
    preprocessor = load_preprocessor_cached()

    if model is None or preprocessor is None:
        st.warning(
            "Churn model or preprocessor not found. "
            "Run `make train` to train models first. "
            "Showing data-only views."
        )
        _show_data_only_views(df)
        return

    feature_names = get_feature_names(preprocessor)
    numeric_cols = config["features"]["numeric_columns"]
    cat_cols = config["features"]["categorical_columns"]
    X = df[numeric_cols + cat_cols]
    X_transformed = preprocessor.transform(X)

    # SHAP sample
    sample_size = min(500, len(df))
    X_sample = X_transformed[:sample_size]

    try:
        shap_values = compute_shap_values(model, X_sample)
        _show_shap_views(shap_values, X_sample, feature_names, df, sample_size)
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        _show_data_only_views(df)
        return

    _show_heatmap(df)
    _show_high_risk_table(df)


def _show_shap_views(shap_values, X_sample, feature_names, df, sample_size):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Global Feature Importance (SHAP)")
        importance = np.abs(shap_values).mean(axis=0)
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Importance": importance,
        }).sort_values("SHAP Importance", ascending=True).tail(15)

        fig = px.bar(
            imp_df,
            x="SHAP Importance",
            y="Feature",
            orientation="h",
            color="SHAP Importance",
            color_continuous_scale="Blues",
            title="Mean |SHAP| Values (Top 15 Features)",
        )
        fig.update_layout(height=450, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Individual Customer Explanation (SHAP Waterfall)")
        customer_idx = st.slider(
            "Select customer index",
            min_value=0,
            max_value=sample_size - 1,
            value=0,
            key="shap_waterfall_idx",
        )

        sv_row = shap_values[customer_idx]
        sorted_idx = np.argsort(np.abs(sv_row))[::-1][:12]
        feats = [feature_names[i] for i in sorted_idx]
        vals = sv_row[sorted_idx]

        colors = ["#16a34a" if v < 0 else "#dc2626" for v in vals]

        fig_wf = go.Figure(go.Bar(
            x=vals,
            y=feats,
            orientation="h",
            marker_color=colors,
        ))
        fig_wf.update_layout(
            title=f"SHAP Waterfall - Customer #{customer_idx}",
            xaxis_title="SHAP Value (impact on churn probability)",
            height=450,
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        if "churn_prob" in df.columns:
            prob = df["churn_prob"].iloc[customer_idx]
            actual = df["churn"].iloc[customer_idx]
            st.metric("Predicted Churn Probability", f"{prob:.2%}")
            st.metric("Actual Churn", "Yes" if actual == 1 else "No")


def _show_heatmap(df: pd.DataFrame):
    st.subheader("Churn Rate Heatmap — Region x Package Type")
    pivot = (
        df.groupby(["region", "package_type"])["churn"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
        .pivot(index="region", columns="package_type", values="churn")
    )
    fig = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="Reds",
        title="Churn Rate (%) by Region and Package Type",
        labels={"color": "Churn Rate (%)"},
    )
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)


def _show_high_risk_table(df: pd.DataFrame):
    st.subheader("High-Risk Customer Segments")

    threshold = st.slider(
        "Churn probability threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        key="risk_threshold",
    )

    if "churn_prob" not in df.columns:
        st.info("Churn probabilities not available. Train the churn model first.")
        return

    high_risk = df[df["churn_prob"] >= threshold].sort_values(
        "churn_prob", ascending=False
    )
    st.write(f"Customers with churn probability >= {threshold:.0%}: **{len(high_risk):,}**")

    display_cols = [
        "customer_id", "tenure", "region", "package_type",
        "complaints_count", "churn_prob",
    ]
    if "uplift_score" in df.columns:
        display_cols.append("uplift_score")
    if "uplift_segment" in df.columns:
        display_cols.append("uplift_segment")

    st.dataframe(high_risk[display_cols].head(100), use_container_width=True)

    csv = high_risk[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download High-Risk List as CSV",
        data=csv,
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )


def _show_data_only_views(df: pd.DataFrame):
    _show_heatmap(df)
    st.subheader("High-Risk Customers (by complaints + inactivity)")
    proxy = df.sort_values(
        ["complaints_count", "last_activity_days"], ascending=[False, False]
    ).head(50)
    st.dataframe(proxy, use_container_width=True)


if __name__ == "__main__":
    main()
else:
    main()
