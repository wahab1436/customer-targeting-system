"""
5_Report.py
-----------
Page 5: Executive summary, ROI table, model card, and one-click PDF download.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimization.roi_calculator import compute_roi, roi_by_segment
from optimization.targeting import select_targets
from reports.report_generator import generate_report

st.set_page_config(page_title="Report", layout="wide")


@st.cache_data
def load_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    st.title("Executive Report")
    st.markdown(
        "Plain-English business output, ROI analysis, model card, and PDF export."
    )

    config = load_config()

    try:
        df = load_data(config["data"]["output_path"])
    except FileNotFoundError:
        st.error("Data not found. Run `make data` first.")
        return

    opt_cfg = config["optimization"]
    budget = float(opt_cfg["default_budget"])
    cost_per_contact = float(opt_cfg["default_cost_per_contact"])
    avg_ltv = float(opt_cfg["avg_customer_ltv"])

    # --- Executive Summary ---
    st.subheader("Executive Summary")

    n_customers = len(df)
    churn_rate = df["churn"].mean() * 100

    n_persuadable = int((df["uplift_segment"] == "Persuadable").sum()) \
        if "uplift_segment" in df.columns else "N/A"

    targeted_df, targeting_summary = (None, {})
    roi_result = {}

    if "uplift_score" in df.columns:
        try:
            targeted_df, targeting_summary = select_targets(
                df, budget=budget, cost_per_contact=cost_per_contact
            )
            roi_result = compute_roi(
                budget=budget,
                expected_retained=targeting_summary.get("expected_retained", 0),
                avg_ltv=avg_ltv,
            )
        except Exception as e:
            st.warning(f"Could not compute targeting: {e}")

    summary_data = {
        "Company": config["report"]["company_name"],
        "Report Date": datetime.now().strftime("%Y-%m-%d"),
        "Customers Analyzed": f"{n_customers:,}",
        "Overall Churn Rate": f"{churn_rate:.1f}%",
        "Persuadable Customers Identified": (
            f"{n_persuadable:,} ({n_persuadable/n_customers*100:.1f}%)"
            if isinstance(n_persuadable, int) else "N/A"
        ),
        "Budget Allocated": f"Rs. {budget:,.0f}",
        "Cost per Contact": f"Rs. {cost_per_contact:,.0f}",
        "Customers Targeted": f"{targeting_summary.get('n_targeted', 'N/A'):,}"
            if isinstance(targeting_summary.get("n_targeted"), int) else "N/A",
        "Expected Customers Retained": f"{targeting_summary.get('expected_retained', 'N/A')}",
        "Estimated Revenue Saved": f"Rs. {roi_result.get('revenue_saved', 0):,.0f}",
        "Estimated ROI": f"{roi_result.get('roi_multiplier', 0):.2f}x",
    }

    col1, col2 = st.columns(2)
    items = list(summary_data.items())
    half = len(items) // 2

    with col1:
        for k, v in items[:half]:
            st.markdown(f"**{k}:** {v}")

    with col2:
        for k, v in items[half:]:
            st.markdown(f"**{k}:** {v}")

    st.markdown("---")

    # --- ROI Table ---
    st.subheader("ROI by Customer Segment")

    if "uplift_segment" in df.columns:
        roi_rows = roi_by_segment(df, budget, cost_per_contact, avg_ltv)
        roi_df = pd.DataFrame(roi_rows)
        st.dataframe(roi_df, use_container_width=True)

        st.caption(
            "Only Persuadable customers are recommended for targeting. "
            "Sure Things waste budget. Lost Causes and Sleeping Dogs should never be contacted."
        )
    else:
        st.info("Uplift model has not been run. No segment-level ROI available.")

    st.markdown("---")

    # --- Model Card ---
    st.subheader("Model Card")

    with st.expander("Methodology", expanded=True):
        st.markdown(
            """
            **Churn Model**
            - Algorithm: XGBoost (primary), LightGBM (benchmark)
            - Class imbalance: SMOTE oversampling
            - Hyperparameter tuning: Optuna (Bayesian optimization, 30 trials)
            - Validation: 5-fold stratified cross-validation
            - Metrics: AUC-ROC, F1-Score, Precision, Recall, PR-AUC
            - Explainability: SHAP TreeExplainer (global + individual)

            **Uplift Model (Two-Model Approach)**
            - Model T: trained on treatment group only — predicts P(retain | treated)
            - Model C: trained on control group only — predicts P(retain | control)
            - Uplift Score = P(churn | control) - P(churn | treated)
            - Segment classification: Persuadable / Sure Thing / Lost Cause / Sleeping Dog
            - Evaluation: Qini curve + AUUC

            **Optimization**
            - Default: Greedy ranking (sort by uplift score, take top N within budget)
            - Advanced: Linear programming via PuLP (maximize total uplift under budget constraint)
            """
        )

    with st.expander("Assumptions"):
        st.markdown(
            """
            - Treatment assignment is random and unconfounded (Randomized Controlled Trial design).
            - The offer does not affect customers in the control group.
            - Customer LTV is assumed uniform; override in config.yaml if real LTV data is available.
            - Churn is measured as a binary event within a fixed observation window.
            - Synthetic data is generated with logistically driven churn probability to approximate
              real telecom customer behavior.
            """
        )

    with st.expander("Limitations"):
        st.markdown(
            """
            - Synthetic data may not perfectly replicate the distribution of a real customer base.
            - Uplift estimates are sensitive to treatment/control split size and balance.
            - The two-model approach does not explicitly model treatment-covariate interactions
              (meta-learner approaches like X-Learner may improve this).
            - Model performance on new data should be monitored using PSI (Population Stability Index)
              and periodic retraining should be scheduled.
            - Qini and AUUC metrics assume binary treatment with no dosage variation.
            """
        )

    with st.expander("Bias and Fairness Notes"):
        st.markdown(
            """
            - Geographic region (Urban / Semi-urban / Rural) is used as a model feature.
              This may result in differential offer rates across regions; this should be
              reviewed with the business before deployment.
            - SMOTE introduces synthetic minority-class samples which may not reflect
              the real population. Review performance on the original imbalanced test set.
            - Sleeping Dog identification is critical: customers who churn faster when
              targeted are explicitly excluded to prevent adverse outcomes.
            - Regular fairness audits are recommended across demographic proxies (region, package type).
            """
        )

    st.markdown("---")

    # --- PDF Download ---
    st.subheader("Download PDF Report")

    if st.button("Generate and Download PDF Report"):
        with st.spinner("Generating PDF report..."):
            try:
                output_path = config["report"]["output_path"]
                roi_rows_for_pdf = (
                    roi_by_segment(df, budget, cost_per_contact, avg_ltv)
                    if "uplift_segment" in df.columns
                    else []
                )
                pdf_path = generate_report(
                    summary=summary_data,
                    roi_rows=roi_rows_for_pdf,
                    output_path=output_path,
                    company_name=config["report"]["company_name"],
                )

                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

                st.download_button(
                    label="Click to download the PDF report",
                    data=pdf_bytes,
                    file_name="customer_targeting_report.pdf",
                    mime="application/pdf",
                )
                st.success(f"PDF report generated successfully: {pdf_path}")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")


if __name__ == "__main__":
    main()
else:
    main()
