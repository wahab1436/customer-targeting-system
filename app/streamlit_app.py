"""
streamlit_app.py
----------------
Main entry point for the Customer Targeting Optimization System dashboard.
5-page multi-page Streamlit app.
"""

import streamlit as st

st.set_page_config(
    page_title="Customer Targeting System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Customer Targeting Optimization System")
st.markdown(
    """
    **Telecom Domain | Uplift Modeling | Python 3.11**

    This system identifies *Persuadable* customers — those who will churn without
    intervention but will stay with a targeted retention offer — enabling maximum
    retention gain from a fixed marketing budget.

    ---

    **How to use this dashboard:**
    - Navigate using the sidebar pages on the left.
    - Run the pipeline first: `make data && make train`
    - Then launch this dashboard: `make run`

    **Pages:**
    1. Overview — dataset summary and key metrics
    2. Churn Analysis — SHAP explainability and high-risk segments
    3. Uplift Explorer — Qini curve and segment breakdown
    4. Targeting Tool — budget slider and live target list
    5. Report — executive summary and PDF download
    """
)

st.info(
    "If you see errors on any page, ensure you have run "
    "`python data/generate_data.py` and `python models/churn_model.py` "
    "and `python models/uplift_model.py` first."
)
