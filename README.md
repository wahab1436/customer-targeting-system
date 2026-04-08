# Customer Targeting Optimization System

**Telecom Domain | Uplift Modeling | Streamlit UI | Python 3.11**

A senior-level data science portfolio project that identifies *Persuadable* customers
in a telecom dataset — those who will churn without intervention but will stay with a
targeted retention offer — enabling maximum retention gain from a fixed marketing budget.

---

## Problem Statement

Standard churn models tell you **who will churn**. Uplift modeling tells you **who will
respond to your offer**. This system uses the two-model uplift approach to measure the
incremental effect of a marketing intervention per customer, avoiding wasted budget on:

- **Sure Things** — customers who will stay regardless
- **Sleeping Dogs** — customers who churn faster when targeted
- **Lost Causes** — customers who will churn no matter what

---

## Architecture — 8 Layers

| Layer | Module | Output |
|---|---|---|
| 1 | `data/generate_data.py` | 10,000 synthetic telecom customers |
| 2 | `data/validate_data.py` | Schema + integrity checks |
| 3 | `features/engineering.py` | 13 ML-ready features |
| 4 | `models/churn_model.py` | Churn probability per customer |
| 5 | `models/uplift_model.py` | Uplift score + 4-segment classification |
| 6 | `optimization/targeting.py` | Budget-ranked target list |
| 7 | `app/streamlit_app.py` | 5-page live dashboard |
| 8 | `reports/report_generator.py` | Downloadable PDF report |

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd customer_targeting_system
make setup
```

### 2. Copy environment file

```bash
cp .env.example .env
```

### 3. Run the full pipeline

```bash
make data    # Generate + validate data
make train   # Engineer features + train models
make run     # Launch Streamlit dashboard
```

Or run everything at once:

```bash
make all
```

### 4. Run tests

```bash
make test
```

### 5. Lint code

```bash
make lint
```

---

## Manual Pipeline Steps

If you prefer running each step individually:

```bash
python data/generate_data.py
python data/validate_data.py
python features/engineering.py
python models/churn_model.py
python models/uplift_model.py
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
customer_targeting_system/
├── data/
│   ├── generate_data.py       # Synthetic data generation
│   ├── validate_data.py       # Schema + integrity checks
│   └── raw/                   # .gitignored CSV files
├── features/
│   ├── engineering.py         # Feature transforms + sklearn Pipeline
│   ├── selection.py           # SHAP-based feature importance
│   └── pipeline.py            # Preprocessor load/save utilities
├── models/
│   ├── churn_model.py         # XGBoost + LightGBM training
│   ├── uplift_model.py        # Two-model uplift approach
│   ├── evaluate.py            # Metrics, SHAP, Qini utilities
│   └── artifacts/             # .gitignored .pkl model files
├── optimization/
│   ├── targeting.py           # Greedy + LP customer selection
│   └── roi_calculator.py      # LTV x retention ROI formula
├── app/
│   ├── streamlit_app.py       # Main entry point
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Churn_Analysis.py
│       ├── 3_Uplift_Explorer.py
│       ├── 4_Targeting_Tool.py
│       └── 5_Report.py
├── reports/
│   └── report_generator.py    # PDF via fpdf2
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_optimization.py
├── config/
│   └── config.yaml            # All hyperparams + defaults
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── requirements.txt
└── README.md
```

---

## Machine Learning Stack

### Churn Model
- **Primary:** XGBoost with SMOTE oversampling
- **Benchmark:** LightGBM
- **Tuning:** Optuna (Bayesian, 30 trials)
- **Validation:** 5-fold stratified cross-validation
- **Explainability:** SHAP TreeExplainer (global + individual waterfall)
- **Metrics:** AUC-ROC, F1, Precision, Recall, PR-AUC

### Uplift Model (Two-Model Approach)
- Model T: trained on treated group only
- Model C: trained on control group only
- Uplift Score = P(churn | control) - P(churn | treated)
- 4-segment classification: Persuadable / Sure Thing / Lost Cause / Sleeping Dog
- Evaluated via Qini curve + AUUC

### Optimization
- **Greedy:** Sort by uplift score, take top N within budget
- **Linear Programming:** PuLP solver, maximize total uplift under budget constraint

---

## Dashboard Pages

| Page | Contents |
|---|---|
| 1 Overview | Metric cards, churn histogram, treatment/control donut, heatmap |
| 2 Churn Analysis | SHAP bar chart, SHAP waterfall, region x package heatmap, high-risk table |
| 3 Uplift Explorer | Qini curve, uplift distribution, 4-quadrant scatter, segment breakdown |
| 4 Targeting Tool | Budget slider, live target list, ROI metrics, CSV download |
| 5 Report | Executive summary, ROI table, model card, PDF download |

---

## Configuration

All hyperparameters and defaults are in `config/config.yaml`. No magic numbers in code.

Key settings:
- `data.n_customers`: number of synthetic customers (default: 10,000)
- `optimization.default_budget`: default marketing budget in Rs. (default: 50,000)
- `optimization.avg_customer_ltv`: average LTV per customer (default: Rs. 5,000)
- `churn_model.xgboost.*`: XGBoost hyperparameters
- `uplift_model.persuadable_threshold`: minimum uplift score to classify as Persuadable

---

## Security Practices

- No secrets hardcoded in source. All config via `.env` (python-dotenv).
- `.gitignore` covers: `data/raw/`, `models/artifacts/`, `.env`, `*.pkl`.
- No PII in dataset. `customer_id` is an anonymous hash.
- Input validation on all Streamlit widgets.
- All model calls wrapped in `try/except` — no stack traces in UI.
- Dependencies pinned in `requirements.txt`.

---

## Code Quality

- `black` — code formatter
- `flake8` — linter (max line length 100)
- `isort` — import sorter
- `pre-commit` — runs all three on every commit
- `pytest` — 4 test modules covering data, features, models, optimization
- `logging` — structured logs throughout (no `print` statements)
- Type hints on all functions

---

## Business Value

| Metric | Value |
|---|---|
| Customers analyzed | 10,000 |
| Persuadable customers identified | ~12.4% of base |
| Expected retention gain (Rs. 50k budget) | +12.4% vs no intervention |
| Estimated ROI | 3.2x |

---

## License

For portfolio and educational use only. Synthetic data only — no real customer PII.
