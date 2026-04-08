"""
report_generator.py
-------------------
Auto-generates an executive PDF report with:
  - Executive summary
  - ROI table by segment
  - Model card (methodology, assumptions, limitations, bias notes)
Uses fpdf2.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml
from fpdf import FPDF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class TelecomReport(FPDF):
    def __init__(self, company_name: str = "TeleConnect Pakistan"):
        super().__init__()
        self.company_name = company_name
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f"{self.company_name} - Customer Targeting Optimization Report", align="C", fill=True)
        self.ln(5)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(
            0, 10,
            f"Confidential | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Page {self.page_no()}",
            align="C",
        )

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(220, 230, 245)
        self.set_text_color(20, 40, 100)
        self.cell(0, 9, title, fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", size=10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def kv_line(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.cell(80, 7, key + ":", border=0)
        self.set_font("Helvetica", size=10)
        self.cell(0, 7, str(value), ln=True)

    def table(self, headers: List[str], rows: List[List[str]], col_widths: List[int]):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, col_widths):
            self.cell(w, 8, h, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", size=9)
        for i, row in enumerate(rows):
            fill = i % 2 == 0
            self.set_fill_color(240, 244, 255) if fill else self.set_fill_color(255, 255, 255)
            for val, w in zip(row, col_widths):
                self.cell(w, 7, str(val), border=1, fill=fill, align="C")
            self.ln()
        self.ln(3)


def generate_report(
    summary: Dict,
    roi_rows: List[Dict],
    output_path: str,
    company_name: str = "TeleConnect Pakistan",
) -> str:
    pdf = TelecomReport(company_name=company_name)
    pdf.add_page()

    # --- Cover / Executive Summary ---
    pdf.section_title("1. Executive Summary")
    pdf.body_text(
        "This report presents the results of the Customer Targeting Optimization System, "
        "designed to identify Persuadable customers: those who will churn without intervention "
        "but will remain loyal when offered a targeted retention offer. The system uses uplift "
        "modeling to measure the incremental effect of interventions rather than simply predicting "
        "churn probability."
    )

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Key Metrics", ln=True)
    pdf.set_font("Helvetica", size=10)

    for k, v in summary.items():
        pdf.kv_line(k, v)

    pdf.ln(4)

    # --- ROI Table ---
    pdf.section_title("2. ROI by Customer Segment")
    pdf.body_text(
        "The table below shows expected retention gain and ROI per segment. "
        "Targeting is recommended only for Persuadable customers."
    )

    if roi_rows:
        headers = list(roi_rows[0].keys())
        rows = [[str(row[h]) for h in headers] for row in roi_rows]
        n_cols = len(headers)
        total_w = 190
        col_w = [total_w // n_cols] * n_cols
        pdf.table(headers, rows, col_w)

    pdf.add_page()

    # --- Model Card ---
    pdf.section_title("3. Model Card")

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Methodology", ln=True)
    pdf.body_text(
        "Churn Model: XGBoost (primary), LightGBM (benchmark). "
        "Class imbalance handled via SMOTE. Hyperparameters tuned with Optuna (Bayesian optimization). "
        "Evaluated via 5-fold stratified cross-validation with AUC-ROC, F1-score, Precision-Recall AUC.\n\n"
        "Uplift Model: Two-model approach. Model T trained on treated customers; "
        "Model C trained on control customers. Uplift Score = P(retain|T) - P(retain|C). "
        "Evaluated via Qini curve and AUUC (Area Under Uplift Curve)."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Data", ln=True)
    pdf.body_text(
        "10,000 synthetic telecom customer records generated with logistically driven churn probability. "
        "Treatment/control split via random assignment (20% treatment rate). "
        "No real PII used. Customer IDs are anonymous hashes."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Assumptions", ln=True)
    pdf.body_text(
        "- The intervention (offer) does not affect customers in the control group.\n"
        "- Customer LTV is uniform across the base (configurable).\n"
        "- Churn is binary (0/1) within the observation window.\n"
        "- Treatment assignment is random and unconfounded."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Limitations", ln=True)
    pdf.body_text(
        "- Synthetic data may not perfectly replicate real-world telecom customer distributions.\n"
        "- Uplift estimates are sensitive to the quality and size of the treatment/control split.\n"
        "- The two-model approach does not directly model interaction effects between treatment "
        "and covariates.\n"
        "- Model performance on future data should be monitored via PSI (Population Stability Index)."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Bias & Fairness Notes", ln=True)
    pdf.body_text(
        "- Region (Urban/Rural) is used as a feature. Differential offer rates across regions "
        "should be reviewed to avoid unintentional discrimination.\n"
        "- SMOTE introduces synthetic minority samples; results may differ on real imbalanced datasets.\n"
        "- Sleeping Dogs (customers who churn faster when offered) are explicitly excluded from "
        "targeting to avoid adverse outcomes."
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    logger.info("PDF report saved to %s.", path)
    return str(path)


def main() -> None:
    config = load_config()

    sample_summary = {
        "Company": config["report"]["company_name"],
        "Report Date": datetime.now().strftime("%Y-%m-%d"),
        "Customers Analyzed": "10,000",
        "Persuadable Customers": "1,240 (12.4%)",
        "Customers Targeted (Budget: Rs. 50,000)": "500",
        "Expected Retention Gain": "+12.4% vs no intervention",
        "Budget Consumed": "Rs. 50,000 (100%)",
        "Estimated ROI": "3.2x",
    }

    sample_roi_rows = [
        {
            "Segment": "Persuadable",
            "Customers": 1240,
            "Cost (Rs.)": 124000,
            "Expected Retained": 186.0,
            "Revenue Saved (Rs.)": 930000,
            "ROI": "7.5x",
        },
        {
            "Segment": "Sure Thing",
            "Customers": 3500,
            "Cost (Rs.)": 350000,
            "Expected Retained": 52.5,
            "Revenue Saved (Rs.)": 262500,
            "ROI": "0.75x",
        },
        {
            "Segment": "Lost Cause",
            "Customers": 2100,
            "Cost (Rs.)": 210000,
            "Expected Retained": 0,
            "Revenue Saved (Rs.)": 0,
            "ROI": "0x",
        },
        {
            "Segment": "Sleeping Dog",
            "Customers": 3160,
            "Cost (Rs.)": 316000,
            "Expected Retained": -31.6,
            "Revenue Saved (Rs.)": -158000,
            "ROI": "-0.5x",
        },
    ]

    generate_report(sample_summary, sample_roi_rows, config["report"]["output_path"])


if __name__ == "__main__":
    main()
