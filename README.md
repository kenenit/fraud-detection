# Fraud Detection — Adey Innovations Inc.

End-to-end ML pipeline for detecting fraud in e-commerce and bank credit card transactions.

## Project Structure
fraud-detection/
├── data/ # Raw and processed data (gitignored)
├── notebooks/ # EDA, feature engineering, modeling, SHAP
├── src/ # Reusable Python modules
├── scripts/ # Standalone scripts
├── tests/ # Unit tests
├── models/ # Saved model artifacts (gitignored)
└── requirements.txt

## Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Datasets
- `Fraud_Data.csv` — E-commerce transactions
- `IpAddress_to_Country.csv` — IP range to country mapping
- `creditcard.csv` — Bank credit card transactions (PCA-anonymized)

## Tasks
- **Task 1**: EDA, preprocessing, geolocation, feature engineering
- **Task 2**: Model building (Logistic Regression + XGBoost/LightGBM)
- **Task 3**: SHAP explainability and business recommendations
# Fraud Detection — Adey Innovations Inc.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/kenenit/fraud-detection/actions/workflows/unittests.yml/badge.svg)

End-to-end machine learning pipeline for detecting fraud in e-commerce 
and bank credit card transactions, built for Adey Innovations Inc.

---

## Project Overview

Fraud detection is critical in FinTech — false negatives cause direct 
financial loss while false positives erode customer trust. This project 
builds, evaluates, and explains fraud detection models across two very 
different transaction datasets.

---

## Project Structure
fraud-detection/
├── .github/workflows/    # CI/CD with GitHub Actions
├── data/
│   ├── raw/              # Original datasets (gitignored)
│   └── processed/        # Cleaned, engineered, resampled data
├── notebooks/
│   ├── eda-fraud-data.ipynb         # Task 1: EDA + feature engineering
│   ├── eda-creditcard.ipynb         # Task 1: Credit card EDA
│   ├── modeling.ipynb               # Task 2: Model training + evaluation
│   └── shap-explainability.ipynb    # Task 3: SHAP analysis
├── models/               # Saved model artifacts (gitignored)
├── src/                  # Reusable Python modules
├── tests/                # Unit tests
├── requirements.txt      # Full dependencies
└── requirements-test.txt # CI dependencies

---

## Datasets

| Dataset | Description | Records |
|---------|-------------|---------|
| `Fraud_Data.csv` | E-commerce transactions with user/device/behavioral context | 151,112 |
| `IpAddress_to_Country.csv` | IP range to country mapping | — |
| `creditcard.csv` | Bank credit card transactions (PCA-anonymized features) | 284,807 |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/kenenit/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

Place the three CSV files in `data/raw/` before running notebooks.

---

## Task 1 — Data Analysis & Preprocessing

- Cleaned missing values, duplicates, and fixed data types
- Converted IP addresses to integers and merged with country data 
  using range-based lookup (`pd.merge_asof`)
- Engineered features: `time_since_signup`, `hour_of_day`, 
  `day_of_week`, `user_tx_count`
- Applied **SMOTE** on training set only to handle class imbalance

| Dataset | Legitimate | Fraud | Fraud Rate |
|---------|-----------|-------|------------|
| Fraud_Data | 90.64% | 9.36% | Moderate imbalance |
| Credit Card | 99.83% | 0.17% | Severe imbalance |

---

## Task 2 — Model Building & Evaluation

Two models trained on each dataset:

| Dataset | Model | AUC-PR | AUC-ROC |
|---------|-------|--------|---------|
| Fraud_Data | Logistic Regression | 0.318 | 0.737 |
| Fraud_Data | **XGBoost** | **0.617** | **0.769** |
| Credit Card | Logistic Regression | 0.725 | 0.970 |
| Credit Card | **XGBoost** | **0.862** | **0.977** |

**Selected model: XGBoost** — superior AUC-PR on both datasets with 
high precision (0.95) minimizing false positives.

---

## Task 3 — SHAP Explainability

Top 5 fraud drivers identified by SHAP:

| Feature | SHAP Importance | Insight |
|---------|----------------|---------|
| `time_since_signup` | 1.109 | Fraudsters act at specific time windows after signup |
| `country_Unknown` | 0.753 | Unresolvable IP is a strong fraud signal |
| `country_United States` | 0.582 | High-volume region needs monitoring |
| `age` | 0.346 | Certain age groups show higher fraud correlation |
| `source_Direct` | 0.237 | Direct traffic linked to higher fraud rates |

---

## Business Recommendations

1. **Time-since-signup rule** — Flag transactions within 24 hours of 
   signup for additional verification (OTP or manual review)
2. **Unknown IP policy** — Automatically hold transactions from 
   unresolvable IP addresses for review
3. **Country risk scoring** — Apply step-up authentication for 
   high fraud-rate countries
4. **Age-based risk scoring** — Incorporate age as a real-time 
   risk factor in the approval pipeline
5. **Direct traffic scrutiny** — Apply stricter velocity checks 
   for users arriving via direct URL

---

## Running Tests

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

---

## Team

Built as part of **10 Academy: AI Mastery — Week 5 & 6 Challenge**

Tutors: Kerod, Mahbubah, Feven