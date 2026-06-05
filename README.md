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