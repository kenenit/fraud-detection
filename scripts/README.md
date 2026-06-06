# Scripts

Standalone scripts for running the full pipeline from the command line.

## Usage

Run from the project root directory:

```bash
# Step 1: Preprocess both datasets
python scripts/preprocess.py

# Step 2: Train and evaluate all models
python scripts/train.py
```

## Scripts

| Script | Description |
|--------|-------------|
| `preprocess.py` | Cleans, enriches, engineers features, applies SMOTE, saves to data/processed/ |
| `train.py` | Trains Logistic Regression and XGBoost on both datasets, saves models to models/ |