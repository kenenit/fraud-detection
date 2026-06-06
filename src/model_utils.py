import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, average_precision_score,
    roc_auc_score, ConfusionMatrixDisplay, PrecisionRecallDisplay
)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib


def split_and_resample(X, y, test_size=0.2, random_state=42):
    """
    Stratified train-test split followed by SMOTE on training set only.
    Returns X_train_res, X_test, y_train_res, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )
    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE:  {pd.Series(y_train_res).value_counts().to_dict()}")
    return X_train_res, X_test, y_train_res, y_test


def evaluate_model(name, model, X_test, y_test, save_dir=None):
    """
    Evaluate a trained classifier and print key metrics.
    Optionally saves confusion matrix and PR curve plots.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_pr = average_precision_score(y_test, y_prob)
    auc_roc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  AUC-PR  : {auc_pr:.4f}")
    print(f"  AUC-ROC : {auc_roc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=['Legit', 'Fraud']))

    if save_dir:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=['Legit', 'Fraud'],
            cmap='Blues', ax=axes[0]
        )
        axes[0].set_title(f'{name} — Confusion Matrix')
        PrecisionRecallDisplay.from_estimator(
            model, X_test, y_test, ax=axes[1], name=name
        )
        axes[1].set_title(f'{name} — Precision-Recall Curve')
        plt.tight_layout()
        fname = f"{save_dir}/{name.replace(' ', '_')}_eval.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Plot saved to {fname}")

    return {'model': name, 'AUC-PR': auc_pr, 'AUC-ROC': auc_roc}


def save_model(model, path):
    """Save a trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """Load a trained model from disk."""
    return joblib.load(path)