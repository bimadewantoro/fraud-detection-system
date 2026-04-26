"""
ML Model Training Pipeline for Fraud Detection & Early Warning System.

Trains:
1. Fraud Detection Model (Random Forest + XGBoost ensemble)
2. EWS Credit Risk Model (XGBoost)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)
from xgboost import XGBClassifier

from models.feature_engineering import engineer_transaction_features, engineer_credit_features


def train_fraud_model(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    output_dir: str = "models/trained",
    seed: int = 42,
) -> dict:
    """
    Train fraud detection model ensemble (Random Forest + XGBoost).

    Returns dict with model metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)

    # ── Feature engineering ──────────────────────────────────────────────
    print("\n[1/5] Engineering features...")
    df, feature_cols = engineer_transaction_features(transactions, customers)

    X = df[feature_cols].fillna(0)
    y = df["is_fraud"]

    print(f"  ✓ {len(feature_cols)} features extracted")
    print(f"  ✓ Class distribution: {dict(y.value_counts())}")

    # ── Train/test split ─────────────────────────────────────────────────
    print("\n[2/5] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(f"  ✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Train Random Forest ──────────────────────────────────────────────
    print("\n[3/5] Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    print(f"  ✓ RF AUC-ROC: {roc_auc_score(y_test, rf_proba):.4f}")

    # ── Train XGBoost ────────────────────────────────────────────────────
    print("\n[4/5] Training XGBoost...")
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=seed,
        use_label_encoder=False,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    print(f"  ✓ XGB AUC-ROC: {roc_auc_score(y_test, xgb_proba):.4f}")

    # ── Ensemble (weighted average) ──────────────────────────────────────
    print("\n[5/5] Creating ensemble prediction...")
    ensemble_proba = 0.4 * rf_proba + 0.6 * xgb_proba
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    auc_roc = roc_auc_score(y_test, ensemble_proba)
    accuracy = accuracy_score(y_test, ensemble_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, ensemble_pred, average="binary")
    cm = confusion_matrix(y_test, ensemble_pred).tolist()

    print(f"\n  ── ENSEMBLE RESULTS ──")
    print(f"  AUC-ROC:   {auc_roc:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # ── Feature importance ───────────────────────────────────────────────
    importance = dict(zip(feature_cols, xgb_model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # ── Save models ──────────────────────────────────────────────────────
    joblib.dump(rf_model, os.path.join(output_dir, "fraud_rf_model.joblib"))
    joblib.dump(xgb_model, os.path.join(output_dir, "fraud_xgb_model.joblib"))
    joblib.dump(feature_cols, os.path.join(output_dir, "fraud_feature_cols.joblib"))

    metrics = {
        "model": "Fraud Detection (RF + XGBoost Ensemble)",
        "auc_roc": round(auc_roc, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "feature_importance": importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_columns": feature_cols,
    }

    with open(os.path.join(output_dir, "fraud_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  ✓ Models saved to {output_dir}/")
    return metrics


def train_ews_model(
    credit_profiles: pd.DataFrame,
    output_dir: str = "models/trained",
    seed: int = 42,
) -> dict:
    """
    Train Early Warning System model (XGBoost).

    Returns dict with model metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  EARLY WARNING SYSTEM MODEL TRAINING")
    print("=" * 60)

    # ── Feature engineering ──────────────────────────────────────────────
    print("\n[1/4] Engineering credit features...")
    df, feature_cols = engineer_credit_features(credit_profiles)

    X = df[feature_cols].fillna(0)
    y = df["is_ews_flag"]

    print(f"  ✓ {len(feature_cols)} features extracted")
    print(f"  ✓ Class distribution: {dict(y.value_counts())}")

    # ── Train/test split ─────────────────────────────────────────────────
    print("\n[2/4] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(f"  ✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Train XGBoost ────────────────────────────────────────────────────
    print("\n[3/4] Training XGBoost for EWS...")
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=seed,
        use_label_encoder=False,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n  ── EWS MODEL RESULTS ──")
    print(f"  AUC-ROC:   {auc_roc:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # ── Feature importance ───────────────────────────────────────────────
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # ── Save model ───────────────────────────────────────────────────────
    joblib.dump(model, os.path.join(output_dir, "ews_model.joblib"))
    joblib.dump(feature_cols, os.path.join(output_dir, "ews_feature_cols.joblib"))

    metrics = {
        "model": "Early Warning System (XGBoost)",
        "auc_roc": round(auc_roc, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "feature_importance": importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_columns": feature_cols,
    }

    with open(os.path.join(output_dir, "ews_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  ✓ Model saved to {output_dir}/")
    return metrics


if __name__ == "__main__":
    # Quick test
    from data.data_generator import generate_all_data
    customers, transactions, credit_profiles, _ = generate_all_data()
    train_fraud_model(transactions, customers)
    train_ews_model(credit_profiles)
