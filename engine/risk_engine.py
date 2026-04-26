"""
Real-Time Risk Scoring Engine.

Loads trained ML models and scores incoming transactions/credit profiles.
Implements decision logic:
  - High Risk (≥ 800):  Auto-block
  - Medium Risk (500-799): Flag for investigation
  - Low Risk (< 500):  Auto-approve
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime


class RiskEngine:
    """Real-time risk scoring engine using trained ML models."""

    RISK_THRESHOLDS = {
        "high": 800,
        "medium": 500,
    }

    def __init__(self, models_dir: str = "models/trained"):
        self.models_dir = models_dir
        self.fraud_rf = None
        self.fraud_xgb = None
        self.fraud_features = None
        self.ews_model = None
        self.ews_features = None
        self._loaded = False

    def load_models(self):
        """Load all trained models from disk."""
        print("[RiskEngine] Loading models...")

        rf_path = os.path.join(self.models_dir, "fraud_rf_model.joblib")
        xgb_path = os.path.join(self.models_dir, "fraud_xgb_model.joblib")
        fraud_feat_path = os.path.join(self.models_dir, "fraud_feature_cols.joblib")
        ews_path = os.path.join(self.models_dir, "ews_model.joblib")
        ews_feat_path = os.path.join(self.models_dir, "ews_feature_cols.joblib")

        if os.path.exists(rf_path):
            self.fraud_rf = joblib.load(rf_path)
            self.fraud_xgb = joblib.load(xgb_path)
            self.fraud_features = joblib.load(fraud_feat_path)
            print("  ✓ Fraud detection models loaded")

        if os.path.exists(ews_path):
            self.ews_model = joblib.load(ews_path)
            self.ews_features = joblib.load(ews_feat_path)
            print("  ✓ EWS model loaded")

        self._loaded = True

    def score_transaction(self, features: dict) -> dict:
        """
        Score a single transaction for fraud risk.

        Args:
            features: Dict with feature values matching fraud_feature_cols

        Returns:
            Dict with risk_score (0-1000), risk_level, decision, and probability
        """
        if not self._loaded:
            self.load_models()

        feature_vector = pd.DataFrame([features])[self.fraud_features].fillna(0)

        # Ensemble prediction
        rf_proba = self.fraud_rf.predict_proba(feature_vector)[:, 1][0]
        xgb_proba = self.fraud_xgb.predict_proba(feature_vector)[:, 1][0]
        fraud_probability = 0.4 * rf_proba + 0.6 * xgb_proba

        # Map probability to 0-1000 score
        risk_score = int(fraud_probability * 1000)
        risk_score = min(max(risk_score, 0), 1000)

        # Decision logic
        if risk_score >= self.RISK_THRESHOLDS["high"]:
            risk_level = "HIGH"
            decision = "BLOCK"
            action = "Transaction automatically blocked. Alert sent to Fraud Investigation Unit."
        elif risk_score >= self.RISK_THRESHOLDS["medium"]:
            risk_level = "MEDIUM"
            decision = "FLAG"
            action = "Transaction flagged. Notification sent to Fraud Investigator for manual review."
        else:
            risk_level = "LOW"
            decision = "APPROVE"
            action = "Transaction approved. No action required."

        return {
            "risk_score": risk_score,
            "fraud_probability": round(float(fraud_probability), 4),
            "risk_level": risk_level,
            "decision": decision,
            "action": action,
            "scored_at": datetime.now().isoformat(),
            "model_version": "v1.0-ensemble",
        }

    def score_credit_profile(self, features: dict) -> dict:
        """
        Score a credit profile for early warning signals.

        Args:
            features: Dict with feature values matching ews_feature_cols

        Returns:
            Dict with ews_score, severity, and recommended action
        """
        if not self._loaded:
            self.load_models()

        feature_vector = pd.DataFrame([features])[self.ews_features].fillna(0)

        default_probability = self.ews_model.predict_proba(feature_vector)[:, 1][0]
        ews_score = int(default_probability * 1000)
        ews_score = min(max(ews_score, 0), 1000)

        if ews_score >= 800:
            severity = "Red"
            action = "Immediate escalation to Risk Committee. Consider loan restructuring or provisioning."
        elif ews_score >= 600:
            severity = "Orange"
            action = "Priority review by Relationship Manager. Schedule client meeting within 7 days."
        elif ews_score >= 400:
            severity = "Yellow"
            action = "Enhanced monitoring. Quarterly review by credit analyst."
        else:
            severity = "Green"
            action = "Normal monitoring. No immediate action required."

        return {
            "ews_score": ews_score,
            "default_probability": round(float(default_probability), 4),
            "severity": severity,
            "action": action,
            "scored_at": datetime.now().isoformat(),
            "model_version": "v1.0-ews",
        }

    def batch_score_transactions(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of transactions and return enriched DataFrame."""
        if not self._loaded:
            self.load_models()

        X = transactions_df[self.fraud_features].fillna(0)

        rf_proba = self.fraud_rf.predict_proba(X)[:, 1]
        xgb_proba = self.fraud_xgb.predict_proba(X)[:, 1]
        ensemble_proba = 0.4 * rf_proba + 0.6 * xgb_proba

        result = transactions_df.copy()
        result["risk_score"] = (ensemble_proba * 1000).clip(0, 1000).astype(int)
        result["fraud_probability"] = np.round(ensemble_proba, 4)
        result["risk_level"] = pd.cut(
            result["risk_score"],
            bins=[-1, self.RISK_THRESHOLDS["medium"], self.RISK_THRESHOLDS["high"], 1001],
            labels=["LOW", "MEDIUM", "HIGH"],
        )
        result["decision"] = result["risk_level"].map({
            "LOW": "APPROVE", "MEDIUM": "FLAG", "HIGH": "BLOCK"
        })

        return result
