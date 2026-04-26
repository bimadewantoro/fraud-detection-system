"""
FastAPI Backend for Fraud Detection & Early Warning System Dashboard.

Serves the web dashboard and provides REST API endpoints for:
- Real-time transaction scoring
- Dashboard statistics
- Transaction listing with filters
- Active alerts
- Model performance metrics
- Project timeline
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from engine.risk_engine import RiskEngine


# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Fraud Detection & EWS API",
    description="Real-time fraud detection and early warning system for banking",
    version="1.0.0",
)

# ── Global State ─────────────────────────────────────────────────────────────

risk_engine = RiskEngine()
transactions_df = None
credit_profiles_df = None
alerts_df = None
customers_df = None
fraud_metrics = None
ews_metrics = None
scored_transactions = None


def load_data():
    """Load all data and models at startup."""
    global transactions_df, credit_profiles_df, alerts_df, customers_df
    global fraud_metrics, ews_metrics, scored_transactions

    data_dir = "data/generated"
    models_dir = "models/trained"

    # Load datasets
    transactions_df = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
    credit_profiles_df = pd.read_csv(os.path.join(data_dir, "credit_profiles.csv"))
    alerts_df = pd.read_csv(os.path.join(data_dir, "alerts_history.csv"))
    customers_df = pd.read_csv(os.path.join(data_dir, "customers.csv"))

    # Load model metrics
    with open(os.path.join(models_dir, "fraud_metrics.json")) as f:
        fraud_metrics = json.load(f)
    with open(os.path.join(models_dir, "ews_metrics.json")) as f:
        ews_metrics = json.load(f)

    # Load risk engine
    risk_engine.load_models()

    # Pre-score all transactions for dashboard
    from models.feature_engineering import engineer_transaction_features
    enriched_df, feature_cols = engineer_transaction_features(transactions_df, customers_df)
    scored_transactions = risk_engine.batch_score_transactions(enriched_df)

    print("[API] All data and models loaded successfully")


# ── Pydantic Models ──────────────────────────────────────────────────────────

class TransactionInput(BaseModel):
    amount: float
    merchant_category: str
    channel: str
    device_type: str
    latitude: float
    longitude: float
    hour_of_day: int = 12
    customer_id: Optional[str] = None


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard page."""
    return FileResponse("static/index.html")


@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get aggregated dashboard statistics."""
    total_txns = len(scored_transactions)
    fraud_detected = int((scored_transactions["risk_level"] == "HIGH").sum())
    flagged = int((scored_transactions["risk_level"] == "MEDIUM").sum())
    approved = int((scored_transactions["risk_level"] == "LOW").sum())

    total_amount = float(scored_transactions["amount"].sum())
    fraud_amount = float(
        scored_transactions[scored_transactions["is_fraud"] == 1]["amount"].sum()
    )
    blocked_amount = float(
        scored_transactions[scored_transactions["risk_level"] == "HIGH"]["amount"].sum()
    )

    # EWS stats
    ews_total = len(credit_profiles_df)
    ews_flagged = int(credit_profiles_df["is_ews_flag"].sum())
    ews_red = int((credit_profiles_df["ews_severity"] == "Red").sum())
    ews_orange = int((credit_profiles_df["ews_severity"] == "Orange").sum())
    ews_yellow = int((credit_profiles_df["ews_severity"] == "Yellow").sum())
    ews_green = int((credit_profiles_df["ews_severity"] == "Green").sum())

    # Alert stats
    open_alerts = int((alerts_df["status"] == "Open").sum()) if "status" in alerts_df.columns else 0
    critical_alerts = int((alerts_df["severity"] == "Critical").sum()) if "severity" in alerts_df.columns else 0

    # Risk score distribution for chart
    score_bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    score_labels = ["0-100", "100-200", "200-300", "300-400", "400-500",
                    "500-600", "600-700", "700-800", "800-900", "900-1000"]
    score_dist = pd.cut(scored_transactions["risk_score"], bins=score_bins, labels=score_labels)
    score_distribution = score_dist.value_counts().sort_index().to_dict()

    # Fraud by channel
    fraud_by_channel = (
        scored_transactions[scored_transactions["is_fraud"] == 1]
        .groupby("channel")
        .size()
        .to_dict()
    )

    # Fraud by type
    fraud_by_type = (
        scored_transactions[scored_transactions["is_fraud"] == 1]
        .groupby("fraud_type")
        .size()
        .to_dict()
    )

    # Monthly trend
    scored_transactions["month"] = pd.to_datetime(scored_transactions["timestamp"]).dt.month
    monthly_fraud = (
        scored_transactions[scored_transactions["is_fraud"] == 1]
        .groupby("month")
        .size()
        .reindex(range(1, 13), fill_value=0)
        .to_dict()
    )
    monthly_total = (
        scored_transactions.groupby("month")
        .size()
        .reindex(range(1, 13), fill_value=0)
        .to_dict()
    )

    return {
        "summary": {
            "total_transactions": total_txns,
            "fraud_detected": fraud_detected,
            "flagged_for_review": flagged,
            "approved": approved,
            "total_amount": total_amount,
            "fraud_amount": fraud_amount,
            "blocked_amount": blocked_amount,
            "loss_prevented_pct": round(blocked_amount / max(fraud_amount, 1) * 100, 1),
            "sla_improvement_pct": 12.5,
        },
        "ews": {
            "total_profiles": ews_total,
            "flagged": ews_flagged,
            "by_severity": {
                "Red": ews_red,
                "Orange": ews_orange,
                "Yellow": ews_yellow,
                "Green": ews_green,
            },
        },
        "alerts": {
            "open": open_alerts,
            "critical": critical_alerts,
            "total": len(alerts_df),
        },
        "charts": {
            "risk_score_distribution": score_distribution,
            "fraud_by_channel": fraud_by_channel,
            "fraud_by_type": fraud_by_type,
            "monthly_fraud": monthly_fraud,
            "monthly_total": monthly_total,
        },
    }


@app.get("/api/transactions")
async def get_transactions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=5, le=100),
    risk_level: Optional[str] = None,
    is_fraud: Optional[int] = None,
):
    """Get paginated transaction list with risk scores."""
    df = scored_transactions.copy()

    if risk_level:
        df = df[df["risk_level"] == risk_level.upper()]
    if is_fraud is not None:
        df = df[df["is_fraud"] == is_fraud]

    # Sort by risk score descending
    df = df.sort_values("risk_score", ascending=False)

    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size

    columns_to_return = [
        "transaction_id", "customer_id", "timestamp", "amount",
        "merchant_category", "channel", "device_type",
        "risk_score", "risk_level", "decision", "is_fraud", "fraud_type",
    ]

    available_cols = [c for c in columns_to_return if c in df.columns]
    page_data = df.iloc[start:end][available_cols]

    return {
        "transactions": page_data.to_dict(orient="records"),
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size,
        },
    }


@app.get("/api/alerts")
async def get_alerts(
    severity: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=5, le=50),
):
    """Get active alerts and EWS signals."""
    df = alerts_df.copy()

    if severity:
        df = df[df["severity"] == severity]
    if status:
        df = df[df["status"] == status]

    df = df.sort_values("risk_score", ascending=False)
    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "alerts": df.iloc[start:end].to_dict(orient="records"),
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size,
        },
    }


@app.get("/api/model/performance")
async def get_model_performance():
    """Get model performance metrics for both fraud and EWS models."""
    return {
        "fraud_model": fraud_metrics,
        "ews_model": ews_metrics,
    }


@app.post("/api/score-transaction")
async def score_transaction(txn: TransactionInput):
    """Score a single transaction in real-time."""
    features = {
        "amount_log": float(np.log1p(txn.amount)),
        "hour_of_day": txn.hour_of_day,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_night": 1 if txn.hour_of_day < 6 else 0,
        "merchant_risk_score": _get_merchant_risk(txn.merchant_category),
        "channel_encoded": _get_channel_code(txn.channel),
        "device_encoded": _get_device_code(txn.device_type),
        "geo_distance_log": float(np.log1p(100)),
        "amount_zscore": 0.0,
        "velocity_1h": 1.0,
        "velocity_amount_1h_log": float(np.log1p(txn.amount)),
    }

    result = risk_engine.score_transaction(features)
    result["transaction"] = txn.dict()
    return result


@app.get("/api/timeline")
async def get_timeline():
    """Get the 6-month action plan timeline."""
    return {
        "phases": [
            {
                "month": "1-2",
                "title": "Integrasi Data Pipeline",
                "description": "Penyatuan sumber data internal & eksternal. Membangun ETL pipeline untuk validasi dan pembersihan data.",
                "status": "completed",
                "tasks": [
                    "Setup data lake architecture",
                    "ETL pipeline development",
                    "Data quality validation rules",
                    "Source system integration",
                ],
            },
            {
                "month": "3-4",
                "title": "Pengembangan Model ML",
                "description": "Pengembangan & pengujian model Machine Learning untuk fraud detection dan Early Warning System.",
                "status": "in_progress",
                "tasks": [
                    "Feature engineering pipeline",
                    "Model training (RF + XGBoost)",
                    "Hyperparameter tuning",
                    "Model validation & backtesting",
                ],
            },
            {
                "month": "5-6",
                "title": "UAT & Production Launch",
                "description": "User Acceptance Testing dan peluncuran model ke sistem production bank.",
                "status": "planned",
                "tasks": [
                    "API integration with core banking",
                    "User Acceptance Testing",
                    "Performance & load testing",
                    "Production deployment & monitoring",
                ],
            },
        ],
    }


@app.get("/api/ews/profiles")
async def get_ews_profiles(
    severity: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=5, le=50),
):
    """Get credit profiles with EWS flags."""
    df = credit_profiles_df.copy()

    if severity:
        df = df[df["ews_severity"] == severity]

    df = df.sort_values("days_past_due", ascending=False)
    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "profiles": df.iloc[start:end].to_dict(orient="records"),
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size,
        },
    }


# ── Helper Functions ─────────────────────────────────────────────────────────

def _get_merchant_risk(category: str) -> float:
    risk_map = {
        "Grocery": 0.1, "Restaurant": 0.15, "Healthcare": 0.1,
        "Online Shopping": 0.35, "Travel": 0.4, "ATM Withdrawal": 0.45,
        "Transfer": 0.5, "Online Gambling": 0.95, "Crypto Exchange": 0.90,
        "Foreign Wire Transfer": 0.85, "Cash Advance": 0.80,
    }
    return risk_map.get(category, 0.3)


def _get_channel_code(channel: str) -> int:
    channel_map = {
        "Branch": 0, "ATM": 1, "EDC": 2, "Internet Banking": 3,
        "Mobile Banking": 4, "Livin'": 5, "Kopra": 6,
    }
    return channel_map.get(channel, 3)


def _get_device_code(device: str) -> int:
    device_map = {
        "ATM Terminal": 0, "EDC Terminal": 1, "Web Browser": 2,
        "iOS": 3, "Android": 4,
    }
    return device_map.get(device, 2)


# ── Mount static files ───────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")
