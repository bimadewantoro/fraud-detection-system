#!/usr/bin/env python3
"""
FraudShield AI — Main Orchestrator

Runs the full pipeline:
1. Generate synthetic data
2. Train ML models (Fraud Detection + EWS)
3. Launch FastAPI dashboard server
"""

import sys
import os
import time
import uvicorn


def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " FraudShield AI".center(58) + "║")
    print("║" + " Real-Time Fraud Detection & Early Warning System".center(58) + "║")
    print("╚" + "═" * 58 + "╝\n")

    start = time.time()

    # ── Step 1: Generate Data ────────────────────────────────────────────
    print("━" * 60)
    print("  PHASE 1: DATA GENERATION")
    print("━" * 60)
    from data.data_generator import generate_all_data
    customers, transactions, credit_profiles, alerts = generate_all_data()

    # ── Step 2: Train Models ─────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  PHASE 2: MODEL TRAINING")
    print("━" * 60)
    from models.model_training import train_fraud_model, train_ews_model

    fraud_metrics = train_fraud_model(transactions, customers)
    ews_metrics = train_ews_model(credit_profiles)

    elapsed = time.time() - start
    print(f"\n{'━' * 60}")
    print(f"  PIPELINE COMPLETE — {elapsed:.1f}s total")
    print(f"{'━' * 60}")

    # ── Step 3: Launch API Server ────────────────────────────────────────
    print(f"\n{'━' * 60}")
    print("  PHASE 3: LAUNCHING DASHBOARD SERVER")
    print(f"{'━' * 60}")

    # Pre-load data into the API module
    from api.app import load_data
    load_data()

    print("\n  ┌────────────────────────────────────────────────┐")
    print("  │                                                │")
    print("  │   🚀 Dashboard: http://localhost:8000          │")
    print("  │   📡 API Docs:  http://localhost:8000/docs     │")
    print("  │                                                │")
    print("  │   Press Ctrl+C to stop the server              │")
    print("  │                                                │")
    print("  └────────────────────────────────────────────────┘\n")

    from api.app import app
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
