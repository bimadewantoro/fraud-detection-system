# Model Documentation — Fraud Detection & Early Warning System

> **FraudShield AI v1.0**
> Comprehensive technical documentation on how the Machine Learning models are trained, evaluated, and deployed for real-time fraud detection and credit risk early warning.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Pipeline](#2-data-pipeline)
3. [Feature Engineering](#3-feature-engineering)
4. [Model 1: Fraud Detection](#4-model-1-fraud-detection)
5. [Model 2: Early Warning System (EWS)](#5-model-2-early-warning-system-ews)
6. [Risk Scoring Engine](#6-risk-scoring-engine)
7. [Decision Logic & Business Rules](#7-decision-logic--business-rules)
8. [Model Evaluation Metrics](#8-model-evaluation-metrics)
9. [Model Artifacts & Serialization](#9-model-artifacts--serialization)
10. [MLOps & Retraining Strategy](#10-mlops--retraining-strategy)
11. [API Integration](#11-api-integration)
12. [Glossary](#12-glossary)

---

## 1. System Overview

### Architecture

The system consists of three interconnected ML pipelines:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                     │
│  Internal Transactions │ Credit Histories │ Bureau Data │ Alt. Data     │
└──────────┬──────────────┴────────┬─────────┴──────┬──────┴──────────────┘
           │                       │                │
           ▼                       ▼                ▼
┌──────────────────────┐ ┌──────────────────────────────────────────────┐
│   Feature Engineering │ │        Feature Engineering                   │
│   (Transaction-level) │ │        (Credit Profile-level)               │
│   12 features         │ │        13 features                          │
└──────────┬────────────┘ └────────────────┬───────────────────────────┘
           │                               │
           ▼                               ▼
┌──────────────────────┐ ┌──────────────────────────────────────────────┐
│ FRAUD DETECTION MODEL│ │    EARLY WARNING SYSTEM (EWS) MODEL          │
│ Random Forest (40%)  │ │    XGBoost Classifier                        │
│ + XGBoost (60%)      │ │                                              │
│ Ensemble             │ │                                              │
└──────────┬────────────┘ └────────────────┬───────────────────────────┘
           │                               │
           ▼                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     RISK SCORING ENGINE                               │
│    Fraud Probability → Risk Score (0–1000) → Decision (BLOCK/FLAG/OK) │
│    Default Probability → EWS Score → Severity (Red/Orange/Yellow/Green)│
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FastAPI REST Service                                │
│     POST /api/score-transaction    Real-time scoring endpoint         │
│     GET  /api/dashboard/stats      Dashboard aggregations             │
│     GET  /api/alerts               Alert management                   │
└──────────────────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Real-time Scoring** | Models loaded in-memory; single-transaction scoring in < 50ms |
| **Ensemble Approach** | Weighted average of RF + XGBoost reduces variance and overfitting |
| **Explainability** | Feature importance exported for audit trail compliance |
| **Class Imbalance Handling** | `class_weight="balanced"` (RF) and `scale_pos_weight` (XGBoost) |
| **Reproducibility** | Fixed `random_state=42` across all splits and training runs |

---

## 2. Data Pipeline

### 2.1 Data Sources

The system ingests and processes data from the following sources:

| Source | Description | Volume | Update Frequency |
|--------|-------------|--------|-----------------|
| **Internal Transactions** | Banking transactions (amount, merchant, channel, geo, device) | 50,000+ records | Real-time streaming |
| **Credit Profiles** | Loan/credit data (outstanding, DPD, repayment history) | 10,000+ profiles | Daily batch |
| **Bureau Scores** | External credit bureau data (scores, trends) | Per-profile | Monthly |
| **Customer Demographics** | Age, income, segment, products, account tenure | 10,000+ customers | Static / quarterly |
| **Alternative Data** | Geolocation, device fingerprint, login behavior | Per-transaction | Real-time |

### 2.2 Data Generation (Synthetic Demo)

For this demonstration, all data is synthetically generated with realistic distributions:

```python
# Transaction amounts follow log-normal distribution
amounts = rng.lognormal(mean=11.0, sigma=1.2, size=n).clip(5_000, 50_000_000)

# Fraud patterns injected at ~4% rate with 5 distinct attack types:
#   - Velocity attacks (20%)
#   - Geographic anomalies (25%)
#   - High-risk merchant categories (20%)
#   - Unusual amounts (20%)
#   - Off-hours patterns (15%)
```

### 2.3 Class Distribution

| Dataset | Positive Class | Rate | Purpose |
|---------|---------------|------|---------|
| Transactions | `is_fraud = 1` | ~4% (2,000 of 50,000) | Fraud detection |
| Credit Profiles | `is_ews_flag = 1` | ~10% (1,000 of 10,000) | EWS prediction |

---

## 3. Feature Engineering

### 3.1 Fraud Detection Features (12 features)

Features are engineered from raw transaction data and customer profiles:

| # | Feature | Type | Description | Rationale |
|---|---------|------|-------------|-----------|
| 1 | `amount_log` | Continuous | `log(1 + amount)` | Log-transform normalizes the heavily right-skewed transaction amount distribution |
| 2 | `hour_of_day` | Integer (0–23) | Hour when transaction occurred | Fraud often occurs in off-hours (midnight–6 AM) |
| 3 | `day_of_week` | Integer (0–6) | Day of week (Mon=0, Sun=6) | Weekend transactions may have different risk profiles |
| 4 | `is_weekend` | Binary (0/1) | Whether transaction is on Saturday/Sunday | Binary flag simplifies weekend detection for the model |
| 5 | `is_night` | Binary (0/1) | Whether transaction is between 00:00–06:00 | Night transactions have significantly higher fraud rates |
| 6 | `merchant_risk_score` | Continuous (0.0–1.0) | Pre-assigned risk score per merchant category | Categories like "Online Gambling" (0.95) carry inherently higher risk than "Grocery" (0.10) |
| 7 | `channel_encoded` | Integer (0–6) | Numeric encoding of transaction channel | Digital channels (Livin', Kopra) behave differently from physical (Branch, ATM) |
| 8 | `device_encoded` | Integer (0–4) | Numeric encoding of device type | Device type correlates with attack vectors |
| 9 | `geo_distance_log` | Continuous | `log(1 + haversine(home, transaction))` in km | Large geographic deviations from home city indicate potential account takeover |
| 10 | `amount_zscore` | Continuous | Z-score of amount vs. customer's historical average | Transactions deviating significantly from a customer's normal pattern are suspicious |
| 11 | `velocity_1h` | Continuous | Rolling count of transactions within 1-hour window | Velocity attacks generate many transactions in a short burst |
| 12 | `velocity_amount_1h_log` | Continuous | `log(1 + rolling_sum_amount)` over 5-transaction window | Captures both frequency and magnitude of rapid spending |

#### Merchant Risk Score Mapping

```
Low Risk (0.10–0.20):   Grocery, Healthcare, Education, Utilities, Insurance, Gas Station
Medium Risk (0.25–0.50): Entertainment, Electronics, Online Shopping, Travel, ATM Withdrawal, Transfer
High Risk (0.80–0.95):  Cash Advance, Foreign Wire Transfer, Crypto Exchange, Online Gambling
```

#### Geographic Distance Calculation

Uses the **Haversine formula** to compute great-circle distance between the customer's registered home city and the transaction coordinates:

```python
def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two points on Earth."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))
```

### 3.2 EWS Credit Risk Features (13 features)

Features are engineered from credit/loan profile data:

| # | Feature | Type | Description | Rationale |
|---|---------|------|-------------|-----------|
| 1 | `bureau_score_normalized` | Continuous (0–1) | `(score - 300) / 550` | Normalized bureau score; lower = higher risk |
| 2 | `bureau_trend_magnitude` | Continuous | `abs(bureau_score_trend_6m)` | Magnitude of score change over 6 months |
| 3 | `bureau_trend_negative` | Binary (0/1) | Whether 6-month trend is negative | Declining scores are a leading indicator of default |
| 4 | `days_past_due` | Integer | Number of days payment is overdue | Direct measure of delinquency severity |
| 5 | `utilization_ratio` | Continuous (0–1) | Credit line utilization | High utilization signals financial stress |
| 6 | `payment_ratio` | Continuous (0–1) | Ratio of payments made vs. payments due | Low payment ratio indicates inability/unwillingness to pay |
| 7 | `debt_to_income` | Continuous (0–1) | Total debt divided by income | Fundamental credit risk indicator |
| 8 | `outstanding_log` | Continuous | `log(1 + outstanding_amount)` | Log-transformed outstanding balance |
| 9 | `loan_type_encoded` | Integer (0–5) | Numeric encoding (KPR=0, KKB=1, KTA=2, Kartu Kredit=3, KMK=4, KI=5) | Different loan types have different risk profiles |
| 10 | `high_utilization` | Binary (0/1) | Flag: utilization > 75% | Binary threshold for high-utilization detection |
| 11 | `low_payment_ratio` | Binary (0/1) | Flag: payment ratio < 50% | Binary threshold for poor payment behavior |
| 12 | `high_dti` | Binary (0/1) | Flag: debt-to-income > 60% | Binary threshold for over-leveraged borrowers |
| 13 | `risk_composite` | Continuous (0–1) | Weighted composite score | Combines multiple risk signals with expert-assigned weights |

#### Risk Composite Formula

```python
risk_composite = (
    0.25 × (1 - bureau_score_normalized)     # Bureau risk
  + 0.25 × utilization_ratio                 # Utilization stress
  + 0.20 × (1 - payment_ratio)              # Payment delinquency
  + 0.15 × debt_to_income                   # Leverage risk
  + 0.15 × min(days_past_due / 180, 1.0)    # DPD severity (capped)
)
```

---

## 4. Model 1: Fraud Detection

### 4.1 Architecture: Weighted Ensemble

The fraud detection model uses a **weighted ensemble** of two complementary algorithms:

```
                    ┌─────────────────────────┐
                    │    Input Features (12)   │
                    └─────────┬───────────────┘
                              │
                    ┌─────────┴───────────┐
                    │                     │
              ┌─────▼──────┐      ┌──────▼──────┐
              │  Random     │      │   XGBoost   │
              │  Forest     │      │   Gradient  │
              │  200 trees  │      │   Boosting  │
              │  depth=15   │      │   300 trees │
              │             │      │   depth=8   │
              └─────┬──────┘      └──────┬──────┘
                    │                     │
                    │  P_rf (40%)         │  P_xgb (60%)
                    │                     │
              ┌─────▼─────────────────────▼─────┐
              │  P_ensemble = 0.4×P_rf + 0.6×P_xgb │
              └─────────────────┬───────────────┘
                                │
                                ▼
                    Risk Score = P × 1000
                      (0 – 1000 scale)
```

### 4.2 Why This Ensemble?

| Model | Strengths | Weaknesses | Role in Ensemble |
|-------|-----------|------------|------------------|
| **Random Forest** | Low variance, handles noise well, resistant to overfitting | Lower sensitivity to complex interactions | Provides stable baseline predictions (40% weight) |
| **XGBoost** | Captures complex feature interactions, handles class imbalance natively | Can overfit on noisy features | Provides boosted, high-sensitivity predictions (60% weight) |

The weighted average (40/60) was chosen because XGBoost generally achieves higher AUC-ROC on fraud datasets due to its sequential boosting, while Random Forest provides stability and reduces false positives.

### 4.3 Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=200,        # 200 decision trees in the forest
    max_depth=15,            # Max tree depth (prevents overfitting)
    min_samples_split=10,    # Min samples to split an internal node
    min_samples_leaf=5,      # Min samples in each leaf node
    class_weight="balanced", # Auto-adjusts weights inversely proportional
                             # to class frequencies (handles 4% fraud rate)
    random_state=42,         # Reproducibility
    n_jobs=-1,               # Parallel training on all CPU cores
)
```

**How `class_weight="balanced"` works:**
- Normal class weight: `50,000 / (2 × 48,000) ≈ 0.52`
- Fraud class weight: `50,000 / (2 × 2,000) ≈ 12.5`
- The model penalizes misclassifying fraud ~24× more than misclassifying legitimate transactions.

### 4.4 XGBoost Configuration

```python
XGBClassifier(
    n_estimators=300,                 # 300 boosting rounds
    max_depth=8,                      # Max tree depth per round
    learning_rate=0.05,               # Step size shrinkage (conservative)
    scale_pos_weight=24.0,            # Ratio: n_negative / n_positive
                                      # Equivalent to class_weight="balanced"
    eval_metric="logloss",            # Binary cross-entropy loss
    random_state=42,
    n_jobs=-1,
)
```

**Key differences from Random Forest:**
- **Sequential learning**: Each tree corrects the errors of the previous ones (boosting)
- **Lower learning rate (0.05)**: Forces the model to learn gradually, reducing overfitting
- **`scale_pos_weight`**: Directly computed as `count(non-fraud) / count(fraud)` for precise class rebalancing

### 4.5 Training Process

```
Step 1: Feature Engineering
   └── Raw transactions + customer data → 12 engineered features

Step 2: Stratified Train/Test Split (80/20)
   └── Stratification ensures both sets maintain the ~4% fraud ratio

Step 3: Train Random Forest on training set
   └── Outputs: P_rf (fraud probability per transaction)

Step 4: Train XGBoost on training set
   └── Outputs: P_xgb (fraud probability per transaction)

Step 5: Ensemble Combination on test set
   └── P_ensemble = 0.4 × P_rf + 0.6 × P_xgb
   └── Classification threshold: P ≥ 0.5 → Fraud

Step 6: Evaluate & Save
   └── Metrics: AUC-ROC, Accuracy, Precision, Recall, F1
   └── Artifacts: fraud_rf_model.joblib, fraud_xgb_model.joblib
```

---

## 5. Model 2: Early Warning System (EWS)

### 5.1 Architecture: XGBoost Classifier

The EWS model uses a single **XGBoost Gradient Boosting** classifier to predict the probability of credit default:

```
                    ┌──────────────────────────┐
                    │    Input Features (13)    │
                    └─────────┬────────────────┘
                              │
                    ┌─────────▼────────────────┐
                    │      XGBoost              │
                    │      250 boosting rounds  │
                    │      max_depth=6          │
                    │      lr=0.05              │
                    └─────────┬────────────────┘
                              │
                              ▼
                    P(default) → EWS Score (0–1000)
                              │
                              ▼
                    ┌──────────────────────────┐
                    │   Severity Classification │
                    │   ≥ 800 → 🔴 Red          │
                    │   ≥ 600 → 🟠 Orange       │
                    │   ≥ 400 → 🟡 Yellow       │
                    │   < 400 → 🟢 Green        │
                    └──────────────────────────┘
```

### 5.2 XGBoost Configuration (EWS)

```python
XGBClassifier(
    n_estimators=250,            # 250 boosting rounds
    max_depth=6,                 # Shallower than fraud model (simpler patterns)
    learning_rate=0.05,          # Conservative learning rate
    scale_pos_weight=9.0,        # ~10% positive class → ratio ≈ 9:1
    eval_metric="logloss",       # Binary cross-entropy
    random_state=42,
)
```

### 5.3 Why Single Model (No Ensemble)?

Credit risk patterns are more structured and less noisy than fraud patterns. The EWS signals rely on clear financial indicators (DPD, utilization, bureau trends) where XGBoost alone achieves sufficient accuracy without the added complexity of an ensemble.

### 5.4 EWS Severity Classification

The model's default probability is mapped to a 4-tier severity system aligned with banking risk management standards:

| Severity | Score Range | EWS Probability | Recommended Action |
|----------|-------------|------------------|--------------------|
| 🟢 **Green** | 0 – 399 | < 0.4 | Normal monitoring. No immediate action. |
| 🟡 **Yellow** | 400 – 599 | 0.4 – 0.6 | Enhanced monitoring. Quarterly review by credit analyst. |
| 🟠 **Orange** | 600 – 799 | 0.6 – 0.8 | Priority review by Relationship Manager. Client meeting within 7 days. |
| 🔴 **Red** | 800 – 1000 | ≥ 0.8 | Immediate escalation to Risk Committee. Consider restructuring/provisioning. |

---

## 6. Risk Scoring Engine

### 6.1 How Scoring Works

The `RiskEngine` class loads trained models into memory and provides real-time scoring:

```python
class RiskEngine:
    def score_transaction(self, features: dict) -> dict:
        # 1. Extract feature vector
        feature_vector = pd.DataFrame([features])[self.fraud_features]

        # 2. Get predictions from both models
        rf_proba  = self.fraud_rf.predict_proba(feature_vector)[:, 1][0]
        xgb_proba = self.fraud_xgb.predict_proba(feature_vector)[:, 1][0]

        # 3. Weighted ensemble
        fraud_probability = 0.4 * rf_proba + 0.6 * xgb_proba

        # 4. Map to 0–1000 risk score
        risk_score = int(fraud_probability * 1000)  # clipped to [0, 1000]

        # 5. Apply decision rules
        return risk_score, risk_level, decision
```

### 6.2 Scoring Modes

| Mode | Method | Use Case | Latency |
|------|--------|----------|---------|
| **Single Transaction** | `score_transaction(features)` | Real-time API calls from Livin'/Kopra | < 50ms |
| **Single Credit Profile** | `score_credit_profile(features)` | On-demand EWS assessment | < 30ms |
| **Batch Scoring** | `batch_score_transactions(df)` | End-of-day batch re-scoring | ~5s for 50K rows |

---

## 7. Decision Logic & Business Rules

### 7.1 Fraud Detection Decisions

```
                        Risk Score
                   0 ──────────────── 1000
                   │                    │
    ┌──────────────┼────────┬───────────┤
    │   APPROVE    │  FLAG  │   BLOCK   │
    │   (< 500)    │(500-799)│  (≥ 800) │
    └──────────────┴────────┴───────────┘
```

| Decision | Risk Score | Action Taken | Response Time |
|----------|-----------|--------------|---------------|
| **APPROVE** | 0 – 499 | Transaction proceeds normally | Instant |
| **FLAG** | 500 – 799 | Notification sent to Fraud Investigator / Relationship Manager for manual review | Within 5 minutes |
| **BLOCK** | 800 – 1000 | Transaction automatically blocked. Alert sent to Fraud Investigation Unit | Instant |

### 7.2 Business Workflow Integration

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────────────┐
│  Transaction │────▶│  Risk Engine  │────▶│  Decision Router           │
│  from Livin' │     │  Score: 850   │     │                           │
│  / Kopra     │     │  Level: HIGH  │     │  ┌─ HIGH → Auto-BLOCK    │
└─────────────┘     └──────────────┘     │  ├─ MEDIUM → Investigator │
                                          │  └─ LOW → Pass-through    │
                                          └───────────────────────────┘
```

---

## 8. Model Evaluation Metrics

### 8.1 Key Metrics Explained

| Metric | Formula | What It Measures | Why It Matters for Fraud/EWS |
|--------|---------|------------------|------------------------------|
| **AUC-ROC** | Area under ROC curve | Model's ability to distinguish classes at all thresholds | Primary metric: threshold-independent performance |
| **Accuracy** | (TP+TN) / Total | Overall correctness | Less meaningful with imbalanced data, but useful as baseline |
| **Precision** | TP / (TP+FP) | Among flagged items, how many are truly fraud? | High precision → fewer false alarms for investigators |
| **Recall** | TP / (TP+FN) | Among actual fraud, how many did we catch? | High recall → fewer missed fraud cases |
| **F1 Score** | 2 × (P×R)/(P+R) | Harmonic mean of precision and recall | Balanced view of precision-recall tradeoff |

### 8.2 Confusion Matrix Interpretation

```
                        Predicted
                    Negative    Positive
              ┌────────────┬────────────┐
  Actual  Neg │    TN      │    FP      │  FP = False alarms
              │            │            │  (legitimate flagged as fraud)
              ├────────────┼────────────┤
  Actual  Pos │    FN      │    TP      │  FN = Missed fraud
              │            │            │  (fraud not detected)
              └────────────┴────────────┘
```

- **Minimizing FN (False Negatives)** is critical — every missed fraud = financial loss
- **Minimizing FP (False Positives)** reduces operational burden on investigators

### 8.3 Metrics Output

Both models generate a `*_metrics.json` file containing:

```json
{
  "model": "Fraud Detection (RF + XGBoost Ensemble)",
  "auc_roc": 0.98,
  "accuracy": 0.96,
  "precision": 0.85,
  "recall": 0.92,
  "f1_score": 0.88,
  "confusion_matrix": [[9200, 200], [80, 920]],
  "feature_importance": {
    "amount_log": 0.18,
    "geo_distance_log": 0.15,
    "merchant_risk_score": 0.14,
    ...
  }
}
```

---

## 9. Model Artifacts & Serialization

### 9.1 Saved Files

All trained models and metadata are persisted in `models/trained/`:

| File | Format | Size (approx.) | Description |
|------|--------|-----------------|-------------|
| `fraud_rf_model.joblib` | Joblib | ~30 MB | Serialized Random Forest (200 trees) |
| `fraud_xgb_model.joblib` | Joblib | ~5 MB | Serialized XGBoost (300 rounds) |
| `fraud_feature_cols.joblib` | Joblib | < 1 KB | List of 12 feature column names |
| `fraud_metrics.json` | JSON | < 5 KB | Evaluation metrics and feature importance |
| `ews_model.joblib` | Joblib | ~3 MB | Serialized EWS XGBoost (250 rounds) |
| `ews_feature_cols.joblib` | Joblib | < 1 KB | List of 13 feature column names |
| `ews_metrics.json` | JSON | < 5 KB | EWS evaluation metrics and feature importance |

### 9.2 Loading Models

```python
import joblib

# Load fraud ensemble
rf_model  = joblib.load("models/trained/fraud_rf_model.joblib")
xgb_model = joblib.load("models/trained/fraud_xgb_model.joblib")
features  = joblib.load("models/trained/fraud_feature_cols.joblib")

# Or use the RiskEngine wrapper
from engine.risk_engine import RiskEngine
engine = RiskEngine(models_dir="models/trained")
engine.load_models()
result = engine.score_transaction(feature_dict)
```

---

## 10. MLOps & Retraining Strategy

### 10.1 When to Retrain

| Trigger | Detection Method | Action |
|---------|------------------|--------|
| **Model Drift** | AUC-ROC drops below 0.90 on recent data | Full retrain with last 6 months data |
| **Data Distribution Shift** | Feature distributions diverge (KS test > 0.1) | Feature engineering review + retrain |
| **New Fraud Patterns** | Manual investigation identifies uncaptured pattern | Add new features + retrain |
| **Scheduled** | Calendar-based (quarterly) | Retrain as preventive maintenance |

### 10.2 Retraining Pipeline

```
1. Extract fresh data (last 6 months)
2. Run feature engineering pipeline
3. Train new models with same hyperparameters
4. Evaluate on holdout set
5. Compare metrics against production model
6. If new model is better → promote to production
7. Archive old model with timestamp
8. Update audit trail
```

### 10.3 Model Monitoring Checklist

- [ ] Weekly: Check prediction distribution for score concentration shifts
- [ ] Monthly: Validate AUC-ROC on labeled recent transactions
- [ ] Monthly: Review false positive rate with Fraud Investigation team
- [ ] Quarterly: Full model retrain and backtesting
- [ ] Ad-hoc: Retrain when new fraud patterns are investigated

---

## 11. API Integration

### 11.1 Real-Time Scoring Endpoint

**Request:**
```http
POST /api/score-transaction
Content-Type: application/json

{
  "amount": 75000000,
  "merchant_category": "Foreign Wire Transfer",
  "channel": "Livin'",
  "device_type": "Android",
  "latitude": 35.6762,
  "longitude": 139.6503,
  "hour_of_day": 3,
  "customer_id": "CUST-001234"
}
```

**Response:**
```json
{
  "risk_score": 892,
  "fraud_probability": 0.892,
  "risk_level": "HIGH",
  "decision": "BLOCK",
  "action": "Transaction automatically blocked. Alert sent to Fraud Investigation Unit.",
  "scored_at": "2025-06-15T03:22:45.123456",
  "model_version": "v1.0-ensemble"
}
```

### 11.2 Integration Points

| System | Integration | Protocol |
|--------|------------|----------|
| **Livin' (Mobile App)** | Pre-authorization check before transaction | REST API |
| **Kopra (Corporate)** | Transaction screening for corporate transfers | REST API |
| **Core Banking** | Post-transaction batch scoring | Batch API |
| **Fraud Investigation** | Alert dashboard and case management | Web Dashboard |
| **Credit Risk Unit** | EWS dashboard and portfolio monitoring | Web Dashboard |

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **AUC-ROC** | Area Under the Receiver Operating Characteristic curve — measures how well the model separates classes |
| **Boosting** | Ensemble method where models are trained sequentially, each correcting the errors of the previous |
| **DPD** | Days Past Due — number of days a payment is overdue |
| **DTI** | Debt-to-Income ratio — total debt obligations divided by gross income |
| **EWS** | Early Warning System — predictive system for detecting potential defaults before they occur |
| **Feature Engineering** | Process of creating model inputs from raw data (e.g., log-transforms, Z-scores, distance calculations) |
| **Haversine** | Formula for calculating great-circle distance between two GPS coordinates |
| **Joblib** | Python serialization library optimized for NumPy arrays (used to save/load models) |
| **Log-Normal** | Probability distribution whose logarithm is normally distributed — commonly used for financial amounts |
| **MLOps** | Machine Learning Operations — practices for deploying and maintaining ML models in production |
| **NPL** | Non-Performing Loan — a loan in which the borrower has not made scheduled payments for a specified period |
| **Random Forest** | Ensemble of many decision trees trained on random subsets of data (bagging) |
| **Scale Pos Weight** | XGBoost parameter that scales the gradient for positive class samples to handle imbalance |
| **Stratified Split** | Train/test split that preserves the class ratio in both sets |
| **XGBoost** | Extreme Gradient Boosting — optimized gradient boosting framework known for high performance |
| **Z-Score** | Number of standard deviations a value is from the mean — measures how unusual a value is |

---

> **Document Version:** 1.0
> **Last Updated:** April 2025
> **Author:** FraudShield AI — Data Science Team
> **Classification:** Internal — Restricted
