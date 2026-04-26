"""
Synthetic Data Generator for Fraud Detection & Early Warning System.

Generates realistic dummy banking data including:
- 50,000+ transaction records with injected fraud patterns (~3-5%)
- 10,000+ credit/loan profiles with EWS signals (~8-12%)
- Alert history records
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Constants ────────────────────────────────────────────────────────────────

MERCHANT_CATEGORIES = [
    "Grocery", "Electronics", "Restaurant", "Gas Station", "Online Shopping",
    "Travel", "ATM Withdrawal", "Transfer", "Subscription", "Healthcare",
    "Education", "Entertainment", "Utilities", "Insurance", "Investment",
]

HIGH_RISK_MERCHANTS = ["Online Gambling", "Crypto Exchange", "Foreign Wire Transfer", "Cash Advance"]

CITIES = [
    ("Jakarta", -6.2088, 106.8456),
    ("Surabaya", -7.2575, 112.7521),
    ("Bandung", -6.9175, 107.6191),
    ("Medan", 3.5952, 98.6722),
    ("Semarang", -6.9666, 110.4196),
    ("Makassar", -5.1477, 119.4327),
    ("Palembang", -2.9761, 104.7754),
    ("Denpasar", -8.6705, 115.2126),
    ("Yogyakarta", -7.7956, 110.3695),
    ("Malang", -7.9786, 112.6304),
]

CHANNELS = ["Livin'", "Kopra", "ATM", "EDC", "Mobile Banking", "Internet Banking", "Branch"]

DEVICE_TYPES = ["Android", "iOS", "Web Browser", "ATM Terminal", "EDC Terminal"]


def generate_customers(n_customers: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate customer base with demographics."""
    rng = np.random.default_rng(seed)

    customer_ids = [f"CUST-{i:06d}" for i in range(1, n_customers + 1)]

    ages = rng.normal(38, 12, n_customers).clip(18, 75).astype(int)
    incomes = (rng.lognormal(10.5, 0.8, n_customers)).clip(3_000_000, 500_000_000).astype(int)

    segments = rng.choice(
        ["Mass", "Mass Affluent", "Affluent", "High Net Worth"],
        n_customers,
        p=[0.55, 0.25, 0.15, 0.05],
    )

    home_cities = rng.choice([c[0] for c in CITIES], n_customers)

    return pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "monthly_income": incomes,
        "segment": segments,
        "home_city": home_cities,
        "account_age_months": rng.integers(1, 240, n_customers),
        "num_products": rng.integers(1, 8, n_customers),
    })


def generate_transactions(
    customers: pd.DataFrame,
    n_transactions: int = 50000,
    fraud_rate: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic transaction data with injected fraud patterns.

    Fraud patterns injected:
    - Velocity attack (many transactions in short window)
    - Geographic anomaly (transaction far from home city)
    - High-risk merchant category
    - Unusual amount (very large for the customer)
    - Off-hours transaction pattern
    """
    rng = np.random.default_rng(seed)

    n_fraud = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud

    # ── Legitimate transactions ──────────────────────────────────────────
    legit_customers = rng.choice(customers["customer_id"].values, n_legit)
    legit_amounts = rng.lognormal(11.0, 1.2, n_legit).clip(5_000, 50_000_000).astype(int)
    legit_merchants = rng.choice(MERCHANT_CATEGORIES, n_legit)
    legit_channels = rng.choice(CHANNELS, n_legit, p=[0.30, 0.10, 0.15, 0.10, 0.20, 0.10, 0.05])
    legit_devices = rng.choice(DEVICE_TYPES, n_legit, p=[0.35, 0.30, 0.20, 0.08, 0.07])

    base_date = datetime(2025, 1, 1)
    legit_timestamps = [
        base_date + timedelta(
            days=int(rng.integers(0, 365)),
            hours=int(rng.choice(range(6, 23), p=_business_hour_probs())),
            minutes=int(rng.integers(0, 60)),
            seconds=int(rng.integers(0, 60)),
        )
        for _ in range(n_legit)
    ]

    legit_cities_idx = rng.integers(0, len(CITIES), n_legit)
    legit_lats = [CITIES[i][1] + rng.normal(0, 0.05) for i in legit_cities_idx]
    legit_lons = [CITIES[i][2] + rng.normal(0, 0.05) for i in legit_cities_idx]

    # ── Fraud transactions ───────────────────────────────────────────────
    fraud_customers = rng.choice(customers["customer_id"].values, n_fraud)
    fraud_types = rng.choice(
        ["velocity_attack", "geo_anomaly", "high_risk_merchant", "unusual_amount", "off_hours"],
        n_fraud,
        p=[0.20, 0.25, 0.20, 0.20, 0.15],
    )

    fraud_amounts = []
    fraud_merchants = []
    fraud_channels_list = []
    fraud_devices_list = []
    fraud_timestamps = []
    fraud_lats = []
    fraud_lons = []

    for i, ftype in enumerate(fraud_types):
        if ftype == "unusual_amount":
            fraud_amounts.append(int(rng.integers(50_000_000, 500_000_000)))
        elif ftype == "high_risk_merchant":
            fraud_amounts.append(int(rng.lognormal(12, 1.5).clip(1_000_000, 100_000_000)))
        else:
            fraud_amounts.append(int(rng.lognormal(11.5, 1.0).clip(100_000, 80_000_000)))

        if ftype == "high_risk_merchant":
            fraud_merchants.append(rng.choice(HIGH_RISK_MERCHANTS))
        else:
            fraud_merchants.append(rng.choice(MERCHANT_CATEGORIES))

        if ftype == "velocity_attack":
            fraud_channels_list.append("Livin'")
            fraud_devices_list.append(rng.choice(["Android", "iOS"]))
        else:
            fraud_channels_list.append(rng.choice(CHANNELS))
            fraud_devices_list.append(rng.choice(DEVICE_TYPES))

        if ftype == "off_hours":
            hour = int(rng.choice([0, 1, 2, 3, 4, 5]))
        elif ftype == "velocity_attack":
            hour = int(rng.integers(0, 24))
        else:
            hour = int(rng.integers(6, 23))

        fraud_timestamps.append(
            base_date + timedelta(
                days=int(rng.integers(0, 365)),
                hours=hour,
                minutes=int(rng.integers(0, 60)),
                seconds=int(rng.integers(0, 60)),
            )
        )

        if ftype == "geo_anomaly":
            fraud_lats.append(float(rng.uniform(-10, 60)))
            fraud_lons.append(float(rng.uniform(50, 150)))
        else:
            city_idx = int(rng.integers(0, len(CITIES)))
            fraud_lats.append(CITIES[city_idx][1] + rng.normal(0, 0.05))
            fraud_lons.append(CITIES[city_idx][2] + rng.normal(0, 0.05))

    # ── Combine ──────────────────────────────────────────────────────────
    all_customer_ids = list(legit_customers) + list(fraud_customers)
    all_amounts = list(legit_amounts) + fraud_amounts
    all_merchants = list(legit_merchants) + fraud_merchants
    all_channels = list(legit_channels) + fraud_channels_list
    all_devices = list(legit_devices) + fraud_devices_list
    all_timestamps = legit_timestamps + fraud_timestamps
    all_lats = legit_lats + fraud_lats
    all_lons = legit_lons + fraud_lons
    all_labels = [0] * n_legit + [1] * n_fraud
    all_fraud_types = ["none"] * n_legit + list(fraud_types)

    txn_ids = [f"TXN-{i:07d}" for i in range(1, n_transactions + 1)]

    df = pd.DataFrame({
        "transaction_id": txn_ids,
        "customer_id": all_customer_ids,
        "timestamp": all_timestamps,
        "amount": all_amounts,
        "merchant_category": all_merchants,
        "channel": all_channels,
        "device_type": all_devices,
        "latitude": all_lats,
        "longitude": all_lons,
        "is_fraud": all_labels,
        "fraud_type": all_fraud_types,
    })

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["transaction_id"] = [f"TXN-{i:07d}" for i in range(1, len(df) + 1)]

    return df


def generate_credit_profiles(
    customers: pd.DataFrame,
    n_profiles: int = 10000,
    ews_rate: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate credit/loan profiles with early warning signals.

    EWS patterns injected:
    - Declining bureau score
    - Increasing days past due
    - High utilization ratio
    - Deteriorating payment patterns
    """
    rng = np.random.default_rng(seed)

    n_ews = int(n_profiles * ews_rate)
    n_normal = n_profiles - n_ews

    selected_customers = rng.choice(customers["customer_id"].values, n_profiles, replace=False)

    loan_ids = [f"LOAN-{i:06d}" for i in range(1, n_profiles + 1)]

    loan_types = rng.choice(
        ["KPR", "KKB", "KTA", "Kartu Kredit", "KMK", "KI"],
        n_profiles,
        p=[0.25, 0.15, 0.20, 0.20, 0.10, 0.10],
    )

    # ── Normal profiles ──────────────────────────────────────────────────
    normal_bureau = rng.normal(720, 50, n_normal).clip(500, 850).astype(int)
    normal_dpd = np.zeros(n_normal, dtype=int)
    normal_utilization = rng.beta(2, 8, n_normal).clip(0.05, 0.60)
    normal_payment_ratio = rng.beta(8, 2, n_normal).clip(0.80, 1.0)
    normal_dti = rng.beta(2, 5, n_normal).clip(0.10, 0.45)
    normal_outstanding = rng.lognormal(17, 1.0, n_normal).clip(5_000_000, 2_000_000_000).astype(int)

    # ── EWS profiles ─────────────────────────────────────────────────────
    ews_bureau = rng.normal(550, 60, n_ews).clip(300, 650).astype(int)
    ews_dpd = rng.choice([30, 60, 90, 120, 150, 180], n_ews, p=[0.30, 0.25, 0.20, 0.12, 0.08, 0.05])
    ews_utilization = rng.beta(7, 2, n_ews).clip(0.70, 1.0)
    ews_payment_ratio = rng.beta(2, 5, n_ews).clip(0.20, 0.70)
    ews_dti = rng.beta(5, 2, n_ews).clip(0.50, 0.95)
    ews_outstanding = rng.lognormal(18, 0.8, n_ews).clip(50_000_000, 5_000_000_000).astype(int)

    # ── Combine ──────────────────────────────────────────────────────────
    bureau_scores = np.concatenate([normal_bureau, ews_bureau])
    dpd = np.concatenate([normal_dpd, ews_dpd])
    utilization = np.concatenate([normal_utilization, ews_utilization])
    payment_ratio = np.concatenate([normal_payment_ratio, ews_payment_ratio])
    dti = np.concatenate([normal_dti, ews_dti])
    outstanding = np.concatenate([normal_outstanding, ews_outstanding])
    is_ews = [0] * n_normal + [1] * n_ews

    # EWS severity
    ews_severity = []
    for i in range(n_profiles):
        if is_ews[i] == 0:
            ews_severity.append("Green")
        else:
            idx = i - n_normal
            if ews_dpd[idx] >= 90:
                ews_severity.append("Red")
            elif ews_dpd[idx] >= 60:
                ews_severity.append("Orange")
            else:
                ews_severity.append("Yellow")

    # Bureau score trend (last 6 months change)
    bureau_trend = []
    for i in range(n_profiles):
        if is_ews[i] == 0:
            bureau_trend.append(int(rng.integers(-10, 20)))
        else:
            bureau_trend.append(int(rng.integers(-120, -20)))

    disbursement_dates = [
        datetime(2020, 1, 1) + timedelta(days=int(rng.integers(0, 1800)))
        for _ in range(n_profiles)
    ]

    tenors = rng.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], n_profiles)

    df = pd.DataFrame({
        "loan_id": loan_ids,
        "customer_id": selected_customers,
        "loan_type": loan_types,
        "disbursement_date": disbursement_dates,
        "tenor_months": tenors,
        "outstanding_amount": outstanding,
        "bureau_score": bureau_scores,
        "bureau_score_trend_6m": bureau_trend,
        "days_past_due": dpd,
        "utilization_ratio": np.round(utilization, 4),
        "payment_ratio": np.round(payment_ratio, 4),
        "debt_to_income": np.round(dti, 4),
        "is_ews_flag": is_ews,
        "ews_severity": ews_severity,
    })

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["loan_id"] = [f"LOAN-{i:06d}" for i in range(1, len(df) + 1)]

    return df


def generate_alerts_history(
    transactions: pd.DataFrame,
    credit_profiles: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate historical alerts from flagged transactions and EWS triggers."""
    rng = np.random.default_rng(seed)

    alerts = []
    alert_id = 1

    # Fraud alerts
    fraud_txns = transactions[transactions["is_fraud"] == 1].head(500)
    for _, row in fraud_txns.iterrows():
        severity = rng.choice(["High", "Critical"], p=[0.4, 0.6])
        status = rng.choice(["Open", "Investigating", "Resolved", "Escalated"], p=[0.2, 0.3, 0.35, 0.15])
        alerts.append({
            "alert_id": f"ALR-{alert_id:06d}",
            "alert_type": "Fraud Detection",
            "reference_id": row["transaction_id"],
            "customer_id": row["customer_id"],
            "severity": severity,
            "status": status,
            "description": f"Suspicious {row['fraud_type'].replace('_', ' ')} detected - Amount: Rp {row['amount']:,.0f}",
            "created_at": row["timestamp"],
            "risk_score": int(rng.integers(750, 1000)),
            "channel": row["channel"],
        })
        alert_id += 1

    # EWS alerts
    ews_profiles = credit_profiles[credit_profiles["is_ews_flag"] == 1].head(300)
    for _, row in ews_profiles.iterrows():
        sev_map = {"Red": "Critical", "Orange": "High", "Yellow": "Medium"}
        severity = sev_map.get(row["ews_severity"], "Medium")
        status = rng.choice(["Open", "Monitoring", "Resolved", "Escalated"], p=[0.25, 0.35, 0.25, 0.15])
        alerts.append({
            "alert_id": f"ALR-{alert_id:06d}",
            "alert_type": "Early Warning Signal",
            "reference_id": row["loan_id"],
            "customer_id": row["customer_id"],
            "severity": severity,
            "status": status,
            "description": f"EWS {row['ews_severity']}: DPD={row['days_past_due']}d, Bureau={row['bureau_score']}, Utilization={row['utilization_ratio']:.0%}",
            "created_at": datetime(2025, 1, 1) + timedelta(days=int(rng.integers(0, 365))),
            "risk_score": int(rng.integers(500, 950)),
            "channel": "Credit Monitoring",
        })
        alert_id += 1

    return pd.DataFrame(alerts)


def _business_hour_probs():
    """Return probability distribution favoring business hours (6-22)."""
    hours = list(range(6, 23))
    probs = [
        0.02, 0.03, 0.06, 0.08, 0.09, 0.09,  # 6-11
        0.08, 0.08, 0.07, 0.07, 0.06, 0.06,   # 12-17
        0.05, 0.05, 0.04, 0.04, 0.03,          # 18-22
    ]
    total = sum(probs)
    return [p / total for p in probs]


def generate_all_data(output_dir: str = "data/generated", seed: int = 42):
    """Generate all datasets and save to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  DATA GENERATION PIPELINE")
    print("=" * 60)

    print("\n[1/4] Generating customer base...")
    customers = generate_customers(n_customers=10000, seed=seed)
    customers.to_csv(os.path.join(output_dir, "customers.csv"), index=False)
    print(f"  ✓ {len(customers):,} customers generated")

    print("\n[2/4] Generating transactions (with fraud injection)...")
    transactions = generate_transactions(customers, n_transactions=50000, fraud_rate=0.04, seed=seed)
    transactions.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
    n_fraud = transactions["is_fraud"].sum()
    print(f"  ✓ {len(transactions):,} transactions generated")
    print(f"  ✓ {n_fraud:,} fraud transactions ({n_fraud/len(transactions)*100:.1f}%)")

    print("\n[3/4] Generating credit profiles (with EWS signals)...")
    credit_profiles = generate_credit_profiles(customers, n_profiles=10000, ews_rate=0.10, seed=seed)
    credit_profiles.to_csv(os.path.join(output_dir, "credit_profiles.csv"), index=False)
    n_ews = credit_profiles["is_ews_flag"].sum()
    print(f"  ✓ {len(credit_profiles):,} credit profiles generated")
    print(f"  ✓ {n_ews:,} EWS flags ({n_ews/len(credit_profiles)*100:.1f}%)")

    print("\n[4/4] Generating alerts history...")
    alerts = generate_alerts_history(transactions, credit_profiles, seed=seed)
    alerts.to_csv(os.path.join(output_dir, "alerts_history.csv"), index=False)
    print(f"  ✓ {len(alerts):,} historical alerts generated")

    print("\n" + "=" * 60)
    print("  DATA GENERATION COMPLETE")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)

    return customers, transactions, credit_profiles, alerts


if __name__ == "__main__":
    generate_all_data()
