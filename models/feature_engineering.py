"""
Feature Engineering Pipeline for Fraud Detection & EWS Models.

Extracts and transforms raw transaction/credit data into ML-ready features.
"""

import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt


def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on Earth (km)."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


# ── Indonesian city coordinates (for home-city distance calculation) ─────
CITY_COORDS = {
    "Jakarta": (-6.2088, 106.8456),
    "Surabaya": (-7.2575, 112.7521),
    "Bandung": (-6.9175, 107.6191),
    "Medan": (3.5952, 98.6722),
    "Semarang": (-6.9666, 110.4196),
    "Makassar": (-5.1477, 119.4327),
    "Palembang": (-2.9761, 104.7754),
    "Denpasar": (-8.6705, 115.2126),
    "Yogyakarta": (-7.7956, 110.3695),
    "Malang": (-7.9786, 112.6304),
}


def engineer_transaction_features(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Engineer features for fraud detection model.

    Features created:
    - amount_log: Log-transformed transaction amount
    - hour_of_day: Transaction hour (0-23)
    - is_weekend: Whether transaction occurred on weekend
    - is_night: Whether transaction occurred between midnight and 6 AM
    - merchant_risk_score: Risk score based on merchant category
    - channel_encoded: Numeric encoding of channel
    - device_encoded: Numeric encoding of device type
    - geo_distance_from_home: Distance from customer's home city (km)
    - amount_zscore: Z-score of amount relative to customer's historical avg
    - velocity_1h: Number of transactions by same customer in rolling 1-hour window
    - velocity_amount_1h: Total amount by same customer in rolling 1-hour window
    """
    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Time features ────────────────────────────────────────────────────
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] < 6)).astype(int)

    # ── Amount features ──────────────────────────────────────────────────
    df["amount_log"] = np.log1p(df["amount"])

    customer_avg = df.groupby("customer_id")["amount"].transform("mean")
    customer_std = df.groupby("customer_id")["amount"].transform("std").fillna(1)
    df["amount_zscore"] = (df["amount"] - customer_avg) / customer_std

    # ── Merchant risk scoring ────────────────────────────────────────────
    merchant_risk_map = {
        "Grocery": 0.1, "Restaurant": 0.15, "Healthcare": 0.1,
        "Education": 0.1, "Utilities": 0.1, "Insurance": 0.1,
        "Gas Station": 0.2, "Subscription": 0.2, "Entertainment": 0.25,
        "Electronics": 0.3, "Online Shopping": 0.35, "Travel": 0.4,
        "ATM Withdrawal": 0.45, "Transfer": 0.5, "Investment": 0.3,
        "Online Gambling": 0.95, "Crypto Exchange": 0.90,
        "Foreign Wire Transfer": 0.85, "Cash Advance": 0.80,
    }
    df["merchant_risk_score"] = df["merchant_category"].map(merchant_risk_map).fillna(0.5)

    # ── Channel encoding ─────────────────────────────────────────────────
    channel_map = {
        "Branch": 0, "ATM": 1, "EDC": 2, "Internet Banking": 3,
        "Mobile Banking": 4, "Livin'": 5, "Kopra": 6,
    }
    df["channel_encoded"] = df["channel"].map(channel_map).fillna(3)

    # ── Device encoding ──────────────────────────────────────────────────
    device_map = {
        "ATM Terminal": 0, "EDC Terminal": 1, "Web Browser": 2,
        "iOS": 3, "Android": 4,
    }
    df["device_encoded"] = df["device_type"].map(device_map).fillna(2)

    # ── Geographic distance from home ────────────────────────────────────
    df = df.merge(customers[["customer_id", "home_city"]], on="customer_id", how="left")
    df["home_lat"] = df["home_city"].map(lambda c: CITY_COORDS.get(c, (0, 0))[0])
    df["home_lon"] = df["home_city"].map(lambda c: CITY_COORDS.get(c, (0, 0))[1])
    df["geo_distance_from_home"] = df.apply(
        lambda r: haversine(r["home_lat"], r["home_lon"], r["latitude"], r["longitude"]),
        axis=1,
    )
    df["geo_distance_log"] = np.log1p(df["geo_distance_from_home"])

    # ── Velocity features (simplified rolling count) ─────────────────────
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    df["velocity_1h"] = (
        df.groupby("customer_id")["timestamp"]
        .transform(lambda x: x.diff().dt.total_seconds().fillna(9999).le(3600).rolling(5, min_periods=1).sum())
    )
    df["velocity_amount_1h"] = (
        df.groupby("customer_id")["amount"]
        .transform(lambda x: x.rolling(5, min_periods=1).sum())
    )
    df["velocity_amount_1h_log"] = np.log1p(df["velocity_amount_1h"])

    # ── Select final feature columns ─────────────────────────────────────
    feature_cols = [
        "amount_log", "hour_of_day", "day_of_week", "is_weekend", "is_night",
        "merchant_risk_score", "channel_encoded", "device_encoded",
        "geo_distance_log", "amount_zscore",
        "velocity_1h", "velocity_amount_1h_log",
    ]

    return df, feature_cols


def engineer_credit_features(credit_profiles: pd.DataFrame) -> tuple:
    """
    Engineer features for Early Warning System model.

    Features created from credit profile data for default prediction.
    """
    df = credit_profiles.copy()

    # ── Derived features ─────────────────────────────────────────────────
    df["dpd_bucket"] = pd.cut(
        df["days_past_due"],
        bins=[-1, 0, 30, 60, 90, 180, 9999],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(int)

    df["bureau_score_normalized"] = (df["bureau_score"] - 300) / (850 - 300)
    df["bureau_trend_negative"] = (df["bureau_score_trend_6m"] < 0).astype(int)
    df["bureau_trend_magnitude"] = np.abs(df["bureau_score_trend_6m"])

    df["high_utilization"] = (df["utilization_ratio"] > 0.75).astype(int)
    df["low_payment_ratio"] = (df["payment_ratio"] < 0.5).astype(int)
    df["high_dti"] = (df["debt_to_income"] > 0.6).astype(int)

    df["outstanding_log"] = np.log1p(df["outstanding_amount"])

    df["risk_composite"] = (
        0.25 * (1 - df["bureau_score_normalized"])
        + 0.25 * df["utilization_ratio"]
        + 0.20 * (1 - df["payment_ratio"])
        + 0.15 * df["debt_to_income"]
        + 0.15 * (df["days_past_due"] / 180).clip(0, 1)
    )

    # ── Loan type encoding ───────────────────────────────────────────────
    loan_type_map = {"KPR": 0, "KKB": 1, "KTA": 2, "Kartu Kredit": 3, "KMK": 4, "KI": 5}
    df["loan_type_encoded"] = df["loan_type"].map(loan_type_map).fillna(2)

    feature_cols = [
        "bureau_score_normalized", "bureau_trend_magnitude", "bureau_trend_negative",
        "days_past_due", "utilization_ratio", "payment_ratio",
        "debt_to_income", "outstanding_log", "loan_type_encoded",
        "high_utilization", "low_payment_ratio", "high_dti",
        "risk_composite",
    ]

    return df, feature_cols
