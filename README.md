# FraudShield AI — Real-Time Fraud Detection & Early Warning System

An AI-driven real-time fraud detection and early warning system (EWS) for banking, built with Python, scikit-learn, XGBoost, and FastAPI.

## Features

- **50,000+ synthetic transactions** with injected fraud patterns (velocity attacks, geo anomalies, high-risk merchants)
- **10,000+ credit/loan profiles** with early warning signals (declining bureau scores, rising DPD, high utilization)
- **Machine Learning Models**: Random Forest + XGBoost ensemble for fraud detection; XGBoost for EWS
- **Real-time Risk Scoring** (0–1000 scale) with auto-block/flag/approve decisions
- **Premium Web Dashboard** with live charts, transaction monitor, alerts, and model performance analytics
- **RESTful API** powered by FastAPI with interactive docs

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (generate data → train models → launch server)
python main.py
```

Then open **http://localhost:8000** in your browser.

## Project Structure

```
fraud-detection-system/
├── main.py                  # Orchestrator (run this!)
├── requirements.txt
├── data/
│   ├── data_generator.py    # Synthetic data generation
│   └── generated/           # CSV outputs (auto-created)
├── models/
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── trained/             # Saved models (auto-created)
├── engine/
│   └── risk_engine.py       # Real-time scoring engine
├── api/
│   └── app.py               # FastAPI backend
└── static/
    ├── index.html            # Dashboard UI
    ├── styles.css            # Dark glassmorphism design
    └── app.js                # Dashboard interactivity
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web dashboard |
| GET | `/api/dashboard/stats` | Aggregated KPIs and chart data |
| GET | `/api/transactions` | Paginated transaction list |
| GET | `/api/alerts` | Active fraud & EWS alerts |
| GET | `/api/model/performance` | Model metrics & feature importance |
| POST | `/api/score-transaction` | Score a single transaction |
| GET | `/api/timeline` | Implementation roadmap |
| GET | `/api/ews/profiles` | Credit profiles with EWS flags |

## Tech Stack

- **Data**: pandas, NumPy
- **ML**: scikit-learn, XGBoost
- **API**: FastAPI, Uvicorn
- **Dashboard**: HTML5, CSS3, JavaScript, Chart.js
- **Design**: Dark glassmorphism, Inter font, gradient accents
