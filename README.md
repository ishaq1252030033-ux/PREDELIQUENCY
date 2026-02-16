# Pre-Delinquency Intervention Engine MVP

## Overview

The Pre-Delinquency Intervention Engine is an end-to-end system that identifies banking customers at risk of default and supports early intervention through ML-based risk scoring, behavioral analysis, and automated outreach. It combines synthetic transaction data, engineered features, trained classifiers (XGBoost/LightGBM), and a FastAPI backend with a Streamlit dashboard for monitoring and triggering interventions.

## Features

- **Real-time risk prediction** — Single and batch risk scores (0–100) with risk levels (low/medium/high/critical)
- **Behavioral pattern analysis** — Cash flow, spending, borrowing, and payment-behavior features; SHAP-based top risk factors
- **Automated intervention triggers** — API to trigger SMS/email/app interventions with configurable messages
- **Interactive dashboard** — Streamlit UI with overview, customer search, high-risk list, analytics, and intervention simulator

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    Streamlit Dashboard                    │
                    │  (Overview | Customer Search | High-Risk | Analytics)     │
                    └─────────────────────────────┬───────────────────────────┘
                                                  │ HTTP
                                                  ▼
┌──────────────┐    ┌─────────────────────────────────────────────────────────┐
│  PostgreSQL  │◄───│              FastAPI Backend (port 8000)                 │
│  (optional)  │    │  /api/v1/predict | /predict/batch | /high-risk-customers  │
└──────────────┘    │  /intervention/trigger | /health | /customers/.../tx      │
                    └─────────────────────────────┬───────────────────────────┘
                                                  │
                    ┌─────────────────────────────▼───────────────────────────┐
                    │  RiskPredictor  (loaded model + features + SHAP)         │
                    │  ml/models/risk_model.pkl  |  ml/data/processed/        │
                    └─────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- **Python 3.11+**
- **Docker** (optional, for containerized run)

### Local Setup

```bash
# 1. Clone and enter project
cd PREDELIQUENCY

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment template and edit as needed
copy .env.example .env           # Windows
# cp .env.example .env           # Linux/macOS
```

### Docker Setup

```bash
# Build and run all services (backend, frontend, PostgreSQL)
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

- Backend API: http://localhost:8000  
- Frontend: http://localhost:8501  
- PostgreSQL: localhost:5432 (when enabled)

## Usage

### 1. Generate Synthetic Data

```bash
python ml/data/generate_synthetic_data.py
```

Optional: `-n 1000` for fewer customers, `-o data/sample` to write CSVs elsewhere.

### 2. Validate & Engineer Features

```bash
python ml/data/validate_data.py
python ml/feature_engineering.py
```

### 3. Train Model

```bash
python ml/train_model.py
```

Outputs: `ml/models/risk_model.pkl`, `ml/reports/model_performance.json`, `ml/reports/feature_importance.csv`.

### 4. (Optional) Evaluate Model

```bash
python ml/evaluate_model.py
```

Generates ROC/PR curves, confusion matrix, and evaluation summary in `ml/reports/`.

### 5. Run Backend API

```bash
uvicorn backend.app.main:app --reload
```

### 6. Run Dashboard

```bash
streamlit run frontend/dashboard.py
```

## API Documentation

- **Swagger UI:** http://localhost:8000/docs  
- **ReDoc:** http://localhost:8000/redoc  
- **Full API reference:** [docs/API.md](docs/API.md)

Key endpoints: `POST /api/v1/predict`, `POST /api/v1/predict/batch`, `GET /api/v1/high-risk-customers`, `POST /api/v1/intervention/trigger`, `GET /api/v1/explain/{customer_id}`, `POST /api/v1/generate-message`, `GET /api/v1/metrics`, `WebSocket /api/v1/ws/updates`.

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the production checklist, full environment variable reference, security considerations, and monitoring setup.

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pip install pytest-cov
pytest tests/ --cov=backend --cov=ml --cov-report=term-missing
```

Test modules: feature engineering, risk predictor, API integration (TestClient), message generator, metrics store, config validation.

## Project Structure

```
PREDELIQUENCY/
├── backend/                    # FastAPI application
│   ├── app/
│   │   ├── main.py             # App entry, CORS, middleware, lifespan
│   │   ├── models/             # Pydantic schemas, SQLAlchemy DB models
│   │   ├── services/           # RiskPredictor, business logic
│   │   ├── api/                # Routes (predict, batch, high-risk, intervention, health)
│   │   └── utils/              # init_db, helpers
│   ├── requirements.txt
│   ├── config.py
│   └── Dockerfile
├── ml/                         # ML pipeline
│   ├── data/                   # CSVs: generate_synthetic_data, validate_data
│   ├── processed/              # features.csv (output of feature_engineering)
│   ├── notebooks/              # Jupyter (e.g. feature_analysis)
│   ├── models/                 # risk_model.pkl (trained model)
│   ├── reports/                # model_performance.json, feature_importance.csv, plots
│   ├── feature_engineering.py # FeatureEngineer class
│   ├── train_model.py          # XGBoost, LightGBM, ensemble + SMOTE
│   └── evaluate_model.py      # ROC/PR, confusion matrix, business metrics
├── frontend/
│   ├── dashboard.py            # Streamlit app (overview, search, high-risk, analytics)
│   ├── requirements.txt
│   └── Dockerfile
├── data/                       # Raw, processed, sample data directories
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_risk_predictor.py
│   └── test_api.py             # FastAPI TestClient tests
├── .env.example
├── docker-compose.yml           # backend, frontend, PostgreSQL
├── requirements.txt            # Project-wide Python deps
└── README.md
```

## Configuration

Copy `.env.example` to `.env` and set values. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | SQLAlchemy database URL | `sqlite:///./pre_delinquency.db` |
| `API_KEY` | Shared API secret (required in staging/prod) | (none) |
| `MODEL_PATH` | Path to trained model | `ml/models/risk_model.pkl` |
| `ENVIRONMENT` | `development` / `staging` / `production` | `development` |
| `DEBUG` | Debug mode | `false` |
| `ENABLE_INTERVENTIONS` | Feature flag for intervention workflows | `false` |
| `AUTO_TRIGGER_THRESHOLD` | Auto-trigger risk score (0–100) | `80` |
| `CACHE_TTL` | Prediction cache TTL (seconds) | `3600` |
| `CORS_ORIGINS` | Allowed CORS origins (comma-separated) | `http://localhost:8501,http://localhost:3000` |
| `RATE_LIMIT_MAX` | Max requests per IP per window | `120` |

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the complete environment variable reference.

## Model Performance

Metrics are written to `ml/reports/model_performance.json` after training. Example (synthetic 100-customer run):

- **Best model:** XGBoost (selected by validation ROC-AUC)
- **Validation:** Accuracy ~0.94, Precision 1.0, Recall ~0.67, F1 ~0.80, ROC-AUC 1.0
- **Test set:** Same metrics reported under `test_metrics`
- **Splits:** 70% train (with SMOTE), 15% validation, 15% test

Run `ml/evaluate_model.py` to regenerate ROC curve, PR curve, confusion matrix, and threshold analysis (e.g. optimal threshold for 80% recall).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `RuntimeError: MODEL_PATH does not exist` | Run the training pipeline first: `python ml/train_model.py` to produce `ml/models/risk_model.pkl`. |
| `ModuleNotFoundError: No module named 'backend'` | Run from the project root (`PREDELIQUENCY/`) so that `backend` is on `sys.path`. Use `python -m uvicorn backend.app.main:app --reload`. |
| Feature engineering produces all-zero features | Make sure `ml/data/transactions.csv` exists and the date column parses correctly. Run `python ml/data/validate_data.py` to check data quality. |
| Dashboard "Connection Error" | Ensure the backend is running on the correct port. The dashboard uses `API_BASE_URL` (default `http://localhost:8000/api/v1`). |
| `429 Too Many Requests` | The in-memory rate limiter defaults to 120 req/60s. Increase via `RATE_LIMIT_MAX` and `RATE_LIMIT_WINDOW` env vars. |
| Hindi text appears garbled on Windows console | Windows cmd/PowerShell default to `cp1252`. Use `chcp 65001` before running, or redirect output to a UTF-8 file. The HTTP API returns UTF-8 correctly. |
| SHAP plots blank / explainability error | Ensure `shap>=0.45.0` and `matplotlib>=3.8.0` are installed. Run `pip install -r backend/requirements.txt`. |
| Config validation error on startup | Check `.env` values against [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). In production `DEBUG=false`, `API_KEY` must be set, `DATABASE_URL` must not be SQLite. |

## Future Enhancements

- **Production data:** Replace synthetic CSVs with live transaction feeds and customer master from core banking
- **Real-time features:** Run FeatureEngineer in the API path on recent transactions instead of precomputed features only
- **Intervention tracking:** Persist interventions in DB and track delivery/response (SMS/email gateway integration)
- **A/B testing:** Compare intervention strategies and measure impact on default rates
- **Alerting:** Slack/email alerts for critical risk spikes and failed health checks
- **Lifespan & datetime:** Migrate to FastAPI lifespan and `datetime.now(timezone.utc)` to clear deprecation warnings

## License

MIT

## Contributors

Team Innovators
