# Pre-Delinquency Intervention Engine — Deployment Guide

This document covers production deployment checklist, environment variables, security, monitoring, and backups.

---

## 1. Production deployment checklist

### Pre-deployment

- [ ] **Code & config**
  - [ ] All secrets and credentials removed from code; use env vars or secret manager.
  - [ ] `.env` (or equivalent) populated for production; do not commit `.env` to version control.
  - [ ] `DEBUG=false`, `ENVIRONMENT=production` (or similar) set.
  - [ ] API base URL / CORS origins updated for production frontend and allowed domains.

- [ ] **Data & model**
  - [ ] Trained model artifact (`risk_model.pkl`) built from production-grade data and versioned.
  - [ ] Feature pipeline (e.g. `feature_engineering.py`) aligned with training; features available for all scored customers.
  - [ ] Database (if used) migrations applied; schema matches application.

- [ ] **Infrastructure**
  - [ ] Python 3.11+ (or version used in Docker) available on target host or in image.
  - [ ] Sufficient CPU/memory for API and model (SHAP can be memory-heavy on large batches).
  - [ ] Persistent storage for model files, DB data, and logs (if applicable).

### Deployment steps

- [ ] Build backend image (or deploy code) with correct `MODEL_PATH` and artifact location.
- [ ] Run backend (e.g. `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`) behind a reverse proxy (nginx/traefik) with TLS.
- [ ] Deploy frontend (Streamlit) with `API_BASE_URL` pointing to production API URL.
- [ ] If using PostgreSQL: create DB, run migrations, set `DATABASE_URL`; run init/seed if needed.
- [ ] Smoke-test: `GET /health`, `GET /api/v1/health`, one `POST /api/v1/predict` with a known customer_id.

### Post-deployment

- [ ] Verify monitoring and alerting (health checks, error rates, latency).
- [ ] Confirm backup schedule for DB and critical config/artifacts.
- [ ] Document rollback steps and where to find logs.

---

## 2. Environment variables reference

| Variable | Required | Default | Description |
|----------|----------|--------|-------------|
| `DATABASE_URL` | No* | `sqlite:///./pre_delinquency.db` | SQLAlchemy database URL. Use PostgreSQL in staging/production. |
| `API_KEY` | Yes (staging/prod) | (none) | Shared API key for securing endpoints. |
| `MODEL_PATH` | No | `ml/models/risk_model.pkl` | Path to the trained ML model file. |
| `APP_NAME` | No | Pre-Delinquency Intervention Engine | Application name (logging/banners). |
| `DEBUG` | No | `false` | Set to `false` in production (enforced by config validation). |
| `ENVIRONMENT` | No | `development` | `development`, `staging`, or `production`. |
| `API_PREFIX` | No | `/api/v1` | URL prefix for API routes (must match backend router). |
| `ENABLE_INTERVENTIONS` | No | `false` | Feature flag: enable intervention workflows. |
| `AUTO_TRIGGER_THRESHOLD` | No | `80` | Risk score (0–100) above which interventions may auto-trigger. |
| `BATCH_SIZE` | No | `100` | Default batch size for background jobs and batch predictions. |
| `CACHE_TTL` | No | `3600` | Prediction cache TTL in seconds. |
| `MODEL_VERSION` | No | (none) | Deployed model version identifier (e.g. git SHA, semver). |
| `RETRAIN_SCHEDULE` | No | (none) | Retraining schedule (e.g. cron expression or `weekly`). |
| `CORS_ORIGINS` | No | `http://localhost:8501,http://localhost:3000` | Comma-separated allowed CORS origins. |
| `RATE_LIMIT_MAX` | No | `120` | Max requests per IP per window. |
| `RATE_LIMIT_WINDOW` | No | `60` | Rate-limit window in seconds. |

**Docker Compose (PostgreSQL):**

| Variable | Required | Default | Description |
|----------|----------|--------|-------------|
| `POSTGRES_DB` | No | `predelinquency` | Database name. |
| `POSTGRES_USER` | No | `app` | Database user. |
| `POSTGRES_PASSWORD` | No | `changeme` | Database password; **must** be changed in production. |

---

## 3. Security considerations

- **Secrets**
  - Do not commit `.env` or any file containing passwords/API keys.
  - Use a secrets manager (e.g. AWS Secrets Manager, HashiCorp Vault) or platform secrets (e.g. Docker/Kubernetes secrets) in production.

- **Network**
  - Expose the API only through a reverse proxy; do not expose the app server directly to the internet unless necessary.
  - Restrict DB access to the backend (and optionally admin networks); do not expose PostgreSQL publicly.

- **API**
  - Add authentication (API key, JWT, or OAuth) before going to production; validate on every request or via middleware.
  - Enforce HTTPS (TLS) for all public endpoints.
  - Consider rate limiting to mitigate abuse and protect downstream systems.

- **Data**
  - Treat customer IDs and risk scores as sensitive; restrict access and log access appropriately.
  - Ensure PII in logs is redacted or omitted (e.g. no full message bodies in logs if they contain PII).

- **Containers**
  - Run processes as non-root inside containers where possible.
  - Keep base images and dependencies updated for security patches.

---

## 4. Monitoring setup

- **Health checks**
  - **Liveness:** `GET /health` or `GET /api/v1/health` (returns 200 when app and model are up).
  - Use in orchestrator (e.g. Kubernetes liveness/readiness) or load balancer health checks.
  - Fail if the process is hung or the model fails to load.

- **Metrics to track**
  - Request rate, latency (e.g. p50, p95), and error rate per endpoint.
  - Model load success/failure and prediction latency.
  - (If applicable) DB connection pool usage and query latency.

- **Logging**
  - Structured logs (JSON) with level, timestamp, request id, and error details.
  - Centralize logs (e.g. ELK, Loki, cloud logging) and set alerts on error rate and latency.

- **Alerting**
  - Alert when health check fails or error rate exceeds a threshold.
  - Alert on repeated 5xx or dependency (DB, file store) failures.
  - Optional: alert when `model_loaded` is false after startup.

- **Dashboards**
  - Service availability, request counts by endpoint, latency, and error rate.
  - Optional: risk score distribution and intervention trigger counts over time.

---

## 5. Backup procedures

- **Database (PostgreSQL)**
  - **Frequency:** At least daily; consider continuous archiving (WAL) for critical data.
  - **Method:** Use `pg_dump` (logical) or filesystem/snapshot backups with consistent stop or WAL archiving.
  - **Retention:** Keep 7–30 days of daily backups; retain monthly backups longer per policy.
  - **Restore:** Document restore steps (restore base backup + WAL, or restore from `pg_dump`).
  - **Testing:** Periodically verify restores in a non-production environment.

- **Application artifacts**
  - **Model files:** Back up `ml/models/` (e.g. `risk_model.pkl`) on each model release; store versioned copies in durable storage (S3, artifact store).
  - **Config:** Back up production `.env` (or equivalent) in a secure, access-controlled store; do not store in plain text in repos.

- **Recovery**
  - Document recovery steps: restore DB, restore/attach correct model version, restart services, run smoke tests.
  - Define RTO/RPO and assign ownership for backup and restore operations.
