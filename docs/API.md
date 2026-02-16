# Pre-Delinquency Intervention Engine — API Reference

Base URL (default): `http://localhost:8000`  
API prefix: `/api/v1`

---

## 1. Authentication

**Status:** Not implemented in the current MVP.

The API does not require authentication. In production you should:

- Add API key or JWT validation (e.g. via middleware or dependency).
- Use the `API_KEY` environment variable for server-side validation.
- Restrict access with network policies or a reverse proxy (e.g. nginx) that enforces auth.

---

## 2. Endpoints

### 2.1 POST `/api/v1/predict`

Single-customer risk prediction.

| Item | Details |
|------|---------|
| **Method** | `POST` |
| **URL** | `/api/v1/predict` |
| **Content-Type** | `application/json` |

**Request body (JSON):**

```json
{
  "customer_id": "C000123"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `customer_id` | string | Yes | Unique customer identifier (must exist in feature set). |

**Response (200 OK):**

```json
{
  "customer_id": "C000123",
  "risk_score": 78.5,
  "risk_level": "high",
  "prediction_date": "2024-06-30T12:00:00Z",
  "top_risk_factors": [
    "salary_delay (↑, |SHAP|=0.12)",
    "lending_app_amount (↑, |SHAP|=0.08)"
  ],
  "recommended_action": "Proactively offer restructuring / partial payment options via digital channels."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `customer_id` | string | Customer identifier. |
| `risk_score` | number | Score 0–100. |
| `risk_level` | string | `low` \| `medium` \| `high` \| `critical`. |
| `prediction_date` | string (ISO 8601) | When the prediction was generated. |
| `top_risk_factors` | string[] | Top contributing risk factors (e.g. SHAP-based). |
| `recommended_action` | string | Suggested next action. |

**Status codes:**

| Code | Meaning |
|------|--------|
| 200 | Success. |
| 404 | No features found for the given `customer_id`. |
| 422 | Validation error (e.g. missing or invalid body). |
| 500 | Server error (e.g. model not loaded). |

**Error response (4xx/5xx):**

```json
{
  "detail": "No features found for customer_id=NON_EXISTENT"
}
```

---

### 2.2 POST `/api/v1/predict/batch`

Batch risk prediction for multiple customers.

| Item | Details |
|------|---------|
| **Method** | `POST` |
| **URL** | `/api/v1/predict/batch` |
| **Content-Type** | `application/json` |

**Request body (JSON):**

```json
{
  "customer_ids": ["C000001", "C000002", "C000003"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `customer_ids` | string[] | Yes | List of customer IDs (min length 1). |

**Response (200 OK):**

```json
[
  {
    "customer_id": "C000001",
    "risk_score": 45.2,
    "risk_level": "medium",
    "prediction_date": "2024-06-30T12:00:00Z",
    "top_risk_factors": ["salary_delay (↑, |SHAP|=0.05)"],
    "recommended_action": "Send gentle payment reminder and highlight upcoming dues."
  },
  {
    "customer_id": "C000002",
    "risk_score": 82.1,
    "risk_level": "high",
    "prediction_date": "2024-06-30T12:00:00Z",
    "top_risk_factors": ["lending_app_amount (↑, |SHAP|=0.11)"],
    "recommended_action": "Proactively offer restructuring / partial payment options via digital channels."
  }
]
```

Array of objects with the same shape as the single `POST /api/v1/predict` response. Length equals the number of customers that had features (unknown IDs are skipped).

**Status codes:**

| Code | Meaning |
|------|--------|
| 200 | Success (possibly empty list if no IDs had features). |
| 404 | No features found for any of the provided `customer_ids`. |
| 422 | Validation error (e.g. empty list). |
| 500 | Server error. |

**Error response (404):**

```json
{
  "detail": "No features found for any of the provided customer_ids."
}
```

---

### 2.3 GET `/api/v1/customers/{customer_id}/transactions`

Recent transactions for a customer.

| Item | Details |
|------|---------|
| **Method** | `GET` |
| **URL** | `/api/v1/customers/{customer_id}/transactions` |
| **Query params** | `limit` (optional, default 50, range 1–500). |

**Example request:**  
`GET /api/v1/customers/C000123/transactions?limit=20`

**Response (200 OK):**

```json
[
  {
    "transaction_id": "C000123-2024-06-15T10:30:00-utility_electricity",
    "customer_id": "C000123",
    "date": "2024-06-15T10:30:00",
    "amount": 2499.0,
    "transaction_type": "debit",
    "category": "utility_electricity",
    "merchant": "BESCOM"
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `transaction_id` | string | Unique transaction id. |
| `customer_id` | string | Customer id. |
| `date` | string (ISO 8601) | Transaction date/time. |
| `amount` | number | Amount in INR. |
| `transaction_type` | string | `credit` \| `debit`. |
| `category` | string | Category (e.g. salary, utility_*, discretionary_*, upi_lending_app). |
| `merchant` | string | Merchant or description. |

**Status codes:**

| Code | Meaning |
|------|--------|
| 200 | Success. |
| 404 | No transactions for this `customer_id`. |
| 503 | Transaction data not available (e.g. CSV missing). |

**Error response (404):**

```json
{
  "detail": "No transactions found for customer_id=UNKNOWN"
}
```

---

### 2.4 GET `/api/v1/high-risk-customers`

List customers above a risk score threshold, sorted by risk score descending.

| Item | Details |
|------|---------|
| **Method** | `GET` |
| **URL** | `/api/v1/high-risk-customers` |
| **Query params** | `threshold` (optional, default 80, 0–100), `limit` (optional, default 100, 1–1000). |

**Example request:**  
`GET /api/v1/high-risk-customers?threshold=70&limit=50`

**Response (200 OK):**

```json
[
  {
    "customer_id": "C000042",
    "risk_score": 92.3,
    "risk_level": "critical",
    "prediction_date": "2024-06-30T12:00:00Z",
    "top_risk_factors": ["salary_delay (↑, |SHAP|=0.15)", "savings_depletion_rate (↑, |SHAP|=0.12)"],
    "recommended_action": "Trigger high-priority outreach via call center and collections team with custom recovery plan."
  },
  {
    "customer_id": "C000017",
    "risk_score": 85.1,
    "risk_level": "high",
    "prediction_date": "2024-06-30T12:00:00Z",
    "top_risk_factors": ["lending_app_amount (↑, |SHAP|=0.10)"],
    "recommended_action": "Proactively offer restructuring / partial payment options via digital channels."
  }
]
```

Same object shape as single predict; array is sorted by `risk_score` descending and limited by `limit`.

**Status codes:**

| Code | Meaning |
|------|--------|
| 200 | Success. |
| 500 | Server error (e.g. model unavailable). |

---

### 2.5 POST `/api/v1/intervention/trigger`

Trigger an intervention workflow for a customer (e.g. reminder, restructuring offer).

| Item | Details |
|------|---------|
| **Method** | `POST` |
| **URL** | `/api/v1/intervention/trigger` |
| **Content-Type** | `application/json` |

**Request body (JSON):**

```json
{
  "customer_id": "C000123",
  "intervention_type": "reminder",
  "channel": "sms",
  "message": "Your EMI is due in 5 days. Pay now to avoid late fees."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `customer_id` | string | Yes | Customer to intervene on. |
| `intervention_type` | string | Yes | e.g. `reminder`, `restructuring_offer`, `collection_call`. |
| `channel` | string | Yes | `sms` \| `email` \| `app`. |
| `message` | string | Yes | Message body to send (or template reference). |

**Response (200 OK):**

```json
{
  "status": "success",
  "message": "Intervention 'reminder' triggered via sms for customer C000123."
}
```

**Status codes:**

| Code | Meaning |
|------|--------|
| 200 | Request accepted; intervention triggered (MVP logs only). |
| 422 | Validation error (invalid body or enum). |

**Error response (422):**

```json
{
  "detail": [
    {
      "loc": ["body", "channel"],
      "msg": "value is not a valid enumeration member; permitted: 'sms', 'email', 'app'",
      "type": "type_error.enum"
    }
  ]
}
```

---

### 2.6 GET `/api/v1/health`

API and model health check.

| Item | Details |
|------|---------|
| **Method** | `GET` |
| **URL** | `/api/v1/health` |

**Response (200 OK):**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Status codes:**

| Code | Meaning |
|------|--------|
| 200 | Service up; `model_loaded` indicates if the risk model is loaded. |
| 500 | Service unhealthy (e.g. model failed to load). |

---

### 2.7 GET `/api/v1/metrics`

Observability metrics (predictions, response times, errors, high-risk history).

| Item | Details |
|------|---------|
| **Method** | `GET` |
| **URL** | `/api/v1/metrics` |

**Response (200 OK):**

```json
{
  "total_predictions": 42,
  "average_prediction_time_seconds": 0.0523,
  "prediction_count": 42,
  "total_requests": 100,
  "total_errors": 2,
  "error_rate": 0.02,
  "average_response_time_ms": 45.12,
  "response_times_sample_size": 100,
  "high_risk_history": [
    {"timestamp": 1739123456.789, "count": 5},
    {"timestamp": 1739123500.123, "count": 6}
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `total_predictions` | number | Total risk predictions made since startup. |
| `average_prediction_time_seconds` | number | Mean prediction latency. |
| `prediction_count` | number | Same as total_predictions (for averaging). |
| `total_requests` | number | Total API requests. |
| `total_errors` | number | Requests that returned 4xx/5xx. |
| `error_rate` | number | total_errors / total_requests. |
| `average_response_time_ms` | number | Mean response time in ms (rolling window). |
| `response_times_sample_size` | number | Number of requests in the response-time sample. |
| `high_risk_history` | array | Recent (timestamp, count) when high-risk-customers was called. |

**Status codes:** 200 only.

---

### 2.8 GET `/api/v1/explain/{customer_id}`

SHAP explanation for a customer's risk prediction.

| Item | Details |
|------|---------|
| **Method** | `GET` |
| **URL** | `/api/v1/explain/{customer_id}` |

**Response (200 OK):**

```json
{
  "customer_id": "C000123",
  "base_value": 0.42,
  "feature_names": ["days_since_salary", "salary_delay", "avg_balance", "..."],
  "shap_values": [0.01, 0.12, -0.05, "..."],
  "feature_contributions": [
    {"feature": "salary_delay", "shap_value": 0.12, "abs_value": 0.12},
    {"feature": "avg_balance", "shap_value": -0.05, "abs_value": 0.05}
  ],
  "force_plot_base64": "<base64-encoded PNG>",
  "waterfall_plot_base64": "<base64-encoded PNG>"
}
```

| Code | Meaning |
|------|--------|
| 200 | Success. |
| 404 | No features found for customer_id. |
| 500 | SHAP computation failed. |

---

### 2.9 POST `/api/v1/generate-message`

Generate a personalized intervention message.

| Item | Details |
|------|---------|
| **Method** | `POST` |
| **URL** | `/api/v1/generate-message` |
| **Content-Type** | `application/json` |

**Request body:**

```json
{
  "customer_id": "C000123",
  "customer_name": "Priya Sharma",
  "risk_score": 88.0,
  "risk_level": "critical",
  "top_risk_factors": ["salary_delay (↑, |SHAP|=0.12)"],
  "channel": "app",
  "language": "en",
  "salary_delay_days": 7,
  "variant": "A"
}
```

**Response (200 OK):**

```json
{
  "customer_id": "C000123",
  "scenario": "salary_delay",
  "channel": "app",
  "language": "en",
  "variant": "A",
  "subject": "We're here to help, Priya",
  "body": "Hi Priya, we noticed your salary credit was 7 days later than usual...",
  "generated_at": "2024-06-30T12:00:00"
}
```

Optional fields: `salary_delay_days`, `savings_drop_pct`, `lending_app_count`, `lending_app_amount`, `failed_payment_count`, `upcoming_emi_amount`, `upcoming_emi_date`, `variant` (`A`/`B`), `scenario`.

---

### 2.10 POST `/api/v1/generate-message/preview`

Same request body as 2.9 but returns both A and B variants for comparison.

**Response (200 OK):**

```json
{
  "A": {"scenario": "salary_delay", "variant": "A", "subject": "...", "body": "..."},
  "B": {"scenario": "salary_delay", "variant": "B", "subject": "...", "body": "..."}
}
```

---

### 2.11 POST `/api/v1/stream/transaction`

Ingest a single streaming transaction, persist it, recalculate risk, and broadcast via WebSocket.

| Item | Details |
|------|---------|
| **Method** | `POST` |
| **URL** | `/api/v1/stream/transaction` |
| **Content-Type** | `application/json` |

**Request body:**

```json
{
  "customer_id": "C000001",
  "date": "2024-06-15T10:30:00",
  "amount": 2499.0,
  "transaction_type": "debit",
  "category": "utility_electricity",
  "merchant": "BESCOM"
}
```

**Response (200 OK):**

```json
{
  "transaction": {"customer_id": "C000001", "date": "...", "amount": 2499.0, "...": "..."},
  "risk": {"customer_id": "C000001", "risk_score": 78.5, "risk_level": "high", "...": "..."}
}
```

---

### 2.12 WebSocket `/api/v1/ws/updates`

Real-time push channel. Connect via `ws://host:port/api/v1/ws/updates`. Server pushes JSON messages when streaming transactions trigger risk recalculations:

```json
{
  "type": "prediction_update",
  "customer_id": "C000001",
  "risk": {"customer_id": "C000001", "risk_score": 78.5, "risk_level": "high", "...": "..."}
}
```

---

### 2.13 Root health (non-versioned)

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/health` | Simple health check; returns `{"status": "healthy", "version": "1.0.0"}`. |

---

## 3. Rate limiting

**Status:** Implemented as in-memory per-IP sliding-window rate limiter.

- Default: **120 requests per 60 seconds** per client IP.
- Configurable via `RATE_LIMIT_MAX` and `RATE_LIMIT_WINDOW` environment variables.
- Returns `429 Too Many Requests` with `Retry-After` header when limits are exceeded.
- For production, consider an external rate limiter (nginx, Kong, or Redis-backed) for distributed deployments.

---

## 4. Webhooks

**Status:** Not implemented.

The API does not support webhooks. Interventions are triggered synchronously via `POST /api/v1/intervention/trigger`; the response confirms acceptance only. Real-time updates are available via the `WebSocket /api/v1/ws/updates` endpoint. For delivery status callbacks, a future design could add webhook URLs and event payloads (e.g. `intervention.sent`, `intervention.failed`).
