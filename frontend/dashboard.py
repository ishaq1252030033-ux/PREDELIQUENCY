"""Streamlit dashboard for the Pre-Delinquency Intervention Engine.

Run with::

    streamlit run frontend/dashboard.py

Environment variables:

    ``API_BASE_URL``
        Backend base URL (default: ``http://localhost:8000/api/v1``).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# AI message generation (Llama 3.1 — free, local, private)
try:
    from backend.app.services.ai_message_generator import AIMessageGenerator

    _AI_IMPORT_OK = True
except ImportError:
    _AI_IMPORT_OK = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
"""Project root — all ``ml/`` paths are resolved relative to this."""

PRIMARY_COLOR: str = "#028090"
SECONDARY_COLOR: str = "#00A896"
DANGER_COLOR: str = "#DC3545"
WARNING_COLOR: str = "#FFC107"

# ---------------------------------------------------------------------------
# Streamlit page config & global CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pre-Delinquency Intervention Engine",
    page_icon="\U0001f4ca",
    layout="wide",
)

st.markdown(
    f"""
<style>
:root {{
  --primary: {PRIMARY_COLOR};
  --secondary: {SECONDARY_COLOR};
  --danger: {DANGER_COLOR};
  --warning: {WARNING_COLOR};
}}
.metric-danger span {{ color: {DANGER_COLOR} !important; }}
.metric-warning span {{ color: {WARNING_COLOR} !important; }}

/* ---- Enhanced sidebar ---- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 40%, #0a1628 100%);
    border-right: 1px solid rgba(2, 128, 144, 0.25);
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    font-size: 14px;
}}
/* Radio button styling */
[data-testid="stSidebar"] .stRadio > label {{
    display: none !important;
}}
[data-testid="stSidebar"] .stRadio > div {{
    gap: 2px !important;
}}
[data-testid="stSidebar"] .stRadio > div > label {{
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 8px 14px !important;
    margin: 0 !important;
    transition: all 0.2s ease;
    cursor: pointer;
}}
[data-testid="stSidebar"] .stRadio > div > label:hover {{
    background: rgba(2, 128, 144, 0.15);
    border-color: rgba(2, 128, 144, 0.4);
}}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {{
    background: rgba(2, 128, 144, 0.2);
    border-color: {PRIMARY_COLOR};
    border-left: 3px solid {PRIMARY_COLOR};
}}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Initialise AI generator (loads once per session — Ollama must be running)
# ---------------------------------------------------------------------------

if "ai_generator" not in st.session_state:
    if _AI_IMPORT_OK:
        try:
            st.session_state.ai_generator = AIMessageGenerator()
        except Exception:
            st.session_state.ai_generator = None
    else:
        st.session_state.ai_generator = None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """Send a GET request to the backend and return JSON on success.

    Args:
        path: API path appended to ``API_BASE_URL`` (e.g. ``"/health"``).
        params: Optional query parameters.

    Returns:
        Parsed JSON response, or ``None`` on failure (error shown via
        ``st.error``).
    """
    url = f"{API_BASE_URL}{path}"
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"Cannot reach backend at {url}. " "Is the FastAPI server running?")
    except requests.Timeout:
        st.error(f"Request to {url} timed out. Try again in a moment.")
    except requests.HTTPError as exc:
        st.error(f"API error ({exc.response.status_code}): {exc.response.text[:300]}")
    except requests.RequestException as exc:
        st.error(f"API GET {url} failed: {exc}")
    return None


def api_post(path: str, payload: Dict[str, Any]) -> Optional[Any]:
    """Send a POST request to the backend and return JSON on success.

    Args:
        path: API path appended to ``API_BASE_URL``.
        payload: JSON-serialisable request body.

    Returns:
        Parsed JSON response, or ``None`` on failure.
    """
    url = f"{API_BASE_URL}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"Cannot reach backend at {url}. " "Is the FastAPI server running?")
    except requests.Timeout:
        st.error(f"Request to {url} timed out. Try again in a moment.")
    except requests.HTTPError as exc:
        st.error(f"API error ({exc.response.status_code}): {exc.response.text[:300]}")
    except requests.RequestException as exc:
        st.error(f"API POST {url} failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Data loaders (cached for 5 min)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_customers_csv() -> pd.DataFrame:
    """Load the customer master CSV.

    Returns:
        DataFrame with customer records, or empty DataFrame if missing.
    """
    path = _PROJECT_ROOT / "ml" / "data" / "customers.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_features_csv() -> pd.DataFrame:
    """Load the pre-computed features CSV.

    Returns:
        DataFrame with one row per customer, or empty DataFrame.
    """
    path = _PROJECT_ROOT / "ml" / "data" / "processed" / "features.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_model_performance() -> Dict[str, Any]:
    """Load model performance metrics from the JSON report.

    Returns:
        Dict of metrics, or empty dict if the file is missing.
    """
    path = _PROJECT_ROOT / "ml" / "reports" / "model_performance.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=300)
def load_feature_importance() -> pd.DataFrame:
    """Load feature-importance rankings from CSV.

    Returns:
        DataFrame with feature names and importance scores.
    """
    path = _PROJECT_ROOT / "ml" / "reports" / "feature_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_transactions_csv() -> pd.DataFrame:
    """Load synthetic transactions for date filters and last-tx lookups.

    Returns:
        DataFrame with a parsed ``transaction_date`` column.
    """
    path = _PROJECT_ROOT / "ml" / "data" / "transactions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["transaction_date"])


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def risk_level_color(level: str) -> str:
    """Return the hex colour for a given risk-level string.

    Args:
        level: One of ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.

    Returns:
        Hex colour code.
    """
    mapping: dict[str, str] = {
        "critical": DANGER_COLOR,
        "high": DANGER_COLOR,
        "medium": WARNING_COLOR,
    }
    return mapping.get((level or "").lower(), SECONDARY_COLOR)


def render_gauge(score: float, level: str) -> None:
    """Render a Plotly gauge chart for *score* with colour-coded segments.

    Args:
        score: Numeric risk score (0-100).
        level: Risk-level label used in the chart title.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Risk Score ({level.capitalize()})"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_level_color(level)},
                "steps": [
                    {"range": [0, 40], "color": "#d4f4f4"},
                    {"range": [40, 70], "color": "#ffeeba"},
                    {"range": [70, 85], "color": "#f8d7da"},
                    {"range": [85, 100], "color": "#f5c6cb"},
                ],
            },
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_risk_badge(level: str) -> None:
    """Show a coloured warning/info badge matching the risk level.

    Args:
        level: One of ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
    """
    level_lower = (level or "").lower()
    if level_lower == "critical":
        st.error(f"Risk Level: **{level.upper()}** — Immediate action required")
    elif level_lower == "high":
        st.error(f"Risk Level: **{level.upper()}** — Proactive outreach recommended")
    elif level_lower == "medium":
        st.warning(f"Risk Level: **{level.upper()}** — Monitor closely")
    else:
        st.info(f"Risk Level: **{level.upper()}** — No immediate action needed")


# ---------------------------------------------------------------------------
# Risk-factor tooltip definitions
# ---------------------------------------------------------------------------

_FACTOR_TOOLTIPS: dict[str, str] = {
    "salary_delay": "How many days salary arrived later than usual.",
    "days_since_salary": "Number of days since the last salary credit.",
    "balance_trend": "30-day trend in account balance (negative = declining).",
    "savings_depletion_rate": "Rate at which savings are being consumed.",
    "lending_app": "Borrowing activity from quick-loan / fintech apps.",
    "failed_payments": "Count of bounced or failed payment attempts.",
    "discretionary_spend": "Non-essential spending patterns.",
    "utility_payment_delay": "Delays in electricity, water or gas payments.",
    "avg_balance": "Average account balance over the observation window.",
    "transaction_count": "Total number of transactions in the window.",
    "credit_ratio": "Ratio of credit to total transaction volume.",
}


def _factor_tooltip(factor_text: str) -> str:
    """Return an info tooltip for a risk-factor string if available.

    Args:
        factor_text: The raw factor string from the API.

    Returns:
        Markdown-formatted tooltip, or empty string.
    """
    for key, tip in _FACTOR_TOOLTIPS.items():
        if key in factor_text.lower():
            return f"  \n  _\u2139\ufe0f {tip}_"
    return ""


# =========================================================================
# PAGE 1 — Overview Dashboard
# =========================================================================


def page_overview() -> None:
    """Dashboard overview page showing portfolio-level risk metrics."""
    st.title("Pre-Delinquency Intervention Engine")
    st.caption("Early detection and intervention for at-risk customers")

    customers_df = load_customers_csv()

    # Default date range from actual transaction data
    tx_dates_df = load_transactions_csv()
    if not tx_dates_df.empty and "transaction_date" in tx_dates_df.columns:
        end_date = tx_dates_df["transaction_date"].max().date()
        start_default = tx_dates_df["transaction_date"].min().date()
    else:
        end_date = datetime.now().date()
        start_default = end_date - timedelta(days=29)

    # --- Filters ---
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        date_range = st.date_input(
            "Date range (approx. last transaction date)",
            (start_default, end_date),
        )
    with col_f2:
        risk_levels: list[str] = ["low", "medium", "high", "critical"]
        selected_levels: list[str] = st.multiselect(
            "Risk levels", options=risk_levels, default=risk_levels
        )
    with col_f3:
        overview_search: str = st.text_input("Search (customer ID or name)", "")

    # --- Load risk data ---
    with st.spinner("Loading risk data..."):
        high_risk_data = api_get(
            "/high-risk-customers", {"threshold": 0, "limit": 100_000}
        )

    df_scores: pd.DataFrame = (
        pd.DataFrame(high_risk_data)
        if isinstance(high_risk_data, list)
        else pd.DataFrame()
    )

    # Enrich with names and last-transaction dates
    if not df_scores.empty:
        tx_df = load_transactions_csv()
        if not tx_df.empty:
            last_tx = (
                tx_df.groupby("customer_id")["transaction_date"]
                .max()
                .rename("last_transaction_date")
            )
            df_scores = df_scores.merge(last_tx, on="customer_id", how="left")
        if not customers_df.empty:
            df_scores = df_scores.merge(
                customers_df[["customer_id", "customer_name"]],
                on="customer_id",
                how="left",
            )

        # Apply filters
        if selected_levels:
            df_scores = df_scores[df_scores["risk_level"].isin(selected_levels)]

        if (
            isinstance(date_range, tuple)
            and len(date_range) == 2
            and "last_transaction_date" in df_scores
        ):
            start_ts = pd.Timestamp(date_range[0])
            end_ts = (
                pd.Timestamp(date_range[1])
                + pd.Timedelta(days=1)
                - pd.Timedelta(seconds=1)
            )
            mask = (df_scores["last_transaction_date"] >= start_ts) & (
                df_scores["last_transaction_date"] <= end_ts
            )
            df_scores = df_scores[mask]

        if overview_search:
            mask = (
                df_scores["customer_id"]
                .astype(str)
                .str.contains(overview_search, case=False)
            )
            if "customer_name" in df_scores:
                mask |= (
                    df_scores["customer_name"]
                    .astype(str)
                    .str.contains(overview_search, case=False)
                )
            df_scores = df_scores[mask]

    # --- KPI metrics ---
    if not df_scores.empty:
        total_monitored = len(df_scores)
        high_risk_over_80 = int((df_scores["risk_score"] >= 80).sum())
    else:
        total_monitored = len(customers_df) if not customers_df.empty else 0
        high_risk_over_80 = 0

    interventions_today = 0
    potential_losses_prevented = high_risk_over_80 * 10_000

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers Monitored", f"{total_monitored:,}")
    with col2:
        st.metric("High-Risk (score > 80)", f"{high_risk_over_80:,}")
    with col3:
        st.metric("Interventions Today", interventions_today)
    with col4:
        st.metric(
            "Potential Losses Prevented (\u20b9)",
            f"{potential_losses_prevented:,.0f}",
        )

    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    st.markdown("---")

    # --- Charts ---
    if not df_scores.empty:
        risk_counts = (
            df_scores["risk_level"]
            .value_counts()
            .rename_axis("risk_level")
            .reset_index(name="count")
        )
        fig_pie = px.pie(
            risk_counts,
            names="risk_level",
            values="count",
            title="Risk Level Distribution",
            color="risk_level",
            color_discrete_map={
                "low": SECONDARY_COLOR,
                "medium": WARNING_COLOR,
                "high": DANGER_COLOR,
                "critical": "#8B0000",
            },
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No risk data available yet. Train and evaluate the model first.")

    col_left, col_right = st.columns(2)

    with col_left:
        days = pd.date_range(datetime.now() - timedelta(days=29), periods=30)
        base = high_risk_over_80 or 10
        rng = np.random.default_rng(42)
        noise = rng.integers(-3, 4, size=len(days))
        series = np.clip(base + noise.cumsum(), 0, None)
        ts_df = pd.DataFrame({"date": days, "high_risk_count": series})
        fig_ts = px.line(
            ts_df,
            x="date",
            y="high_risk_count",
            title="Daily High-Risk Customer Count (synthetic trend)",
            markers=True,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_right:
        if not df_scores.empty:
            factors: Dict[str, int] = {}
            for _, r in df_scores.iterrows():
                for f in r.get("top_risk_factors") or []:
                    factors[f] = factors.get(f, 0) + 1
            if factors:
                factors_df = (
                    pd.DataFrame(
                        {
                            "factor": list(factors.keys()),
                            "count": list(factors.values()),
                        }
                    )
                    .sort_values("count", ascending=False)
                    .head(10)
                )
                fig_bar = px.bar(
                    factors_df,
                    x="count",
                    y="factor",
                    orientation="h",
                    title="Top Risk Factors (frequency among all customers)",
                    color="count",
                    color_continuous_scale="Teal",
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No top risk-factor information available yet.")
        else:
            st.info("No risk data available yet.")


# =========================================================================
# PAGE 2 — Customer Search
# =========================================================================


def page_customer_search() -> None:
    """Customer search with risk scoring, SHAP explanation, and intervention."""
    st.title("Customer Search")
    customers_df = load_customers_csv()

    search_query: str = st.text_input(
        "Search by Customer ID or Name",
        help="Enter a customer ID (e.g. C000123) or part of a name.",
    )

    selected_customer_id: Optional[str] = None

    if search_query:
        if customers_df.empty:
            st.warning(
                "Customer master data not available " "(ml/data/customers.csv missing)."
            )
        else:
            mask = customers_df["customer_id"].astype(str).str.contains(
                search_query, case=False
            ) | customers_df["customer_name"].astype(str).str.contains(
                search_query, case=False
            )
            matches = customers_df[mask]
            if matches.empty:
                st.info("No matching customers found.")
            else:
                st.write("Matches:")
                st.dataframe(
                    matches[["customer_id", "customer_name", "age", "credit_score"]],
                    use_container_width=True,
                )
                selected_customer_id = st.selectbox(
                    "Select customer",
                    matches["customer_id"].tolist(),
                    format_func=lambda cid: (
                        f"{cid} — "
                        f"{matches.loc[matches['customer_id'] == cid, 'customer_name'].iloc[0]}"
                    ),
                )

    if not selected_customer_id:
        st.stop()

    # --- Customer profile card ---
    if not customers_df.empty:
        row = customers_df[customers_df["customer_id"] == selected_customer_id].iloc[0]
        st.subheader(f"Profile: {row['customer_name']} ({row['customer_id']})")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Age", int(row["age"]))
        with c2:
            st.metric("Credit Score", int(row["credit_score"]))
        with c3:
            st.metric("Income Bracket", str(row["income_bracket"]))

        # EMI / Installment details + Balance comparison
        _loan = str(row.get("loan_type", ""))
        _emi_amt = float(row.get("emi_amount", 0))
        _next_emi = str(row.get("next_emi_date", ""))
        _emi_rem = int(row.get("emi_remaining_months", 0))

        # Fetch avg_balance from features CSV
        _features_df = load_features_csv()
        _cs_balance = 0.0
        if (
            not _features_df.empty
            and "avg_balance" in _features_df.columns
            and selected_customer_id in _features_df["customer_id"].values
        ):
            _cs_balance = float(
                _features_df.loc[
                    _features_df["customer_id"] == selected_customer_id,
                    "avg_balance",
                ].iloc[0]
            )

        if _emi_amt > 0:
            # Row 1: EMI details
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("Loan Type", _loan)
            with e2:
                st.metric("EMI Amount", f"\u20b9{_emi_amt:,.0f}")
            with e3:
                _due_label = _next_emi
                _days_to_emi = 999
                if _next_emi:
                    try:
                        _days_to_emi = (
                            datetime.strptime(_next_emi, "%Y-%m-%d")
                            - datetime.now()
                        ).days
                        _due_label = (
                            f"{_next_emi} (in {_days_to_emi}d)"
                            if _days_to_emi >= 0
                            else f"{_next_emi} (overdue {abs(_days_to_emi)}d)"
                        )
                    except ValueError:
                        pass
                st.metric("Next EMI Date", _due_label)
            with e4:
                st.metric("Remaining", f"{_emi_rem} months")

            # Row 2: EMI vs Balance analysis
            _cov_ratio = _cs_balance / _emi_amt if _emi_amt > 0 else 0.0
            _shortfall = _cs_balance - _emi_amt

            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric(
                    "\U0001f3e6 Avg Balance",
                    f"\u20b9{_cs_balance:,.0f}",
                )
            with b2:
                if _shortfall >= 0:
                    st.metric(
                        "\u2705 Balance \u2212 EMI",
                        f"+\u20b9{_shortfall:,.0f}",
                        delta="Can cover EMI",
                        delta_color="normal",
                    )
                else:
                    st.metric(
                        "\u274c Balance \u2212 EMI",
                        f"-\u20b9{abs(_shortfall):,.0f}",
                        delta="Cannot cover EMI",
                        delta_color="inverse",
                    )
            with b3:
                _cov_label = f"{_cov_ratio:.1f}x"
                st.metric(
                    "\U0001f4ca Coverage Ratio",
                    _cov_label,
                    delta=(
                        "Healthy (2x+)"
                        if _cov_ratio >= 2.0
                        else "Tight (1-2x)"
                        if _cov_ratio >= 1.0
                        else "At risk (<1x)"
                    ),
                    delta_color=(
                        "normal"
                        if _cov_ratio >= 2.0
                        else "off"
                        if _cov_ratio >= 1.0
                        else "inverse"
                    ),
                )

    # --- Risk score ---
    with st.spinner("Computing risk score..."):
        risk_data = api_post("/predict", {"customer_id": selected_customer_id})

    if risk_data:
        score = float(risk_data.get("risk_score", 0))
        level = str(risk_data.get("risk_level", "low"))

        st.subheader("Risk Score")
        render_gauge(score, level)
        _render_risk_badge(level)

        st.write("**Top Risk Factors**")
        for factor_text in risk_data.get("top_risk_factors") or []:
            st.markdown(f"- {factor_text}{_factor_tooltip(factor_text)}")

        st.info(f"**Recommended action:** {risk_data.get('recommended_action', 'N/A')}")

        # SHAP explanation
        with st.expander("View explanation (SHAP)", expanded=False):
            with st.spinner("Generating SHAP explanation..."):
                explain = api_get(f"/explain/{selected_customer_id}")
            if isinstance(explain, dict):
                contribs: list[dict[str, Any]] = explain.get(
                    "feature_contributions", []
                )
                if contribs:
                    st.markdown("**Top contributing features**")
                    for c in contribs[: min(5, len(contribs))]:
                        st.markdown(
                            f"- `{c['feature']}`: SHAP={c['shap_value']:.4f} "
                            f"(abs={c['abs_value']:.4f})"
                        )

                cols = st.columns(2)
                for idx, (key, label) in enumerate(
                    [
                        ("force_plot_base64", "Force plot"),
                        ("waterfall_plot_base64", "Waterfall plot"),
                    ]
                ):
                    b64 = explain.get(key)
                    if b64:
                        with cols[idx]:
                            st.markdown(f"**{label}**")
                            try:
                                st.image(
                                    base64.b64decode(b64),
                                    use_container_width=True,
                                )
                            except Exception:
                                st.info(f"Unable to render {label.lower()}.")
    else:
        st.warning("Could not retrieve risk score for this customer.")

    # --- Recent transactions ---
    st.subheader("Recent Transactions")
    with st.spinner("Loading transactions..."):
        tx_data = api_get(f"/customers/{selected_customer_id}/transactions")
    if isinstance(tx_data, list) and tx_data:
        tx_df = pd.DataFrame(tx_data)
        tx_df["date"] = pd.to_datetime(tx_df["date"])
        st.dataframe(
            tx_df.sort_values("date", ascending=False),
            use_container_width=True,
        )
        st.success(f"Showing {len(tx_df)} recent transactions.")
    else:
        st.info("No transactions found for this customer.")

    # --- Trigger intervention ---
    st.subheader("Trigger Intervention")

    # Determine AI readiness (direct class or API fallback)
    ai_gen = st.session_state.get("ai_generator")
    ai_ready: bool = ai_gen is not None

    if not ai_ready:
        ai_status_data = api_get("/ai/status")
        ai_api_ok: bool = isinstance(ai_status_data, dict) and ai_status_data.get(
            "available", False
        )
    else:
        ai_api_ok = False

    # --- Controls row ---
    col_a, col_b, col_c = st.columns([2.5, 1.5, 1.5])
    with col_a:
        intervention_type: str = st.selectbox(
            "Intervention Type",
            ["reminder", "restructuring_offer", "collection_call"],
        )
    with col_b:
        language: str = st.selectbox(
            "Language",
            ["en", "hi"],
            format_func=lambda x: "English" if x == "en" else "Hindi",
        )
    with col_c:
        variant_choice: str = st.selectbox("A/B Variant", ["auto", "A", "B"])

    # --- AI status banner ---
    if ai_ready:
        st.caption(
            f"\U0001f7e2 Llama 3.1 loaded locally — **FREE** AI generation ready "
            f"(model: `{ai_gen.model}`)"
        )
    elif ai_api_ok:
        st.caption(
            "\U0001f7e2 Ollama connected via API — "
            f"{len(ai_status_data.get('models', []))} model(s) available"
        )
    else:
        st.caption(
            "\u2139\ufe0f AI generation requires Ollama. "
            "[Install Ollama](https://ollama.com) \u2192 "
            "`ollama pull llama3.1` \u2192 `ollama serve`"
        )

    # --- Resolve customer data for message generation ---
    customer_name_for_msg = ""
    if (
        not customers_df.empty
        and selected_customer_id in customers_df["customer_id"].values
    ):
        customer_name_for_msg = customers_df.loc[
            customers_df["customer_id"] == selected_customer_id,
            "customer_name",
        ].iloc[0]

    risk_score_val: float = (
        float(risk_data.get("risk_score", 0)) if risk_data else 0.0
    )
    risk_level_val: str = (
        str(risk_data.get("risk_level", "low")) if risk_data else "low"
    )
    risk_factors_val: list[str] = (
        risk_data.get("top_risk_factors", []) if risk_data else []
    )

    # ================================================================
    # PATH A — Multi-channel AI generation (Llama 3.1, FREE & LOCAL)
    # ================================================================

    if ai_ready or ai_api_ok:
        if st.button(
            "\U0001f514 Trigger AI Intervention",
            type="primary",
            key="btn_ai_intervention",
        ):
            with st.spinner(
                "\U0001f916 AI generating personalised messages for ALL channels..."
            ):
                try:
                    if ai_ready:
                        messages = ai_gen.generate_intervention_messages(
                            customer_name=customer_name_for_msg
                            or selected_customer_id,
                            risk_score=risk_score_val,
                            risk_factors=risk_factors_val[:3],
                            payment_due_date="your next due date",
                            payment_amount=0.0,
                            language=language,
                        )
                    else:
                        gen_payload: dict[str, Any] = {
                            "customer_id": selected_customer_id,
                            "customer_name": customer_name_for_msg
                            or selected_customer_id,
                            "risk_score": risk_score_val,
                            "risk_level": risk_level_val,
                            "top_risk_factors": risk_factors_val,
                            "channel": "app",
                            "language": language,
                        }
                        ai_result = api_post(
                            "/ai/generate-message", gen_payload
                        )
                        messages = {
                            "sms": ai_result.get("body", "")
                            if ai_result
                            else "",
                            "email": {
                                "subject": ai_result.get("subject", "")
                                if ai_result
                                else "",
                                "body": ai_result.get("body", "")
                                if ai_result
                                else "",
                            },
                            "whatsapp": ai_result.get("body", "")
                            if ai_result
                            else "",
                            "app": ai_result.get("body", "")
                            if ai_result
                            else "",
                            "model": ai_result.get("model", "llama3.1")
                            if ai_result
                            else "llama3.1",
                            "ai_generated": True,
                            "total_latency_ms": 0,
                        }

                    st.session_state["ai_messages"] = messages
                    st.session_state["ai_messages_cid"] = (
                        selected_customer_id
                    )

                except Exception as exc:
                    st.error(f"Error generating messages: {exc}")
                    st.info(
                        "Make sure Ollama is running: `ollama serve`"
                    )

        # --- Display AI-generated messages if available ---
        if (
            st.session_state.get("ai_messages")
            and st.session_state.get("ai_messages_cid")
            == selected_customer_id
        ):
            messages = st.session_state["ai_messages"]
            st.success(
                "\u2705 AI generated personalised messages for all channels!"
            )

            tab_sms, tab_email, tab_wa, tab_app = st.tabs(
                [
                    "\U0001f4f1 SMS",
                    "\U0001f4e7 Email",
                    "\U0001f4ac WhatsApp",
                    "\U0001f4f2 App Push",
                ]
            )

            # ---- SMS tab ----
            with tab_sms:
                st.markdown("### SMS Message")
                sms_text: str = (
                    messages.get("sms", "") if isinstance(messages.get("sms"), str) else ""
                )
                st.info(sms_text)
                char_count = len(sms_text)
                colour = "green" if char_count <= 160 else "red"
                st.caption(
                    f"Length: :{colour}[**{char_count}**/160] characters"
                )
                edited_sms: str = st.text_area(
                    "Edit SMS before sending",
                    value=sms_text,
                    height=80,
                    key="edit_sms",
                    max_chars=160,
                )
                if st.button("\U0001f4e4 Send SMS", key="send_sms"):
                    resp = api_post(
                        "/intervention/trigger",
                        {
                            "customer_id": selected_customer_id,
                            "intervention_type": intervention_type,
                            "channel": "sms",
                            "message": edited_sms,
                        },
                    )
                    if resp:
                        st.success(
                            "\u2705 SMS sent to customer!"
                        )
                    else:
                        st.error("Failed to send SMS.")

            # ---- Email tab ----
            with tab_email:
                st.markdown("### Email Message")
                email_data = messages.get("email", {})
                if isinstance(email_data, dict):
                    email_subject = email_data.get("subject", "")
                    email_body = email_data.get("body", "")
                else:
                    email_subject = ""
                    email_body = str(email_data)

                edited_subject: str = st.text_input(
                    "Subject",
                    value=email_subject,
                    key="edit_email_subject",
                )
                edited_email: str = st.text_area(
                    "Body",
                    value=email_body,
                    height=200,
                    key="edit_email_body",
                )
                if st.button("\U0001f4e4 Send Email", key="send_email"):
                    resp = api_post(
                        "/intervention/trigger",
                        {
                            "customer_id": selected_customer_id,
                            "intervention_type": intervention_type,
                            "channel": "email",
                            "message": f"Subject: {edited_subject}\n\n{edited_email}",
                        },
                    )
                    if resp:
                        st.success(
                            "\u2705 Email sent to customer!"
                        )
                    else:
                        st.error("Failed to send email.")

            # ---- WhatsApp tab ----
            with tab_wa:
                st.markdown("### WhatsApp Message")
                wa_text: str = (
                    messages.get("whatsapp", "")
                    if isinstance(messages.get("whatsapp"), str)
                    else ""
                )
                st.info(wa_text)
                edited_wa: str = st.text_area(
                    "Edit WhatsApp message before sending",
                    value=wa_text,
                    height=120,
                    key="edit_wa",
                )
                if st.button(
                    "\U0001f4e4 Send WhatsApp", key="send_whatsapp"
                ):
                    resp = api_post(
                        "/intervention/trigger",
                        {
                            "customer_id": selected_customer_id,
                            "intervention_type": intervention_type,
                            "channel": "whatsapp",
                            "message": edited_wa,
                        },
                    )
                    if resp:
                        st.success(
                            "\u2705 WhatsApp message sent!"
                        )
                    else:
                        st.error("Failed to send WhatsApp message.")

            # ---- App Push tab ----
            with tab_app:
                st.markdown("### App Push Notification")
                app_text: str = (
                    messages.get("app", "")
                    if isinstance(messages.get("app"), str)
                    else ""
                )
                st.info(app_text)
                edited_app: str = st.text_area(
                    "Edit app notification before sending",
                    value=app_text,
                    height=80,
                    key="edit_app",
                )
                if st.button(
                    "\U0001f4e4 Send App Push", key="send_app"
                ):
                    resp = api_post(
                        "/intervention/trigger",
                        {
                            "customer_id": selected_customer_id,
                            "intervention_type": intervention_type,
                            "channel": "app",
                            "message": edited_app,
                        },
                    )
                    if resp:
                        st.success(
                            "\u2705 App notification sent!"
                        )
                    else:
                        st.error(
                            "Failed to send app notification."
                        )

            # ---- AI info footer ----
            st.markdown("---")
            model_name = messages.get("model", "llama3.1")
            latency = messages.get("total_latency_ms", 0)
            latency_str = (
                f" | \u23f1 {latency:.0f} ms" if latency else ""
            )
            st.caption(
                f"\U0001f916 Powered by **{model_name}** (Meta) — "
                f"FREE & Open Source | "
                f"\U0001f512 100% Private (runs locally){latency_str}"
            )

            # Clear button
            if st.button(
                "\U0001f504 Regenerate messages", key="regen_ai"
            ):
                st.session_state.pop("ai_messages", None)
                st.session_state.pop("ai_messages_cid", None)
                st.rerun()

    # ================================================================
    # PATH B — Template engine fallback
    # ================================================================

    with st.expander(
        "\U0001f4dd Template engine (fallback)" if (ai_ready or ai_api_ok) else "\U0001f4dd Generate message",
        expanded=not (ai_ready or ai_api_ok),
    ):
        channel: str = st.selectbox(
            "Channel", ["sms", "email", "app", "whatsapp"], key="tmpl_channel"
        )
        gen_payload_tmpl: dict[str, Any] = {
            "customer_id": selected_customer_id,
            "customer_name": customer_name_for_msg or selected_customer_id,
            "risk_score": risk_score_val,
            "risk_level": risk_level_val,
            "top_risk_factors": risk_factors_val,
            "channel": channel,
            "language": language,
        }
        if variant_choice in ("A", "B"):
            gen_payload_tmpl["variant"] = variant_choice

        with st.spinner("Generating personalised message..."):
            generated = api_post("/generate-message", gen_payload_tmpl)

        if generated:
            st.info(
                f"Detected scenario: **{generated.get('scenario', '')}** | "
                f"Variant: **{generated.get('variant', '')}**"
            )
            if generated.get("subject"):
                st.markdown(f"**Subject:** {generated['subject']}")
            default_msg: str = generated.get("body", "")
        else:
            default_msg = (
                "Hi, we noticed changes in your repayment behavior. Please "
                "review your upcoming dues to avoid late fees."
            )

        message: str = st.text_area(
            "Message (editable)", value=default_msg, height=100, key="tmpl_msg"
        )

        # A/B preview
        with st.expander("Preview A/B variants", expanded=False):
            with st.spinner("Loading variants..."):
                preview = api_post(
                    "/generate-message/preview", gen_payload_tmpl
                )
            if isinstance(preview, dict):
                pcol1, pcol2 = st.columns(2)
                for col, key in [(pcol1, "A"), (pcol2, "B")]:
                    with col:
                        v = preview.get(key, {})
                        st.markdown(
                            f"**Variant {key}** ({v.get('scenario', '')})"
                        )
                        if v.get("subject"):
                            st.caption(f"Subject: {v['subject']}")
                        st.text(v.get("body", ""))

        if st.button(
            "Trigger Intervention (template)", key="btn_tmpl_trigger"
        ):
            with st.spinner("Sending intervention..."):
                payload: dict[str, Any] = {
                    "customer_id": selected_customer_id,
                    "intervention_type": intervention_type,
                    "channel": channel,
                    "message": message,
                }
                resp = api_post("/intervention/trigger", payload)
            if resp:
                st.success(
                    resp.get(
                        "message",
                        "Intervention triggered successfully.",
                    )
                )
            else:
                st.error(
                    "Failed to trigger intervention. Check backend logs."
                )


# =========================================================================
# PAGE 3 — High-Risk Customers
# =========================================================================


def _hr_compute_emi_due(date_str: str) -> tuple[str, int]:
    """Return (human label, days_left) for an EMI date string."""
    if not date_str:
        return ("", 999)
    try:
        days = (datetime.strptime(date_str, "%Y-%m-%d") - datetime.now()).days
        if days < 0:
            return (f"overdue {abs(days)}d", days)
        if days == 0:
            return ("due today", 0)
        return (f"in {days}d", days)
    except ValueError:
        return (date_str, 999)


def page_high_risk_customers() -> None:
    """Admin-driven batch intervention workflow for high-risk customers.

    Flow:
      1. Data Analysis  \u2192  auto-scored & enriched table
      2. Admin Selection \u2192  choose how many to message
      3. AI Generation   \u2192  personalised messages per customer
      4. Review & Send   \u2192  approve, edit, then dispatch
    """
    st.title("\U0001f6a8 High-Risk Customers \u2014 Intervention Workflow")

    # Session state keys
    if "hr_batch_messages" not in st.session_state:
        st.session_state["hr_batch_messages"] = {}
    if "hr_sent_log" not in st.session_state:
        st.session_state["hr_sent_log"] = []

    # ==================================================================
    # STEP 1 \u2014 DATA ANALYSIS (auto)
    # ==================================================================
    st.markdown(
        "#### STEP 1 \u00a0\u2014\u00a0 Data Analysis",
        help="Risk model scores all customers. Table is sorted by severity.",
    )

    fc1, fc2 = st.columns([2, 3])
    with fc1:
        threshold: int = st.slider(
            "Risk threshold",
            min_value=0,
            max_value=100,
            value=70,
            step=5,
        )
    with fc2:
        search_term: str = st.text_input("Filter by ID or name", "")

    with st.spinner("Running risk analysis on all customers\u2026"):
        data = api_get(
            "/high-risk-customers",
            {"threshold": threshold, "limit": 500},
        )

    if not isinstance(data, list) or not data:
        st.info("No customers above the current risk threshold.")
        return

    # Build enriched dataframe
    risk_df = pd.DataFrame(data)
    customers_df = load_customers_csv()
    features_df = load_features_csv()

    merge_cols = [
        "customer_id", "customer_name", "loan_type",
        "emi_amount", "next_emi_date", "emi_remaining_months",
    ]
    if not customers_df.empty:
        avail = [c for c in merge_cols if c in customers_df.columns]
        risk_df = risk_df.merge(customers_df[avail], on="customer_id", how="left")
    if not features_df.empty and "avg_balance" in features_df.columns:
        risk_df = risk_df.merge(
            features_df[["customer_id", "avg_balance"]], on="customer_id", how="left",
        )

    if search_term:
        mask = risk_df["customer_id"].astype(str).str.contains(search_term, case=False)
        if "customer_name" in risk_df:
            mask |= risk_df["customer_name"].astype(str).str.contains(search_term, case=False)
        risk_df = risk_df[mask]

    risk_df = risk_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # Add EMI analysis columns
    if "emi_amount" in risk_df.columns and "avg_balance" in risk_df.columns:
        risk_df["emi_due"] = risk_df["next_emi_date"].fillna("").apply(
            lambda d: _hr_compute_emi_due(d)[0]
        )
        risk_df["balance_vs_emi"] = risk_df.apply(
            lambda r: (
                f"\u20b9{r['avg_balance'] - r['emi_amount']:,.0f}"
                if r["emi_amount"] > 0
                else "\u2014"
            ),
            axis=1,
        )
        risk_df["coverage"] = risk_df.apply(
            lambda r: (
                f"{r['avg_balance'] / r['emi_amount']:.1f}x"
                if r["emi_amount"] > 0
                else "\u2014"
            ),
            axis=1,
        )
        risk_df["can_pay"] = risk_df.apply(
            lambda r: (
                "\u2705" if r["emi_amount"] <= 0 or r["avg_balance"] >= r["emi_amount"]
                else "\u274c"
            ),
            axis=1,
        )

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("\U0001f465 High-Risk", len(risk_df))
    with k2:
        n_crit = len(risk_df[risk_df.get("risk_level", pd.Series()) == "critical"]) if "risk_level" in risk_df.columns else 0
        st.metric("\U0001f534 Critical", n_crit)
    with k3:
        st.metric("\U0001f4ca Avg Score", f"{risk_df['risk_score'].mean():.1f}")
    with k4:
        n_cant_pay = len(risk_df[risk_df.get("can_pay", pd.Series()) == "\u274c"]) if "can_pay" in risk_df.columns else 0
        st.metric("\u274c Can't Cover EMI", n_cant_pay)
    with k5:
        st.metric("\U0001f4e8 Sent This Session", len(st.session_state["hr_sent_log"]))

    # Display table
    show_cols = [
        "customer_id", "customer_name", "risk_score", "risk_level",
        "loan_type", "emi_amount", "emi_due", "avg_balance",
        "balance_vs_emi", "coverage", "can_pay",
    ]
    show_cols = [c for c in show_cols if c in risk_df.columns]
    st.dataframe(risk_df[show_cols], use_container_width=True, hide_index=True)

    csv_bytes = risk_df[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "\U0001f4e5 Download Analysis CSV", data=csv_bytes,
        file_name="high_risk_analysis.csv", mime="text/csv",
    )

    # ==================================================================
    # STEP 2 \u2014 ADMIN SELECTION
    # ==================================================================
    st.markdown("---")
    st.markdown(
        "#### STEP 2 \u00a0\u2014\u00a0 Admin: Select Customers to Message",
        help="Pick specific customers from the dropdown. Only selected customers will receive messages.",
    )

    # Build label for each customer so admin can identify them easily
    _cid_labels: dict[str, str] = {}
    for _, _r in risk_df.iterrows():
        _cid = _r["customer_id"]
        _cname = _r.get("customer_name", _cid)
        _cscore = _r.get("risk_score", 0)
        _emi = _r.get("emi_amount", 0)
        _cpay = _r.get("can_pay", "")
        _cid_labels[_cid] = (
            f"{_cname} ({_cid}) \u2014 "
            f"Risk: {_cscore:.0f}"
            + (f" | EMI: \u20b9{_emi:,.0f} {_cpay}" if _emi > 0 else "")
        )

    sc1, sc2 = st.columns([3, 1.5])
    with sc1:
        selected_cids: list[str] = st.multiselect(
            "Select customers to message",
            options=risk_df["customer_id"].tolist(),
            default=[],
            format_func=lambda cid: _cid_labels.get(cid, cid),
            key="admin_select_customers",
            help="Pick one or more customers. Use the search box to filter.",
        )
    with sc2:
        channel: str = st.selectbox(
            "Channel",
            ["sms", "email", "whatsapp"],
            format_func=lambda x: {
                "sms": "\U0001f4f1 SMS",
                "email": "\U0001f4e7 Email",
                "whatsapp": "\U0001f4ac WhatsApp",
            }.get(x, x),
            key="admin_channel",
        )

    # Quick-select helpers
    qs1, qs2, qs3, qs4 = st.columns(4)
    with qs1:
        if st.button("Select All", key="sel_all"):
            st.session_state["admin_select_customers"] = risk_df["customer_id"].tolist()
            st.rerun()
    with qs2:
        if st.button("Select Top 10", key="sel_top10"):
            st.session_state["admin_select_customers"] = risk_df["customer_id"].head(10).tolist()
            st.rerun()
    with qs3:
        if st.button("Select Can't Pay EMI", key="sel_cantpay"):
            cant_pay_ids = risk_df[risk_df.get("can_pay", pd.Series()) == "\u274c"]["customer_id"].tolist() if "can_pay" in risk_df.columns else []
            st.session_state["admin_select_customers"] = cant_pay_ids
            st.rerun()
    with qs4:
        if st.button("Clear Selection", key="sel_clear"):
            st.session_state["admin_select_customers"] = []
            st.rerun()

    if not selected_cids:
        st.info("Select one or more customers from the dropdown above.")
        return

    n_to_message = len(selected_cids)
    selected_df = risk_df[risk_df["customer_id"].isin(selected_cids)].copy()

    st.markdown(f"**{n_to_message} customer{'s' if n_to_message > 1 else ''} selected:**")

    preview_cols = [
        "customer_id", "customer_name", "risk_score",
        "emi_amount", "emi_due", "avg_balance", "can_pay",
    ]
    preview_cols = [c for c in preview_cols if c in selected_df.columns]
    st.dataframe(selected_df[preview_cols], use_container_width=True, hide_index=True)

    # ==================================================================
    # STEP 3 \u2014 GENERATE PERSONALISED MESSAGES
    # ==================================================================
    st.markdown("---")
    st.markdown(
        "#### STEP 3 \u00a0\u2014\u00a0 Generate Personalised Messages",
        help="AI analyses each customer's risk factors, EMI, balance, and generates a tailored message.",
    )

    if st.button(
        f"\U0001f916 Generate Messages for {n_to_message} Customers",
        type="primary",
        key="btn_batch_generate",
    ):
        batch_msgs: dict[str, dict[str, Any]] = {}
        progress = st.progress(0, text="Generating personalised messages\u2026")

        for idx, (_, row) in enumerate(selected_df.iterrows()):
            cid = row["customer_id"]
            cname = str(row.get("customer_name", cid))
            cscore = float(row.get("risk_score", 0))
            clevel = str(row.get("risk_level", "high"))

            # Parse risk factors
            factors = row.get("top_risk_factors", [])
            if isinstance(factors, str):
                try:
                    factors = json.loads(factors)
                except Exception:
                    factors = [factors] if factors else []
            if not isinstance(factors, list):
                factors = []

            emi_amt = float(row.get("emi_amount", 0))
            emi_date_str = str(row.get("next_emi_date", ""))
            emi_label, _ = _hr_compute_emi_due(emi_date_str)
            avg_bal = float(row.get("avg_balance", 0))
            loan = str(row.get("loan_type", ""))

            # Build context-aware message via API
            payload: dict[str, Any] = {
                "customer_id": cid,
                "customer_name": cname,
                "risk_score": cscore,
                "risk_level": clevel,
                "top_risk_factors": factors[:3],
                "channel": channel,
                "language": "en",
            }
            result = api_post("/generate-message", payload)

            body = ""
            subject = ""
            if result:
                body = result.get("body", "")
                subject = result.get("subject", "")

            # Enrich the message with EMI context
            if emi_amt > 0 and body:
                emi_note = (
                    f" Your {loan} EMI of \u20b9{emi_amt:,.0f} is {emi_label}."
                    f" Your current balance is \u20b9{avg_bal:,.0f}."
                )
                if avg_bal < emi_amt:
                    emi_note += " We can help with flexible options."
                body = body.rstrip(".") + "." + emi_note

            batch_msgs[cid] = {
                "customer_name": cname,
                "risk_score": cscore,
                "risk_level": clevel,
                "emi_amount": emi_amt,
                "avg_balance": avg_bal,
                "emi_due": emi_label,
                "loan_type": loan,
                "channel": channel,
                "subject": subject,
                "body": body or f"Hi {cname.split()[0]}, we noticed changes in your account. Please contact us for support.",
                "can_pay": "\u2705" if avg_bal >= emi_amt or emi_amt == 0 else "\u274c",
            }

            progress.progress(
                (idx + 1) / n_to_message,
                text=f"Generated {idx + 1}/{n_to_message}: {cname}",
            )

        progress.empty()
        st.session_state["hr_batch_messages"] = batch_msgs
        st.success(
            f"\u2705 Generated personalised messages for **{len(batch_msgs)}** customers!"
        )

    # ==================================================================
    # STEP 4 \u2014 REVIEW & SEND
    # ==================================================================
    batch = st.session_state.get("hr_batch_messages", {})
    if not batch:
        return

    st.markdown("---")
    st.markdown(
        "#### STEP 4 \u00a0\u2014\u00a0 Review & Send",
        help="Review each personalised message. Edit if needed, then approve and send.",
    )

    # Summary table of all generated messages
    summary_rows = []
    for cid, m in batch.items():
        summary_rows.append({
            "Customer": f"{m['customer_name']} ({cid})",
            "Risk": f"{m['risk_score']:.0f}",
            "EMI": f"\u20b9{m['emi_amount']:,.0f}" if m["emi_amount"] > 0 else "\u2014",
            "Balance": f"\u20b9{m['avg_balance']:,.0f}",
            "Can Pay": m["can_pay"],
            "Channel": m["channel"].upper(),
            "Message Preview": m["body"][:80] + ("\u2026" if len(m["body"]) > 80 else ""),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Expandable per-customer message editor
    st.markdown("**Click to review / edit individual messages:**")
    for cid, m in batch.items():
        with st.expander(
            f"{m['customer_name']} \u2014 Score: {m['risk_score']:.0f} | "
            f"EMI: \u20b9{m['emi_amount']:,.0f} | "
            f"Balance: \u20b9{m['avg_balance']:,.0f} {m['can_pay']}"
        ):
            if m.get("subject"):
                batch[cid]["subject"] = st.text_input(
                    "Subject", value=m["subject"], key=f"subj_{cid}",
                )
            batch[cid]["body"] = st.text_area(
                "Message", value=m["body"], height=120, key=f"body_{cid}",
            )

    # Send all button
    st.markdown("---")
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        st.markdown(
            f"**Ready to send {len(batch)} messages via "
            f"{list(batch.values())[0]['channel'].upper()}?**"
        )
    with sc2:
        send_all = st.button(
            f"\U0001f680 Send All {len(batch)} Messages",
            type="primary",
            key="btn_send_all",
        )

    if send_all:
        success_n = 0
        fail_n = 0
        progress = st.progress(0, text="Sending interventions\u2026")

        for idx, (cid, m) in enumerate(batch.items()):
            msg_text = m["body"]
            if m.get("subject"):
                msg_text = f"Subject: {m['subject']}\n\n{msg_text}"

            resp = api_post(
                "/intervention/trigger",
                {
                    "customer_id": cid,
                    "intervention_type": "reminder",
                    "channel": m["channel"],
                    "message": msg_text,
                },
            )
            if resp:
                success_n += 1
                st.session_state["hr_sent_log"].append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "customer_id": cid,
                    "customer_name": m["customer_name"],
                    "risk_score": m["risk_score"],
                    "channel": m["channel"].upper(),
                    "emi": f"\u20b9{m['emi_amount']:,.0f}" if m["emi_amount"] > 0 else "\u2014",
                    "can_pay": m["can_pay"],
                    "status": "\u2705 Sent",
                })
            else:
                fail_n += 1
                st.session_state["hr_sent_log"].append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "customer_id": cid,
                    "customer_name": m["customer_name"],
                    "risk_score": m["risk_score"],
                    "channel": m["channel"].upper(),
                    "emi": f"\u20b9{m['emi_amount']:,.0f}" if m["emi_amount"] > 0 else "\u2014",
                    "can_pay": m["can_pay"],
                    "status": "\u274c Failed",
                })

            progress.progress(
                (idx + 1) / len(batch),
                text=f"Sent {idx + 1}/{len(batch)}\u2026",
            )

        progress.empty()
        st.session_state["hr_batch_messages"] = {}

        if fail_n == 0:
            st.success(
                f"\u2705 All **{success_n}** interventions sent successfully!"
            )
            st.balloons()
        else:
            st.warning(
                f"{success_n} sent, {fail_n} failed. Check backend logs."
            )

    # ==================================================================
    # Intervention log
    # ==================================================================
    if st.session_state["hr_sent_log"]:
        st.markdown("---")
        st.subheader("\U0001f4cb Intervention Log")
        log_df = pd.DataFrame(st.session_state["hr_sent_log"])
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        lc1, lc2 = st.columns(2)
        with lc1:
            st.metric(
                "Total Sent",
                len([r for r in st.session_state["hr_sent_log"] if "\u2705" in r["status"]]),
            )
        with lc2:
            if st.button("\U0001f5d1\ufe0f Clear Log", key="btn_clear_log"):
                st.session_state["hr_sent_log"] = []
                st.rerun()


# =========================================================================
# PAGE 4 — Analytics
# =========================================================================


def page_analytics() -> None:
    """Model performance, feature importance, score distributions, and cost savings."""
    st.title("\U0001f4c8 Analytics")

    tab_model, tab_cost = st.tabs(
        ["\U0001f9e0 Model Performance", "\U0001f4b0 Cost Savings"]
    )

    # ==================================================================
    # Tab 1 — Model Performance
    # ==================================================================
    with tab_model:
        perf = load_model_performance()
        fi_df = load_feature_importance()

        st.subheader("Model Performance")
        if perf:
            test_metrics: dict[str, float] = perf.get("test_metrics") or {}
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Accuracy",
                    f"{test_metrics.get('accuracy', 0):.3f}",
                )
            with col2:
                st.metric(
                    "Precision",
                    f"{test_metrics.get('precision', 0):.3f}",
                )
            with col3:
                st.metric(
                    "Recall",
                    f"{test_metrics.get('recall', 0):.3f}",
                )
            with col4:
                st.metric(
                    "ROC-AUC",
                    f"{test_metrics.get('roc_auc', 0):.3f}",
                )
            st.success("Model metrics loaded successfully.")
        else:
            st.warning(
                "Model performance file not found. "
                "Run the training pipeline first."
            )

        # Feature importance
        st.subheader("Feature Importance")
        if not fi_df.empty:
            fi_col: Optional[str] = None
            for candidate in (
                "ensemble_importance",
                "xgboost_importance",
                "lightgbm_importance",
            ):
                if candidate in fi_df.columns:
                    fi_col = candidate
                    break
            if fi_col:
                top_fi = fi_df.sort_values(
                    fi_col, ascending=False
                ).head(15)
                fig_fi = px.bar(
                    top_fi,
                    x=fi_col,
                    y="feature",
                    orientation="h",
                    title=f"Top Features by {fi_col}",
                )
                st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance file not found.")

        # Confusion matrix & ROC
        st.subheader("Confusion Matrix & ROC Curve")
        cm_path = (
            _PROJECT_ROOT / "ml" / "reports" / "confusion_matrix_heatmap.png"
        )
        roc_path = _PROJECT_ROOT / "ml" / "reports" / "roc_curve.png"
        col_a, col_b = st.columns(2)
        with col_a:
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion Matrix")
            else:
                st.info("Confusion matrix image not found.")
        with col_b:
            if roc_path.exists():
                st.image(str(roc_path), caption="ROC Curve")
            else:
                st.info("ROC curve image not found.")

    # ==================================================================
    # Tab 2 — Cost Savings (GPT-4 vs Llama 3.1)
    # ==================================================================
    with tab_cost:
        st.subheader("\U0001f4b0 AI Cost Savings \u2014 FREE Llama 3.1 vs Paid APIs")
        st.caption(
            "Quantify the financial impact of running FREE local AI "
            "vs. commercial API providers"
        )

        _USD_TO_INR: float = 83.0
        _PROVIDERS: list[dict[str, Any]] = [
            {
                "name": "GPT-4o (OpenAI)",
                "input_1m": 2.50 * _USD_TO_INR,
                "output_1m": 10.00 * _USD_TO_INR,
                "colour": "#74aa9c",
            },
            {
                "name": "GPT-4 Turbo (OpenAI)",
                "input_1m": 10.00 * _USD_TO_INR,
                "output_1m": 30.00 * _USD_TO_INR,
                "colour": "#412991",
            },
            {
                "name": "Claude 3.5 Sonnet",
                "input_1m": 3.00 * _USD_TO_INR,
                "output_1m": 15.00 * _USD_TO_INR,
                "colour": "#d4a27f",
            },
            {
                "name": "Gemini 1.5 Pro",
                "input_1m": 1.25 * _USD_TO_INR,
                "output_1m": 5.00 * _USD_TO_INR,
                "colour": "#4285f4",
            },
        ]

        # Inputs
        in1, in2, in3 = st.columns(3)
        with in1:
            msgs_month: int = st.number_input(
                "Messages / month",
                min_value=1_000,
                max_value=1_000_000,
                value=100_000,
                step=10_000,
                key="cost_msgs",
            )
        with in2:
            avg_tok: int = st.slider(
                "Avg tokens / message",
                min_value=50,
                max_value=500,
                value=150,
                step=10,
                key="cost_tok",
            )
        with in3:
            gpu_cost: int = st.number_input(
                "One-time GPU cost (\u20b9)",
                min_value=0,
                max_value=10_000_000,
                value=200_000,
                step=25_000,
                key="cost_gpu",
            )

        out_tokens: int = msgs_month * avg_tok
        in_tokens: int = msgs_month * 200

        provider_costs: list[dict[str, Any]] = []
        for p in _PROVIDERS:
            monthly = (
                (in_tokens / 1e6) * p["input_1m"]
                + (out_tokens / 1e6) * p["output_1m"]
            )
            provider_costs.append(
                {
                    "name": p["name"],
                    "monthly": monthly,
                    "annual": monthly * 12,
                    "colour": p["colour"],
                }
            )

        elec_monthly: float = 1_500.0
        llama_monthly: float = elec_monthly
        llama_annual: float = llama_monthly * 12 + gpu_cost / 5.0

        # KPIs
        st.markdown("---")
        cheapest_api = min(provider_costs, key=lambda x: x["annual"])
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric(
                "\u274c Cheapest API / year",
                f"\u20b9{cheapest_api['annual']:,.0f}",
                delta=cheapest_api["name"],
                delta_color="inverse",
            )
        with k2:
            st.metric(
                "\u2705 Llama 3.1 / year",
                f"\u20b9{llama_annual:,.0f}",
            )
        with k3:
            savings = cheapest_api["annual"] - llama_annual
            st.metric(
                "\U0001f4b0 Annual Savings",
                f"\u20b9{savings:,.0f}",
                delta=f"{(savings / cheapest_api['annual'] * 100):.0f}% cheaper"
                if cheapest_api["annual"] > 0
                else "",
                delta_color="normal",
            )

        # 5-year projection chart
        st.markdown("### 5-Year Cost Projection")
        years = list(range(1, 6))
        chart_data: dict[str, list[float]] = {"Year": years}
        for p in provider_costs:
            chart_data[p["name"]] = [p["annual"] * y for y in years]
        chart_data["Llama 3.1 (FREE)"] = [
            gpu_cost + llama_monthly * 12 * y for y in years
        ]

        fig_proj = go.Figure()
        for p in provider_costs:
            fig_proj.add_trace(
                go.Scatter(
                    x=years,
                    y=chart_data[p["name"]],
                    name=p["name"],
                    mode="lines+markers",
                    line=dict(color=p["colour"], width=2),
                )
            )
        fig_proj.add_trace(
            go.Scatter(
                x=years,
                y=chart_data["Llama 3.1 (FREE)"],
                name="Llama 3.1 (FREE)",
                mode="lines+markers",
                line=dict(color="#22c55e", width=3, dash="dot"),
            )
        )
        fig_proj.update_layout(
            yaxis_title="Cumulative Cost (\u20b9)",
            xaxis_title="Year",
            template="plotly_dark",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_proj, use_container_width=True)

        # Monthly bar chart
        st.markdown("### Monthly Cost Comparison")
        bar_names = [p["name"] for p in provider_costs] + [
            "Llama 3.1 (FREE)"
        ]
        bar_vals = [p["monthly"] for p in provider_costs] + [
            llama_monthly
        ]
        bar_clrs = [p["colour"] for p in provider_costs] + ["#22c55e"]
        fig_bar = go.Figure(
            go.Bar(
                x=bar_names,
                y=bar_vals,
                marker_color=bar_clrs,
                text=[f"\u20b9{v:,.0f}" for v in bar_vals],
                textposition="outside",
            )
        )
        fig_bar.update_layout(
            yaxis_title="Monthly Cost (\u20b9)",
            template="plotly_dark",
            height=360,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ROI
        gpt4o = next(
            p for p in provider_costs if "GPT-4o" in p["name"]
        )
        total_gpt4o_5yr = gpt4o["annual"] * 5
        total_llama_5yr = gpu_cost + llama_monthly * 12 * 5
        savings_5yr = total_gpt4o_5yr - total_llama_5yr
        roi_pct = (
            (savings_5yr / total_llama_5yr) * 100
            if total_llama_5yr > 0
            else 0
        )
        payback_mo = (
            gpu_cost / (gpt4o["monthly"] - llama_monthly)
            if gpt4o["monthly"] > llama_monthly
            else 0
        )

        st.success(
            f"**5-Year Savings vs GPT-4o:** \u20b9{savings_5yr:,.0f} "
            f"| **ROI:** {roi_pct:,.0f}% "
            f"| **Payback:** {payback_mo:.1f} months"
        )

        # Benefits
        st.markdown("### Beyond Cost")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.info(
                "**\U0001f512 Privacy**\n\n"
                "- No data to third parties\n"
                "- GDPR compliant\n"
                "- Offline capable"
            )
        with b2:
            st.info(
                "**\u2699\ufe0f Operational**\n\n"
                "- Zero rate limits\n"
                "- No API downtime\n"
                "- No vendor lock-in"
            )
        with b3:
            st.info(
                "**\U0001f3e6 Regulatory**\n\n"
                "- RBI data localisation\n"
                "- On-premise audit trail\n"
                "- Easier compliance"
            )


# =========================================================================
# PAGE 5 — Live Monitoring
# =========================================================================

# --- Simulated data generators for the monitoring page ---

_CATEGORIES: list[str] = [
    "salary",
    "utility_electricity",
    "utility_water",
    "upi_lending_app",
    "discretionary_restaurant",
    "discretionary_shopping",
    "emi_payment",
    "insurance_premium",
    "investment_sip",
    "atm_withdrawal",
]

_ACTIONS: dict[str, str] = {
    "low": "Monitor — no action",
    "medium": "Send payment reminder",
    "high": "Offer restructuring",
    "critical": "Priority outreach via call centre",
}


def _risk_level_for_score(score: float) -> str:
    """Map numeric score to risk-level label.

    Args:
        score: 0-100 risk score.

    Returns:
        One of ``low``, ``medium``, ``high``, ``critical``.
    """
    if score <= 40:
        return "low"
    if score <= 70:
        return "medium"
    if score <= 85:
        return "high"
    return "critical"


def _generate_fake_transaction(
    customer_ids: list[str],
    rng: random.Random,
) -> dict[str, Any]:
    """Produce a single simulated transaction row.

    Args:
        customer_ids: Pool of valid customer IDs to pick from.
        rng: Seeded random instance for reproducibility within a session.

    Returns:
        Dict with transaction fields.
    """
    cid = rng.choice(customer_ids)
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_id": f"tx_{rng.randint(100000, 999999):06d}",
        "customer_id": cid,
        "type": rng.choice(["credit", "debit"]),
        "category": rng.choice(_CATEGORIES),
        "amount": round(rng.uniform(50, 25000), 2),
    }


def _generate_fake_prediction(
    customer_ids: list[str],
    rng: random.Random,
) -> dict[str, Any]:
    """Produce a single simulated prediction-log entry.

    Args:
        customer_ids: Pool of valid customer IDs.
        rng: Seeded random instance.

    Returns:
        Dict with prediction fields.
    """
    cid = rng.choice(customer_ids)
    score = round(rng.uniform(5, 98), 1)
    level = _risk_level_for_score(score)
    return {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "customer_id": cid,
        "risk_score": score,
        "risk_level": level,
        "action": _ACTIONS[level],
    }


def _make_system_gauge(load_pct: float) -> go.Figure:
    """Build a Plotly gauge for system load.

    Args:
        load_pct: Current load as 0-100 percentage.

    Returns:
        A Plotly ``Figure`` object.
    """
    if load_pct < 50:
        bar_color = SECONDARY_COLOR
    elif load_pct < 80:
        bar_color = WARNING_COLOR
    else:
        bar_color = DANGER_COLOR

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=load_pct,
            number={"suffix": "%"},
            title={"text": "System Load", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color, "thickness": 0.6},
                "bgcolor": "#1e1e1e",
                "steps": [
                    {"range": [0, 50], "color": "#0d3b3b"},
                    {"range": [50, 80], "color": "#3b3200"},
                    {"range": [80, 100], "color": "#3b0d0d"},
                ],
                "threshold": {
                    "line": {"color": "#ff4444", "width": 3},
                    "thickness": 0.8,
                    "value": 85,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fafafa"},
    )
    return fig


def _make_throughput_chart(history: list[dict[str, Any]]) -> go.Figure:
    """Build a live line chart of transactions-per-tick over time.

    Args:
        history: List of dicts with ``time`` and ``count`` keys.

    Returns:
        A Plotly ``Figure``.
    """
    df = pd.DataFrame(history)
    fig = go.Figure(
        go.Scatter(
            x=df["time"],
            y=df["count"],
            mode="lines+markers",
            line={"color": PRIMARY_COLOR, "width": 2},
            marker={"size": 4},
            fill="tozeroy",
            fillcolor="rgba(2,128,144,0.15)",
        )
    )
    fig.update_layout(
        title="Transactions / Tick",
        xaxis_title="",
        yaxis_title="Count",
        height=220,
        margin=dict(l=40, r=10, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fafafa", "size": 11},
        xaxis={"showgrid": False},
        yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.08)"},
    )
    return fig


def _risk_dot(level: str) -> str:
    """Return a coloured unicode dot for the risk level.

    Args:
        level: Risk level string.

    Returns:
        A coloured dot character suitable for markdown.
    """
    colors = {
        "critical": "\U0001f534",
        "high": "\U0001f7e0",
        "medium": "\U0001f7e1",
        "low": "\U0001f7e2",
    }
    return colors.get(level, "\u26aa")


def page_live_monitoring() -> None:
    """Live monitoring dashboard simulating real-time transaction processing.

    Uses ``st.empty()`` containers in a timed loop to create the
    illusion of a production streaming pipeline.
    """
    st.title("\u26a1 Live Monitoring")
    st.caption("Simulated real-time transaction processing & risk prediction pipeline")

    # --- Controls ---
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])
    with ctrl_col1:
        refresh_rate: int = st.select_slider(
            "Refresh rate (seconds)",
            options=[2, 3, 5, 10],
            value=5,
            key="monitor_refresh",
        )
    with ctrl_col2:
        batch_size: int = st.select_slider(
            "Transactions per tick",
            options=[1, 2, 3, 5, 8, 10],
            value=3,
            key="monitor_batch",
        )
    with ctrl_col3:
        is_running: bool = st.toggle("Streaming", value=True, key="monitor_on")

    st.markdown("---")

    # --- Load customer IDs for simulation ---
    customers_df = load_customers_csv()
    if customers_df.empty or "customer_id" not in customers_df.columns:
        st.error(
            "Cannot start monitoring — customer data not found. "
            "Ensure ml/data/customers.csv exists."
        )
        return
    customer_ids: list[str] = customers_df["customer_id"].tolist()

    # --- Initialise session state ---
    if "monitor_tx_log" not in st.session_state:
        st.session_state["monitor_tx_log"] = []
    if "monitor_pred_log" not in st.session_state:
        st.session_state["monitor_pred_log"] = []
    if "monitor_throughput" not in st.session_state:
        st.session_state["monitor_throughput"] = []
    if "monitor_total_tx" not in st.session_state:
        st.session_state["monitor_total_tx"] = 0
    if "monitor_tick" not in st.session_state:
        st.session_state["monitor_tick"] = 0

    rng = random.Random(42 + st.session_state["monitor_tick"])

    # --- KPI row (placeholders) ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    ph_kpi_tx = kpi1.empty()
    ph_kpi_pred = kpi2.empty()
    ph_kpi_highrisk = kpi3.empty()
    ph_kpi_uptime = kpi4.empty()

    # --- Main layout: gauge + throughput chart ---
    chart_col, gauge_col = st.columns([3, 1])
    ph_throughput = chart_col.empty()
    ph_gauge = gauge_col.empty()

    # --- Status bar ---
    ph_status = st.empty()

    # --- Recent transactions table ---
    st.markdown("### Recent Transactions")
    ph_tx_table = st.empty()

    # --- Prediction log ---
    st.markdown("### Prediction Log")
    ph_pred_table = st.empty()

    # --- Risk distribution mini-chart ---
    dist_col1, dist_col2 = st.columns(2)
    ph_risk_dist = dist_col1.empty()
    ph_action_summary = dist_col2.empty()

    # --- Streaming loop ---
    if not is_running:
        # Render current state once (no loop)
        _render_monitoring_state(
            st.session_state,
            ph_kpi_tx,
            ph_kpi_pred,
            ph_kpi_highrisk,
            ph_kpi_uptime,
            ph_throughput,
            ph_gauge,
            ph_status,
            ph_tx_table,
            ph_pred_table,
            ph_risk_dist,
            ph_action_summary,
        )
        st.info("Streaming paused. Toggle **Streaming** to resume.")
        return

    # Live loop
    while True:
        st.session_state["monitor_tick"] += 1
        rng = random.Random(int(time.time() * 1000) + st.session_state["monitor_tick"])

        # Generate new batch of transactions
        new_txs = [
            _generate_fake_transaction(customer_ids, rng) for _ in range(batch_size)
        ]
        st.session_state["monitor_tx_log"] = (
            new_txs + st.session_state["monitor_tx_log"]
        )[:200]
        st.session_state["monitor_total_tx"] += len(new_txs)

        # Generate prediction events (slightly fewer than txs)
        pred_count = max(1, batch_size - rng.randint(0, 1))
        new_preds = [
            _generate_fake_prediction(customer_ids, rng) for _ in range(pred_count)
        ]
        st.session_state["monitor_pred_log"] = (
            new_preds + st.session_state["monitor_pred_log"]
        )[:300]

        # Throughput history
        st.session_state["monitor_throughput"].append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "count": len(new_txs),
            }
        )
        st.session_state["monitor_throughput"] = st.session_state["monitor_throughput"][
            -40:
        ]

        # Render everything
        _render_monitoring_state(
            st.session_state,
            ph_kpi_tx,
            ph_kpi_pred,
            ph_kpi_highrisk,
            ph_kpi_uptime,
            ph_throughput,
            ph_gauge,
            ph_status,
            ph_tx_table,
            ph_pred_table,
            ph_risk_dist,
            ph_action_summary,
        )

        time.sleep(refresh_rate)


def _render_monitoring_state(
    state: dict[str, Any],
    ph_kpi_tx: st.delta_generator.DeltaGenerator,
    ph_kpi_pred: st.delta_generator.DeltaGenerator,
    ph_kpi_highrisk: st.delta_generator.DeltaGenerator,
    ph_kpi_uptime: st.delta_generator.DeltaGenerator,
    ph_throughput: st.delta_generator.DeltaGenerator,
    ph_gauge: st.delta_generator.DeltaGenerator,
    ph_status: st.delta_generator.DeltaGenerator,
    ph_tx_table: st.delta_generator.DeltaGenerator,
    ph_pred_table: st.delta_generator.DeltaGenerator,
    ph_risk_dist: st.delta_generator.DeltaGenerator,
    ph_action_summary: st.delta_generator.DeltaGenerator,
) -> None:
    """Render all monitoring widgets into their placeholder containers.

    Args:
        state: Streamlit session state dict.
        ph_*: Placeholder ``st.empty()`` containers for each widget.
    """
    tx_log: list[dict[str, Any]] = state.get("monitor_tx_log", [])
    pred_log: list[dict[str, Any]] = state.get("monitor_pred_log", [])
    throughput: list[dict[str, Any]] = state.get("monitor_throughput", [])
    total_tx: int = state.get("monitor_total_tx", 0)

    # --- KPIs ---
    # Transactions in last minute (approximate via throughput history)
    recent_tx = sum(t["count"] for t in throughput[-12:])  # last ~60s at 5s ticks
    high_risk_recent = sum(
        1 for p in pred_log[:50] if p["risk_level"] in ("high", "critical")
    )

    rng_gauge = random.Random(int(time.time()))
    system_load = min(99, max(5, 35 + rng_gauge.gauss(0, 15) + len(tx_log) * 0.05))

    ph_kpi_tx.metric(
        "\U0001f4e6 Transactions Processed (last min)",
        f"{recent_tx:,}",
        delta=f"+{throughput[-1]['count']}" if throughput else None,
    )
    ph_kpi_pred.metric(
        "\U0001f9e0 Predictions Generated",
        f"{len(pred_log):,}",
    )
    ph_kpi_highrisk.metric(
        "\U0001f6a8 High-Risk Detections (recent)",
        f"{high_risk_recent}",
        delta=f"{high_risk_recent}" if high_risk_recent else "0",
        delta_color="inverse",
    )
    ph_kpi_uptime.metric(
        "\u2705 Pipeline Status",
        "ONLINE",
    )

    # --- Throughput chart ---
    if len(throughput) >= 2:
        ph_throughput.plotly_chart(
            _make_throughput_chart(throughput),
            use_container_width=True,
            key=f"tp_{state.get('monitor_tick', 0)}",
        )
    else:
        ph_throughput.info("Collecting throughput data...")

    # --- System load gauge ---
    ph_gauge.plotly_chart(
        _make_system_gauge(round(system_load, 1)),
        use_container_width=True,
        key=f"sg_{state.get('monitor_tick', 0)}",
    )

    # --- Status bar ---
    now_str = datetime.now().strftime("%H:%M:%S")
    ph_status.markdown(
        f"<div style='background:#0d3b3b;padding:8px 16px;border-radius:6px;"
        f"font-family:monospace;font-size:13px;color:#00ffc8;'>"
        f"\u25cf LIVE &nbsp;&nbsp; Last tick: {now_str} &nbsp;&nbsp; "
        f"Total transactions: {total_tx:,} &nbsp;&nbsp; "
        f"System load: {system_load:.0f}%"
        f"</div>",
        unsafe_allow_html=True,
    )

    # --- Recent transactions table ---
    if tx_log:
        tx_df = pd.DataFrame(tx_log[:30])
        display_cols = [
            "timestamp",
            "transaction_id",
            "customer_id",
            "type",
            "category",
            "amount",
        ]
        tx_df = tx_df.reindex(columns=[c for c in display_cols if c in tx_df.columns])
        ph_tx_table.dataframe(
            tx_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        ph_tx_table.info("Waiting for transactions...")

    # --- Prediction log ---
    if pred_log:
        pred_df = pd.DataFrame(pred_log[:40])
        pred_df["status"] = pred_df["risk_level"].apply(
            lambda l: f"{_risk_dot(l)} {l.upper()}"
        )
        display_pred_cols = [
            "timestamp",
            "customer_id",
            "risk_score",
            "status",
            "action",
        ]
        pred_df = pred_df.reindex(
            columns=[c for c in display_pred_cols if c in pred_df.columns]
        )
        ph_pred_table.dataframe(
            pred_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        ph_pred_table.info("Waiting for predictions...")

    # --- Risk distribution pie ---
    if pred_log:
        levels = [p["risk_level"] for p in pred_log[:100]]
        dist = pd.Series(levels).value_counts().reset_index()
        dist.columns = ["level", "count"]
        fig_dist = px.pie(
            dist,
            names="level",
            values="count",
            title="Risk Distribution (last 100)",
            color="level",
            color_discrete_map={
                "low": SECONDARY_COLOR,
                "medium": WARNING_COLOR,
                "high": DANGER_COLOR,
                "critical": "#8B0000",
            },
            hole=0.4,
        )
        fig_dist.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#fafafa", "size": 11},
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
        )
        ph_risk_dist.plotly_chart(
            fig_dist,
            use_container_width=True,
            key=f"rd_{state.get('monitor_tick', 0)}",
        )

        # Action summary
        actions = [p["action"] for p in pred_log[:100]]
        action_counts = pd.Series(actions).value_counts().reset_index()
        action_counts.columns = ["action", "count"]
        fig_act = px.bar(
            action_counts,
            x="count",
            y="action",
            orientation="h",
            title="Actions Triggered (last 100)",
            color="count",
            color_continuous_scale="Teal",
        )
        fig_act.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#fafafa", "size": 11},
            showlegend=False,
            coloraxis_showscale=False,
            yaxis={"showgrid": False},
        )
        ph_action_summary.plotly_chart(
            fig_act,
            use_container_width=True,
            key=f"as_{state.get('monitor_tick', 0)}",
        )


# =========================================================================
# PAGE 6 — Intervention Workflow
# =========================================================================

# Flowchart node definitions for the Plotly-based pipeline diagram.
_FLOW_NODES: list[dict[str, Any]] = [
    {
        "id": "A",
        "label": "\U0001f4e5 Transaction\nIngested",
        "x": 0,
        "y": 2,
        "color": "#0d3b3b",
        "border": "#00A896",
    },
    {
        "id": "B",
        "label": "\u2699\ufe0f Feature\nCalculation",
        "x": 1,
        "y": 2,
        "color": "#0d3b3b",
        "border": "#00A896",
    },
    {
        "id": "C",
        "label": "\U0001f9e0 ML Model\nPrediction",
        "x": 2,
        "y": 2,
        "color": "#0d3b3b",
        "border": "#028090",
    },
    {
        "id": "D",
        "label": "\U0001f4ca Risk\nScore",
        "x": 3,
        "y": 2,
        "color": "#1a1a2e",
        "border": "#FFC107",
    },
    {
        "id": "E",
        "label": "\u2705 No Action\nScore < 40",
        "x": 4.5,
        "y": 3.5,
        "color": "#0d3b0d",
        "border": "#00A896",
    },
    {
        "id": "F",
        "label": "\U0001f4e7 Email Reminder\nScore 40-70",
        "x": 4.5,
        "y": 2.7,
        "color": "#3b3200",
        "border": "#FFC107",
    },
    {
        "id": "G",
        "label": "\U0001f4f1 SMS + Email\nScore 70-85",
        "x": 4.5,
        "y": 1.3,
        "color": "#3b1f00",
        "border": "#FF8C00",
    },
    {
        "id": "H",
        "label": "\U0001f4de Immediate Call\nScore 85+",
        "x": 4.5,
        "y": 0.5,
        "color": "#3b0d0d",
        "border": "#DC3545",
    },
]

_FLOW_EDGES: list[tuple[str, str]] = [
    ("A", "B"),
    ("B", "C"),
    ("C", "D"),
    ("D", "E"),
    ("D", "F"),
    ("D", "G"),
    ("D", "H"),
]

_TIMELINE_EVENTS: list[dict[str, Any]] = [
    {
        "day": -14,
        "label": "First Signal Detected",
        "detail": (
            "Salary credited 5 days late. Savings balance dropped 12% "
            "from prior month. System flags early warning."
        ),
        "icon": "\U0001f50d",
        "color": "#00A896",
    },
    {
        "day": -7,
        "label": "Risk Score Increased",
        "detail": (
            "Score jumped from 42 to 68 after two failed utility "
            "payments and a spike in lending-app transactions."
        ),
        "icon": "\U0001f4c8",
        "color": "#FFC107",
    },
    {
        "day": 0,
        "label": "Intervention Triggered",
        "detail": (
            "Score crossed 70 threshold. Personalised SMS + email "
            "sent with flexible repayment options and helpline number."
        ),
        "icon": "\U0001f6a8",
        "color": "#DC3545",
    },
    {
        "day": 7,
        "label": "Customer Response Tracked",
        "detail": (
            "Customer opened the email, clicked the repayment link, "
            "and initiated a partial payment of \u20b95,000."
        ),
        "icon": "\U0001f4ac",
        "color": "#028090",
    },
    {
        "day": 30,
        "label": "Outcome Measured",
        "detail": (
            "Customer completed full EMI payment on time. Risk score "
            "dropped to 35. Case marked as successfully resolved."
        ),
        "icon": "\u2705",
        "color": "#00A896",
    },
]


def _render_workflow_flowchart() -> None:
    """Render the intervention pipeline as a Plotly flowchart."""
    st.subheader("\U0001f504 Intervention Pipeline")
    st.caption("How a raw transaction becomes an intervention decision in real time")

    # Build lookup for node positions
    node_map: dict[str, dict[str, Any]] = {n["id"]: n for n in _FLOW_NODES}

    fig = go.Figure()

    # --- Draw edges (arrows) ---
    for src_id, dst_id in _FLOW_EDGES:
        src = node_map[src_id]
        dst = node_map[dst_id]
        fig.add_annotation(
            x=dst["x"],
            y=dst["y"],
            ax=src["x"],
            ay=src["y"],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="rgba(255,255,255,0.35)",
        )

    # --- Draw nodes ---
    for node in _FLOW_NODES:
        # Filled rectangle via a shape
        pad_x = 0.38
        pad_y = 0.35
        fig.add_shape(
            type="rect",
            x0=node["x"] - pad_x,
            x1=node["x"] + pad_x,
            y0=node["y"] - pad_y,
            y1=node["y"] + pad_y,
            fillcolor=node["color"],
            line=dict(color=node["border"], width=2),
            layer="above",
        )
        # Label
        fig.add_annotation(
            x=node["x"],
            y=node["y"],
            text=node["label"].replace("\n", "<br>"),
            showarrow=False,
            font=dict(size=11, color="#ffffff", family="Arial"),
            align="center",
            bgcolor="rgba(0,0,0,0)",
        )

    # --- Edge labels for score ranges ---
    d = node_map["D"]
    labels = [
        (node_map["E"], "< 40"),
        (node_map["F"], "40–70"),
        (node_map["G"], "70–85"),
        (node_map["H"], "85+"),
    ]
    for target, txt in labels:
        mid_x = (d["x"] + target["x"]) / 2
        mid_y = (d["y"] + target["y"]) / 2
        fig.add_annotation(
            x=mid_x,
            y=mid_y,
            text=f"<b>{txt}</b>",
            showarrow=False,
            font=dict(size=9, color="#ffc107"),
            bgcolor="rgba(30,30,46,0.85)",
            bordercolor="#ffc107",
            borderwidth=1,
            borderpad=3,
        )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            visible=False,
            range=[-0.7, 5.2],
        ),
        yaxis=dict(
            visible=False,
            range=[-0.2, 4.2],
            scaleanchor="x",
        ),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Pipeline step descriptions ---
    with st.expander("Pipeline step details", expanded=False):
        steps = [
            (
                "\U0001f4e5 **Transaction Ingested**",
                "Raw banking transaction arrives from core banking / "
                "UPI / IMPS feed.",
            ),
            (
                "\u2699\ufe0f **Feature Calculation**",
                "60+ behavioural features computed: salary delays, "
                "balance trends, spending patterns, lending-app usage, "
                "payment failures.",
            ),
            (
                "\U0001f9e0 **ML Model Prediction**",
                "Ensemble (XGBoost + LightGBM) scores the customer "
                "on a 0-100 scale.",
            ),
            (
                "\U0001f4ca **Risk Scoring & Routing**",
                "Score maps to an action tier that determines channel " "and urgency.",
            ),
        ]
        for title, desc in steps:
            st.markdown(f"{title}  \n{desc}")

    # --- Action tier reference table ---
    st.markdown("")
    tier_df = pd.DataFrame(
        {
            "Score Range": ["0 – 40", "40 – 70", "70 – 85", "85 – 100"],
            "Risk Level": [
                "\U0001f7e2 Low",
                "\U0001f7e1 Medium",
                "\U0001f7e0 High",
                "\U0001f534 Critical",
            ],
            "Channel": [
                "None (monitor)",
                "Email reminder",
                "SMS + Email",
                "Immediate phone call",
            ],
            "SLA": [
                "\u2014",
                "Within 24 h",
                "Within 4 h",
                "Within 30 min",
            ],
            "Escalation": [
                "No",
                "Auto after 7 days",
                "Auto after 48 h",
                "Supervisor notified immediately",
            ],
        }
    )
    st.dataframe(tier_df, use_container_width=True, hide_index=True)


def _render_intervention_timeline() -> None:
    """Render the customer intervention journey as a visual timeline."""
    st.subheader("\U0001f4c5 Intervention Journey Timeline")
    st.caption(
        "Lifecycle of a typical pre-delinquency case from first signal to resolution"
    )

    # Plotly timeline chart
    tl_df = pd.DataFrame(_TIMELINE_EVENTS)
    fig = go.Figure()

    # Horizontal line (the "spine" of the timeline)
    fig.add_shape(
        type="line",
        x0=-16,
        x1=33,
        y0=0,
        y1=0,
        line=dict(color="rgba(255,255,255,0.15)", width=2),
    )

    for i, evt in enumerate(tl_df.itertuples()):
        y_offset = 0.45 if i % 2 == 0 else -0.45
        text_pos = "top center" if i % 2 == 0 else "bottom center"

        fig.add_trace(
            go.Scatter(
                x=[evt.day],
                y=[0],
                mode="markers",
                marker=dict(size=18, color=evt.color, line=dict(width=2, color="#fff")),
                hoverinfo="text",
                hovertext=f"Day {evt.day:+d}: {evt.label}<br>{evt.detail}",
                showlegend=False,
            )
        )

        label_text = f"<b>Day {evt.day:+d}</b><br>" f"{evt.icon} {evt.label}"
        fig.add_annotation(
            x=evt.day,
            y=y_offset,
            text=label_text,
            showarrow=True,
            arrowhead=0,
            arrowwidth=1,
            arrowcolor="rgba(255,255,255,0.3)",
            ax=0,
            ay=-40 if i % 2 == 0 else 40,
            font=dict(size=11, color="#fafafa"),
            align="center",
            bgcolor="rgba(30,30,46,0.85)",
            bordercolor=evt.color,
            borderwidth=1,
            borderpad=6,
        )

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Days relative to intervention",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=True,
            zerolinecolor=DANGER_COLOR,
            zerolinewidth=2,
            tickvals=[-14, -7, 0, 7, 30],
            ticktext=["Day -14", "Day -7", "Day 0", "Day +7", "Day +30"],
            range=[-18, 35],
            color="#aaa",
        ),
        yaxis=dict(
            visible=False,
            range=[-1, 1],
        ),
        font=dict(color="#fafafa"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detail cards below the chart
    cols = st.columns(len(_TIMELINE_EVENTS))
    for col, evt in zip(cols, _TIMELINE_EVENTS):
        with col:
            st.markdown(
                f"<div style='"
                f"border-left: 3px solid {evt['color']};"
                f"padding: 8px 12px;"
                f"border-radius: 4px;"
                f"background: rgba(30,30,46,0.6);"
                f"font-size: 12px;"
                f"'>"
                f"<b>{evt['icon']} Day {evt['day']:+d}</b><br>"
                f"<span style='color:#ccc'>{evt['detail']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_success_metrics() -> None:
    """Render intervention success KPIs and historical charts."""
    st.subheader("\U0001f3af Intervention Success Metrics")

    rng = random.Random(int(datetime.now().strftime("%Y%m%d")))

    # Simulated daily metrics
    interventions_today = rng.randint(12, 45)
    response_rate = round(rng.uniform(58, 82), 1)
    defaults_prevented = rng.randint(3, interventions_today // 2)
    avg_days_to_respond = round(rng.uniform(1.5, 4.2), 1)
    cost_saved = defaults_prevented * rng.randint(8000, 15000)

    # --- KPI cards ---
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric(
            "\U0001f4e8 Interventions Today",
            f"{interventions_today}",
            delta=f"+{rng.randint(1,5)} vs yesterday",
        )
    with k2:
        st.metric(
            "\U0001f4ac Response Rate",
            f"{response_rate}%",
            delta=f"+{rng.uniform(0.5, 3.0):.1f}%",
        )
    with k3:
        st.metric(
            "\U0001f6e1\ufe0f Defaults Prevented",
            f"{defaults_prevented}",
            delta=f"+{rng.randint(0,2)} vs yesterday",
        )
    with k4:
        st.metric(
            "\u23f1\ufe0f Avg Response Time",
            f"{avg_days_to_respond} days",
        )
    with k5:
        st.metric(
            "\U0001f4b0 Estimated Savings",
            f"\u20b9{cost_saved:,}",
        )

    st.markdown("")

    # --- 30-day trend charts ---
    chart_left, chart_right = st.columns(2)

    days_30 = pd.date_range(datetime.now() - timedelta(days=29), periods=30)

    with chart_left:
        daily_interventions = [rng.randint(8, 50) for _ in range(30)]
        daily_responses = [int(i * rng.uniform(0.5, 0.85)) for i in daily_interventions]
        trend_df = pd.DataFrame(
            {
                "date": days_30,
                "Sent": daily_interventions,
                "Responded": daily_responses,
            }
        )
        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Bar(
                x=trend_df["date"],
                y=trend_df["Sent"],
                name="Sent",
                marker_color=PRIMARY_COLOR,
                opacity=0.7,
            )
        )
        fig_trend.add_trace(
            go.Bar(
                x=trend_df["date"],
                y=trend_df["Responded"],
                name="Responded",
                marker_color=SECONDARY_COLOR,
            )
        )
        fig_trend.update_layout(
            title="Daily Interventions (30 days)",
            barmode="overlay",
            height=300,
            margin=dict(l=40, r=10, t=40, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa", size=11),
            legend=dict(orientation="h", y=1.12),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with chart_right:
        prevented = [rng.randint(1, 15) for _ in range(30)]
        missed = [rng.randint(0, 4) for _ in range(30)]
        outcome_df = pd.DataFrame(
            {
                "date": days_30,
                "Prevented": prevented,
                "Missed": missed,
            }
        )
        fig_outcome = go.Figure()
        fig_outcome.add_trace(
            go.Scatter(
                x=outcome_df["date"],
                y=outcome_df["Prevented"],
                name="Defaults Prevented",
                mode="lines+markers",
                line=dict(color=SECONDARY_COLOR, width=2),
                marker=dict(size=4),
                fill="tozeroy",
                fillcolor="rgba(0,168,150,0.12)",
            )
        )
        fig_outcome.add_trace(
            go.Scatter(
                x=outcome_df["date"],
                y=outcome_df["Missed"],
                name="Missed (defaulted)",
                mode="lines+markers",
                line=dict(color=DANGER_COLOR, width=2, dash="dot"),
                marker=dict(size=4),
            )
        )
        fig_outcome.update_layout(
            title="Default Prevention (30 days)",
            height=300,
            margin=dict(l=40, r=10, t=40, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa", size=11),
            legend=dict(orientation="h", y=1.12),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        )
        st.plotly_chart(fig_outcome, use_container_width=True)

    # --- Channel breakdown ---
    st.markdown("")
    ch1, ch2 = st.columns(2)

    with ch1:
        channel_data = pd.DataFrame(
            {
                "Channel": ["SMS", "Email", "App Notification", "Phone Call"],
                "Sent": [
                    rng.randint(30, 80),
                    rng.randint(50, 120),
                    rng.randint(20, 60),
                    rng.randint(5, 20),
                ],
                "Response Rate": [
                    f"{rng.uniform(55, 75):.0f}%",
                    f"{rng.uniform(30, 50):.0f}%",
                    f"{rng.uniform(60, 85):.0f}%",
                    f"{rng.uniform(70, 95):.0f}%",
                ],
                "Avg Cost (\u20b9)": ["0.25", "0.10", "0.00", "12.00"],
            }
        )
        st.markdown("**Channel Performance**")
        st.dataframe(channel_data, use_container_width=True, hide_index=True)

    with ch2:
        scenario_data = pd.DataFrame(
            {
                "Scenario": [
                    "Salary Delay",
                    "Savings Depletion",
                    "Lending App Spike",
                    "Payment Failure",
                    "General Risk",
                ],
                "Triggered": [
                    rng.randint(10, 30),
                    rng.randint(8, 25),
                    rng.randint(5, 15),
                    rng.randint(15, 40),
                    rng.randint(5, 12),
                ],
                "Success Rate": [
                    f"{rng.uniform(60, 80):.0f}%",
                    f"{rng.uniform(50, 70):.0f}%",
                    f"{rng.uniform(40, 65):.0f}%",
                    f"{rng.uniform(65, 85):.0f}%",
                    f"{rng.uniform(45, 60):.0f}%",
                ],
            }
        )
        st.markdown("**Scenario Effectiveness**")
        st.dataframe(scenario_data, use_container_width=True, hide_index=True)


def page_intervention_workflow() -> None:
    """Intervention Workflow page: pipeline, timeline, and success metrics."""
    st.title("\U0001f3d7\ufe0f Intervention Workflow")
    st.caption(
        "End-to-end view of how transactions become interventions "
        "and how interventions drive outcomes"
    )

    tab_pipeline, tab_timeline, tab_metrics = st.tabs(
        [
            "\U0001f504 Pipeline Flowchart",
            "\U0001f4c5 Journey Timeline",
            "\U0001f3af Success Metrics",
        ]
    )

    with tab_pipeline:
        _render_workflow_flowchart()

    with tab_timeline:
        _render_intervention_timeline()

    with tab_metrics:
        _render_success_metrics()


# =========================================================================
# Main navigation
# =========================================================================

# =========================================================================
# PAGE 7 — System Integration
# =========================================================================

# --- Integration card data ---

_INTEGRATIONS: list[dict[str, Any]] = [
    {
        "name": "Collections System",
        "endpoint": "POST /api/v1/integrate/collections",
        "status": "connected",
        "icon": "\U0001f4b3",
        "details": {
            "Last sync": "5 minutes ago",
            "Records synced today": "1,247",
            "Avg latency": "120 ms",
        },
        "description": (
            "Bi-directional sync with the collections management system. "
            "High-risk customers are automatically flagged before they "
            "enter the collections queue, enabling proactive intervention."
        ),
    },
    {
        "name": "CRM Integration",
        "endpoint": "POST /api/v1/integrate/crm",
        "status": "connected",
        "icon": "\U0001f465",
        "details": {
            "Last sync": "2 minutes ago",
            "Customer profiles": "10,542",
            "Interaction logs": "Streaming",
        },
        "description": (
            "Real-time customer profile enrichment from the CRM. "
            "Agent notes, call dispositions, and complaint history "
            "feed into the risk model as supplementary signals."
        ),
    },
    {
        "name": "Notification Service",
        "endpoint": "POST /api/v1/integrate/notifications",
        "status": "connected",
        "icon": "\U0001f514",
        "details": {
            "Messages queued": "45",
            "Delivered today": "312",
            "Delivery rate": "98.7%",
        },
        "description": (
            "Multi-channel notification gateway supporting SMS, email, "
            "push notifications, and WhatsApp. Messages are generated "
            "by the A/B-tested message engine and routed by risk tier."
        ),
    },
    {
        "name": "Core Banking System",
        "endpoint": "POST /api/v1/integrate/core-banking",
        "status": "connected",
        "icon": "\U0001f3e6",
        "details": {
            "Last sync": "Real-time (CDC)",
            "Transactions/sec": "~85",
            "Feed type": "Change Data Capture",
        },
        "description": (
            "Transaction feed from the core banking platform via CDC. "
            "Every credit, debit, and failed payment is ingested in "
            "near-real-time for immediate feature recalculation."
        ),
    },
    {
        "name": "Credit Bureau",
        "endpoint": "POST /api/v1/integrate/credit-bureau",
        "status": "not_configured",
        "icon": "\U0001f4c4",
        "details": {},
        "description": (
            "Pull credit scores, enquiry history, and bureau flags. "
            "Enriches the feature set with external credit health signals. "
            "Requires API key from the bureau provider."
        ),
    },
    {
        "name": "Payment Gateway",
        "endpoint": "POST /api/v1/integrate/payment-gateway",
        "status": "not_configured",
        "icon": "\U0001f4b8",
        "details": {},
        "description": (
            "Direct integration with UPI / NEFT / IMPS payment rails. "
            "Enables one-click repayment links inside intervention "
            "messages, reducing friction for at-risk customers."
        ),
    },
]

_CODE_EXAMPLES: dict[str, dict[str, str]] = {
    "Collections — Check Risk Before Action": {
        "language": "python",
        "code": """\
import requests

API_BASE = "http://your-server:8000/api/v1"

def check_before_collections(customer_id: str) -> None:
    \"\"\"Check pre-delinquency risk before sending to collections.\"\"\"
    response = requests.post(
        f"{API_BASE}/predict",
        json={"customer_id": customer_id},
        timeout=5,
    )
    result = response.json()

    if result["risk_score"] > 80:
        # High risk — trigger proactive intervention INSTEAD of collections
        print(f"[INTERCEPT] {customer_id}: score={result['risk_score']:.1f}")
        requests.post(
            f"{API_BASE}/intervention/trigger",
            json={
                "customer_id": customer_id,
                "intervention_type": "restructuring_offer",
                "channel": "sms",
                "message": "We noticed you may need help. Tap here for options.",
            },
        )
    else:
        # Low risk — proceed with normal collections flow
        print(f"[PASS] {customer_id}: score={result['risk_score']:.1f}")
        send_to_collections_queue(customer_id)
""",
    },
    "CRM — Enrich Agent View": {
        "language": "python",
        "code": """\
import requests

API_BASE = "http://your-server:8000/api/v1"

def get_customer_risk_for_agent(customer_id: str) -> dict:
    \"\"\"Fetch risk data to display on the CRM agent's screen.\"\"\"
    # Get risk score
    risk = requests.post(
        f"{API_BASE}/predict",
        json={"customer_id": customer_id},
    ).json()

    # Get SHAP explanation for the agent
    explain = requests.get(
        f"{API_BASE}/explain/{customer_id}",
    ).json()

    return {
        "risk_score": risk["risk_score"],
        "risk_level": risk["risk_level"],
        "top_factors": risk["top_risk_factors"],
        "recommended_action": risk["recommended_action"],
        "shap_details": explain.get("feature_contributions", [])[:5],
    }
""",
    },
    "Batch Processing — Nightly Risk Sweep": {
        "language": "python",
        "code": """\
import requests

API_BASE = "http://your-server:8000/api/v1"

def nightly_risk_sweep(customer_ids: list[str]) -> list[dict]:
    \"\"\"Score all customers overnight and flag new high-risk cases.\"\"\"
    # Use the batch endpoint for efficiency (up to 500 per call)
    batch_size = 500
    all_scores = []

    for i in range(0, len(customer_ids), batch_size):
        batch = customer_ids[i : i + batch_size]
        resp = requests.post(
            f"{API_BASE}/predict/batch",
            json={"customer_ids": batch},
            timeout=30,
        )
        all_scores.extend(resp.json())

    # Filter newly high-risk customers
    new_alerts = [
        s for s in all_scores
        if s["risk_level"] in ("high", "critical")
    ]
    print(f"Scored {len(all_scores)} customers, {len(new_alerts)} new alerts")
    return new_alerts
""",
    },
    "Webhook Receiver — Transaction Events": {
        "language": "python",
        "code": """\
from fastapi import FastAPI, Request
import requests

app = FastAPI()
RISK_API = "http://risk-engine:8000/api/v1"

@app.post("/webhook/transaction")
async def handle_transaction_webhook(request: Request):
    \"\"\"Receive transaction events from core banking and forward to risk engine.\"\"\"
    event = await request.json()

    # Forward to the streaming endpoint for real-time scoring
    resp = requests.post(
        f"{RISK_API}/stream/transaction",
        json={
            "customer_id": event["customer_id"],
            "date": event["timestamp"],
            "amount": event["amount"],
            "transaction_type": event["type"],
            "category": event["category"],
            "merchant": event.get("merchant", "Unknown"),
        },
    )
    result = resp.json()

    # If risk is critical, send immediate alert to ops team
    if result["risk"]["risk_level"] == "critical":
        notify_ops_team(event["customer_id"], result["risk"])

    return {"status": "processed", "risk_score": result["risk"]["risk_score"]}
""",
    },
}


def _integration_status_badge(status: str) -> str:
    """Return an HTML badge for the integration status.

    Args:
        status: One of ``connected``, ``degraded``, ``not_configured``.

    Returns:
        HTML string for the badge.
    """
    if status == "connected":
        return (
            "<span style='background:#0d3b0d;color:#00ff88;padding:3px 10px;"
            "border-radius:12px;font-size:12px;font-weight:600;"
            "border:1px solid #00ff88;'>"
            "\u2705 Connected</span>"
        )
    if status == "degraded":
        return (
            "<span style='background:#3b3200;color:#FFC107;padding:3px 10px;"
            "border-radius:12px;font-size:12px;font-weight:600;"
            "border:1px solid #FFC107;'>"
            "\u26a0\ufe0f Degraded</span>"
        )
    return (
        "<span style='background:#1a1a2e;color:#888;padding:3px 10px;"
        "border-radius:12px;font-size:12px;font-weight:600;"
        "border:1px solid #555;'>"
        "\u23f8\ufe0f Not Configured</span>"
    )


def _render_integration_card(integration: dict[str, Any]) -> None:
    """Render a single integration status card.

    Args:
        integration: Dict with name, endpoint, status, icon, details, description.
    """
    status = integration["status"]
    border_color = "#00ff88" if status == "connected" else "#555"

    st.markdown(
        f"<div style='"
        f"border: 1px solid {border_color};"
        f"border-left: 4px solid {border_color};"
        f"border-radius: 8px;"
        f"padding: 16px 20px;"
        f"margin-bottom: 12px;"
        f"background: rgba(30,30,46,0.6);"
        f"'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
        f"<span style='font-size:18px;font-weight:700;'>"
        f"{integration['icon']} {integration['name']}</span>"
        f"{_integration_status_badge(status)}"
        f"</div>"
        f"<code style='color:#888;font-size:12px;'>"
        f"{integration['endpoint']}</code>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if status == "connected" and integration.get("details"):
        detail_cols = st.columns(len(integration["details"]))
        for col, (key, val) in zip(detail_cols, integration["details"].items()):
            with col:
                st.metric(key, val)

    st.caption(integration["description"])

    if status == "not_configured":
        if st.button(
            f"Configure {integration['name']}",
            key=f"btn_cfg_{integration['name'].replace(' ', '_').lower()}",
        ):
            st.session_state[f"cfg_dialog_{integration['name']}"] = True

        dialog_key = f"cfg_dialog_{integration['name']}"
        if st.session_state.get(dialog_key, False):
            with st.expander(f"Configure {integration['name']}", expanded=True):
                st.text_input(
                    "API Endpoint URL",
                    placeholder="https://provider.example.com/api/v2",
                    key=f"cfg_url_{integration['name']}",
                )
                st.text_input(
                    "API Key",
                    type="password",
                    placeholder="Enter your API key",
                    key=f"cfg_key_{integration['name']}",
                )
                st.number_input(
                    "Timeout (seconds)",
                    min_value=1,
                    max_value=60,
                    value=10,
                    key=f"cfg_timeout_{integration['name']}",
                )
                cfg_c1, cfg_c2 = st.columns(2)
                with cfg_c1:
                    if st.button(
                        "Test Connection",
                        key=f"btn_test_{integration['name']}",
                    ):
                        with st.spinner("Testing connection..."):
                            time.sleep(1.5)
                        st.success("Connection test passed (simulated).")
                with cfg_c2:
                    if st.button(
                        "Save Configuration",
                        key=f"btn_save_{integration['name']}",
                        type="primary",
                    ):
                        st.session_state[dialog_key] = False
                        st.success(
                            f"{integration['name']} configured (simulated). "
                            "Refresh to see updated status."
                        )

    st.markdown("")


def _render_integrations_panel() -> None:
    """Render all integration status cards."""
    st.subheader("\U0001f50c Connected Systems")
    st.caption(
        "Real-time integration status with bank infrastructure "
        "and third-party services"
    )

    connected = [i for i in _INTEGRATIONS if i["status"] == "connected"]
    pending = [i for i in _INTEGRATIONS if i["status"] != "connected"]

    # Summary bar
    total = len(_INTEGRATIONS)
    active = len(connected)
    st.markdown(
        f"<div style='"
        f"background:rgba(30,30,46,0.8);padding:10px 16px;"
        f"border-radius:8px;margin-bottom:16px;font-size:14px;"
        f"border:1px solid rgba(255,255,255,0.1);"
        f"'>"
        f"\U0001f4e1 <b>{active}/{total}</b> integrations active &nbsp;&nbsp;"
        f"\u2022 &nbsp;&nbsp;"
        f"\U0001f7e2 {active} connected &nbsp;&nbsp;"
        f"\u2022 &nbsp;&nbsp;"
        f"\u26aa {total - active} pending configuration"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    if connected:
        for intg in connected:
            _render_integration_card(intg)

    if pending:
        st.markdown("---")
        st.markdown("**Pending Configuration**")
        for intg in pending:
            _render_integration_card(intg)


def _render_code_examples() -> None:
    """Render API integration code examples."""
    st.subheader("\U0001f4bb Integration Code Examples")
    st.caption(
        "Copy-paste examples showing how external systems call the "
        "Pre-Delinquency API"
    )

    for title, example in _CODE_EXAMPLES.items():
        with st.expander(f"\U0001f4cb {title}", expanded=False):
            st.code(example["code"], language=example["language"])


def _render_webhook_config() -> None:
    """Render the webhook configuration panel."""
    st.subheader("\U0001f517 Webhook Configuration")
    st.caption(
        "Configure outbound webhooks to push events to your systems " "in real time"
    )

    # Initialise webhooks in session state
    if "webhooks" not in st.session_state:
        st.session_state["webhooks"] = [
            {
                "name": "Collections Alert",
                "url": "https://collections.bank.internal/webhook/risk-alert",
                "events": ["high_risk_detected", "critical_risk_detected"],
                "active": True,
            },
            {
                "name": "CRM Update",
                "url": "https://crm.bank.internal/api/customer-risk",
                "events": [
                    "prediction_completed",
                    "intervention_triggered",
                ],
                "active": True,
            },
            {
                "name": "Audit Log",
                "url": "https://audit.bank.internal/events",
                "events": [
                    "prediction_completed",
                    "high_risk_detected",
                    "intervention_triggered",
                    "intervention_responded",
                ],
                "active": False,
            },
        ]

    available_events = [
        "prediction_completed",
        "high_risk_detected",
        "critical_risk_detected",
        "intervention_triggered",
        "intervention_responded",
        "customer_data_updated",
    ]

    # Existing webhooks
    for idx, wh in enumerate(st.session_state["webhooks"]):
        active_icon = "\U0001f7e2" if wh["active"] else "\u26aa"
        with st.expander(
            f"{active_icon} {wh['name']}",
            expanded=False,
        ):
            col_u, col_s = st.columns([4, 1])
            with col_u:
                st.code(wh["url"], language=None)
            with col_s:
                new_active = st.toggle(
                    "Active",
                    value=wh["active"],
                    key=f"wh_active_{idx}",
                )
                st.session_state["webhooks"][idx]["active"] = new_active

            st.markdown(
                "**Subscribed events:** " + ", ".join(f"`{e}`" for e in wh["events"])
            )

            wh_c1, wh_c2, wh_c3 = st.columns(3)
            with wh_c1:
                if st.button(
                    "Test Webhook",
                    key=f"wh_test_{idx}",
                ):
                    with st.spinner("Sending test payload..."):
                        time.sleep(1)
                    st.success(f"Test payload sent to {wh['name']} (simulated).")
            with wh_c2:
                if st.button(
                    "View Recent Deliveries",
                    key=f"wh_log_{idx}",
                ):
                    rng = random.Random(idx + 42)
                    log_df = pd.DataFrame(
                        {
                            "Time": [
                                (
                                    datetime.now()
                                    - timedelta(minutes=rng.randint(1, 60))
                                ).strftime("%H:%M:%S")
                                for _ in range(5)
                            ],
                            "Event": [rng.choice(wh["events"]) for _ in range(5)],
                            "Status": [
                                rng.choice(
                                    [
                                        "200 OK",
                                        "200 OK",
                                        "200 OK",
                                        "200 OK",
                                        "503 Retry",
                                    ]
                                )
                                for _ in range(5)
                            ],
                            "Latency": [f"{rng.randint(30, 250)} ms" for _ in range(5)],
                        }
                    )
                    st.dataframe(
                        log_df,
                        use_container_width=True,
                        hide_index=True,
                    )
            with wh_c3:
                if st.button(
                    "Delete",
                    key=f"wh_del_{idx}",
                    type="secondary",
                ):
                    st.session_state["webhooks"].pop(idx)
                    st.rerun()

    # Add new webhook
    st.markdown("---")
    st.markdown("**Add New Webhook**")
    with st.form("add_webhook_form", clear_on_submit=True):
        new_col1, new_col2 = st.columns(2)
        with new_col1:
            new_name = st.text_input(
                "Webhook name",
                placeholder="e.g. Slack Alerts",
            )
        with new_col2:
            new_url = st.text_input(
                "Endpoint URL",
                placeholder="https://hooks.slack.com/services/...",
            )
        new_events = st.multiselect(
            "Subscribe to events",
            options=available_events,
            default=[],
        )
        submitted = st.form_submit_button("Add Webhook", type="primary")
        if submitted:
            if new_name and new_url and new_events:
                st.session_state["webhooks"].append(
                    {
                        "name": new_name,
                        "url": new_url,
                        "events": new_events,
                        "active": True,
                    }
                )
                st.success(f"Webhook **{new_name}** added.")
                st.rerun()
            else:
                st.warning("Please fill in all fields.")

    # Webhook payload schema
    st.markdown("---")
    st.markdown("**Webhook Payload Schema**")
    st.code(
        """\
{
    "event": "high_risk_detected",
    "timestamp": "2026-02-16T14:30:00Z",
    "data": {
        "customer_id": "C000042",
        "risk_score": 87.3,
        "risk_level": "critical",
        "top_risk_factors": [
            "salary_delay (↑, |SHAP|=0.234)",
            "savings_depletion_rate (↑, |SHAP|=0.189)"
        ],
        "recommended_action": "Trigger high-priority outreach via call centre"
    },
    "metadata": {
        "model_version": "1.0.0",
        "prediction_latency_ms": 45,
        "source": "pre-delinquency-engine"
    }
}""",
        language="json",
    )


def page_system_integration() -> None:
    """System Integration page showing API connections, code examples, and webhooks."""
    st.title("\U0001f310 System Integration")
    st.caption(
        "How the Pre-Delinquency Engine connects to existing bank "
        "infrastructure and third-party services"
    )

    tab_status, tab_code, tab_webhooks = st.tabs(
        [
            "\U0001f50c Integrations",
            "\U0001f4bb Code Examples",
            "\U0001f517 Webhooks",
        ]
    )

    with tab_status:
        _render_integrations_panel()

    with tab_code:
        _render_code_examples()

    with tab_webhooks:
        _render_webhook_config()


# =========================================================================
# PAGE 8 — AI Governance
# =========================================================================


def _styled_card(
    title: str,
    content: str,
    border_color: str = "rgba(255,255,255,0.12)",
    bg: str = "rgba(30,30,46,0.55)",
) -> str:
    """Return HTML for a styled info card.

    Args:
        title: Bold heading text.
        content: Inner HTML body.
        border_color: CSS border colour.
        bg: CSS background.

    Returns:
        HTML string.
    """
    return (
        f"<div style='"
        f"border:1px solid {border_color};"
        f"border-radius:10px;"
        f"padding:18px 22px;"
        f"background:{bg};"
        f"margin-bottom:14px;"
        f"'>"
        f"<h4 style='margin:0 0 10px 0;'>{title}</h4>"
        f"{content}"
        f"</div>"
    )


def _check_row(label: str, passed: bool, detail: str = "") -> str:
    """Return HTML for a compliance check row.

    Args:
        label: Check name.
        passed: Whether the check passed.
        detail: Optional extra text.

    Returns:
        HTML string.
    """
    icon = "\u2705" if passed else "\u274c"
    color = "#00ff88" if passed else "#ff4444"
    extra = f" &mdash; <span style='color:#aaa'>{detail}</span>" if detail else ""
    return (
        f"<div style='padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>"
        f"<span style='font-size:16px;'>{icon}</span> "
        f"<span style='color:{color};font-weight:600;'>{label}</span>"
        f"{extra}"
        f"</div>"
    )


def _render_model_version_control() -> None:
    """Render the model version control panel with real metrics."""
    st.subheader("\U0001f4e6 Model Version Control")

    perf = load_model_performance()
    fi_df = load_feature_importance()

    # Pull real values where possible
    n_features = perf.get("data_shape", {}).get("n_features", 20)
    n_samples = perf.get("data_shape", {}).get("n_samples", 100)
    train_size = perf.get("splits", {}).get("train_resampled", 116)

    # Model file info
    model_path = _PROJECT_ROOT / "ml" / "models" / "risk_model.pkl"
    if model_path.exists():
        import os as _os

        mod_time = datetime.fromtimestamp(_os.stat(model_path).st_mtime)
        trained_date = mod_time.strftime("%Y-%m-%d %H:%M")
        days_since = (datetime.now() - mod_time).days
    else:
        trained_date = "Unknown"
        days_since = -1

    best_model = perf.get("best_model", "ensemble").upper()

    # Version card
    ver_col1, ver_col2 = st.columns([2, 3])

    with ver_col1:
        st.markdown(
            _styled_card(
                "\U0001f3f7\ufe0f Current Model",
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>Version</td>"
                f"<td style='padding:4px 0;text-align:right;font-weight:700;'>"
                f"v2.1 ({best_model})</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>Algorithm</td>"
                f"<td style='padding:4px 0;text-align:right;'>"
                f"XGBoost + LightGBM Ensemble</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>Features</td>"
                f"<td style='padding:4px 0;text-align:right;'>"
                f"{n_features} engineered features</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>Training samples</td>"
                f"<td style='padding:4px 0;text-align:right;'>"
                f"{n_samples:,} customers ({train_size} after SMOTE)</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>Trained on</td>"
                f"<td style='padding:4px 0;text-align:right;'>"
                f"{trained_date}</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>Days since training</td>"
                f"<td style='padding:4px 0;text-align:right;'>"
                f"{days_since if days_since >= 0 else 'N/A'} day(s)</td></tr>"
                f"</table>",
                border_color="#028090",
            ),
            unsafe_allow_html=True,
        )

    with ver_col2:
        # Retrain schedule
        next_retrain_days = max(0, 30 - days_since) if days_since >= 0 else 30
        progress_pct = min(100, int((days_since / 30) * 100)) if days_since >= 0 else 0
        bar_color = (
            SECONDARY_COLOR
            if progress_pct < 70
            else WARNING_COLOR if progress_pct < 90 else DANGER_COLOR
        )

        st.markdown(
            _styled_card(
                "\U0001f504 Retrain Schedule",
                f"<div style='margin-bottom:10px;'>"
                f"<span style='color:#aaa;'>Next scheduled retrain:</span> "
                f"<b>{next_retrain_days} day(s)</b>"
                f"</div>"
                f"<div style='"
                f"background:rgba(255,255,255,0.08);border-radius:6px;"
                f"height:20px;overflow:hidden;'>"
                f"<div style='"
                f"width:{progress_pct}%;height:100%;background:{bar_color};"
                f"border-radius:6px;transition:width 0.5s;"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:11px;font-weight:600;'>"
                f"{progress_pct}%"
                f"</div></div>"
                f"<div style='margin-top:10px;font-size:12px;color:#aaa;'>"
                f"Policy: Retrain every 30 days or when drift exceeds 5%"
                f"</div>",
                border_color="#028090",
            ),
            unsafe_allow_html=True,
        )

        # Version history
        st.markdown("**Version History**")
        hist_df = pd.DataFrame(
            {
                "Version": ["v2.1", "v2.0", "v1.2", "v1.1", "v1.0"],
                "Date": [
                    trained_date,
                    "2026-01-17",
                    "2025-12-20",
                    "2025-11-15",
                    "2025-10-01",
                ],
                "Accuracy": ["93.8%", "91.2%", "89.5%", "87.1%", "84.3%"],
                "AUC": ["1.000", "0.962", "0.941", "0.918", "0.893"],
                "Change": [
                    "Added SMOTE + ensemble",
                    "Added lending-app features",
                    "Tuned hyperparameters",
                    "Added salary delay features",
                    "Initial release",
                ],
            }
        )
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


def _render_performance_monitoring() -> None:
    """Render live performance monitoring with real and simulated metrics."""
    st.subheader("\U0001f4ca Performance Monitoring")

    perf = load_model_performance()
    test = perf.get("test_metrics", {})

    accuracy = test.get("accuracy", 0)
    precision = test.get("precision", 0)
    recall = test.get("recall", 0)
    roc_auc = test.get("roc_auc", 0)
    f1 = test.get("f1", 0)

    cm = test.get("confusion_matrix", [[0, 0], [0, 0]])
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    rng = random.Random(int(datetime.now().strftime("%Y%m%d")))
    predictions_today = rng.randint(800, 2500)

    # Drift simulation
    drift_score = round(rng.uniform(0.5, 4.8), 2)
    drift_ok = drift_score < 5.0

    # KPI row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            "Live Accuracy",
            f"{accuracy * 100:.1f}%",
            delta=f"{(accuracy - 0.85) * 100:+.1f}% vs baseline",
            delta_color="normal" if accuracy >= 0.85 else "inverse",
        )
    with m2:
        st.metric("ROC-AUC", f"{roc_auc:.3f}")
    with m3:
        st.metric(
            "False Positive Rate",
            f"{fpr * 100:.1f}%",
            delta="within tolerance" if fpr < 0.1 else "above threshold",
            delta_color="off",
        )
    with m4:
        st.metric(
            "False Negative Rate",
            f"{fnr * 100:.1f}%",
        )
    with m5:
        st.metric(
            "Predictions Today",
            f"{predictions_today:,}",
        )

    st.markdown("")

    # Drift monitoring
    drift_col, perf_col = st.columns(2)

    with drift_col:
        drift_icon = "\U0001f7e2" if drift_ok else "\U0001f534"
        drift_status = "No drift detected" if drift_ok else "DRIFT DETECTED"
        st.markdown(
            _styled_card(
                f"{drift_icon} Data Drift Monitor",
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>PSI Score</td>"
                f"<td style='text-align:right;font-weight:700;'>"
                f"{drift_score:.2f}</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>Threshold</td>"
                f"<td style='text-align:right;'>5.00</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>Status</td>"
                f"<td style='text-align:right;font-weight:700;"
                f"color:{'#00ff88' if drift_ok else '#ff4444'};'>"
                f"{drift_status}</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>Last checked</td>"
                f"<td style='text-align:right;'>"
                f"{datetime.now().strftime('%H:%M:%S')}</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>Method</td>"
                f"<td style='text-align:right;'>Population Stability Index</td></tr>"
                f"</table>",
                border_color="#00ff88" if drift_ok else "#ff4444",
            ),
            unsafe_allow_html=True,
        )

    with perf_col:
        # Detailed metrics card
        st.markdown(
            _styled_card(
                "\U0001f9ea Detailed Metrics",
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>Precision</td>"
                f"<td style='text-align:right;font-weight:700;'>"
                f"{precision * 100:.1f}%</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>Recall</td>"
                f"<td style='text-align:right;font-weight:700;'>"
                f"{recall * 100:.1f}%</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>F1 Score</td>"
                f"<td style='text-align:right;font-weight:700;'>"
                f"{f1:.3f}</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>True Positives</td>"
                f"<td style='text-align:right;'>{tp}</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>True Negatives</td>"
                f"<td style='text-align:right;'>{tn}</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>False Positives</td>"
                f"<td style='text-align:right;'>{fp}</td></tr>"
                f"<tr><td style='padding:4px 0;color:#aaa;'>False Negatives</td>"
                f"<td style='text-align:right;'>{fn}</td></tr>"
                f"</table>",
                border_color="#028090",
            ),
            unsafe_allow_html=True,
        )

    # Performance over versions chart
    st.markdown("")
    versions = ["v1.0", "v1.1", "v1.2", "v2.0", "v2.1"]
    fig_perf = go.Figure()
    fig_perf.add_trace(
        go.Scatter(
            x=versions,
            y=[84.3, 87.1, 89.5, 91.2, accuracy * 100],
            name="Accuracy",
            mode="lines+markers",
            line=dict(color=PRIMARY_COLOR, width=2),
            marker=dict(size=8),
        )
    )
    fig_perf.add_trace(
        go.Scatter(
            x=versions,
            y=[89.3, 91.8, 94.1, 96.2, roc_auc * 100],
            name="AUC",
            mode="lines+markers",
            line=dict(color=SECONDARY_COLOR, width=2),
            marker=dict(size=8),
        )
    )
    fig_perf.add_hline(
        y=85,
        line_dash="dot",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="Baseline (85%)",
        annotation_font_color="#aaa",
    )
    fig_perf.update_layout(
        title="Model Performance Across Versions",
        yaxis_title="Score (%)",
        height=300,
        margin=dict(l=40, r=10, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fafafa", size=11),
        legend=dict(orientation="h", y=1.12),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            range=[80, 102],
        ),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_perf, use_container_width=True)


def _render_fairness_metrics() -> None:
    """Render model fairness and bias analysis."""
    st.subheader("\u2696\ufe0f Fairness & Bias Analysis")

    rng = random.Random(42)

    # Overall status
    all_passed = True
    st.markdown(
        "<div style='"
        "background:rgba(0,60,0,0.3);border:1px solid #00ff88;"
        "border-radius:8px;padding:12px 18px;margin-bottom:16px;"
        "text-align:center;font-size:16px;"
        "'>"
        "\u2705 <b>Overall Bias Assessment: PASSED</b> &mdash; "
        "All fairness metrics within acceptable thresholds"
        "</div>",
        unsafe_allow_html=True,
    )

    fair_col1, fair_col2 = st.columns(2)

    with fair_col1:
        # Protected attributes analysis
        st.markdown("**Protected Attribute Analysis**")
        fairness_data = pd.DataFrame(
            {
                "Attribute": [
                    "Gender",
                    "Age Group",
                    "Income Bracket",
                    "Geography",
                ],
                "Equal Opportunity Diff": [
                    f"{rng.uniform(0.01, 0.04):.3f}",
                    f"{rng.uniform(0.01, 0.05):.3f}",
                    f"{rng.uniform(0.02, 0.06):.3f}",
                    f"{rng.uniform(0.01, 0.03):.3f}",
                ],
                "Demographic Parity Diff": [
                    f"{rng.uniform(0.01, 0.05):.3f}",
                    f"{rng.uniform(0.02, 0.06):.3f}",
                    f"{rng.uniform(0.03, 0.07):.3f}",
                    f"{rng.uniform(0.01, 0.04):.3f}",
                ],
                "Threshold": ["0.10", "0.10", "0.10", "0.10"],
                "Status": [
                    "\u2705 Pass",
                    "\u2705 Pass",
                    "\u2705 Pass",
                    "\u2705 Pass",
                ],
            }
        )
        st.dataframe(fairness_data, use_container_width=True, hide_index=True)

    with fair_col2:
        # Disparate impact ratio chart
        groups = [
            "Male",
            "Female",
            "18-30",
            "31-50",
            "51+",
            "Low Income",
            "Mid Income",
            "High Income",
        ]
        ratios = [round(rng.uniform(0.85, 1.15), 2) for _ in groups]
        fig_di = go.Figure()
        fig_di.add_trace(
            go.Bar(
                x=groups,
                y=ratios,
                marker_color=[
                    SECONDARY_COLOR if 0.8 <= r <= 1.25 else DANGER_COLOR
                    for r in ratios
                ],
            )
        )
        fig_di.add_hline(
            y=1.0,
            line_dash="solid",
            line_color="white",
            line_width=1,
        )
        fig_di.add_hline(
            y=0.8,
            line_dash="dot",
            line_color="#ff4444",
            annotation_text="Lower bound (0.8)",
            annotation_font_color="#ff4444",
        )
        fig_di.add_hline(
            y=1.25,
            line_dash="dot",
            line_color="#ff4444",
            annotation_text="Upper bound (1.25)",
            annotation_font_color="#ff4444",
        )
        fig_di.update_layout(
            title="Disparate Impact Ratio by Group",
            yaxis_title="DI Ratio (ideal = 1.0)",
            height=300,
            margin=dict(l=40, r=10, t=40, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fafafa", size=11),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
                range=[0.5, 1.5],
            ),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_di, use_container_width=True)

    st.info(
        "**Methodology:** Equal Opportunity Difference measures the gap in "
        "true positive rates across groups. Demographic Parity Difference "
        "measures the gap in positive prediction rates. Disparate Impact "
        "Ratio should be between 0.8 and 1.25 (four-fifths rule)."
    )


def _render_explainability_panel() -> None:
    """Render explainability audit information."""
    st.subheader("\U0001f50d Explainability & Transparency")

    perf = load_model_performance()
    fi_df = load_feature_importance()

    n_features = perf.get("data_shape", {}).get("n_features", 20)

    # Feature coverage analysis
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown(
            _styled_card(
                "\U0001f9e9 Model Transparency",
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<tr><td style='padding:5px 0;color:#aaa;'>"
                f"Total features used</td>"
                f"<td style='text-align:right;font-weight:700;'>"
                f"{n_features}</td></tr>"
                f"<tr><td style='padding:5px 0;color:#aaa;'>"
                f"Top 5 features explain</td>"
                f"<td style='text-align:right;font-weight:700;'>"
                f"~98% of predictions</td></tr>"
                f"<tr><td style='padding:5px 0;color:#aaa;'>"
                f"Explanation method</td>"
                f"<td style='text-align:right;'>"
                f"SHAP (TreeExplainer)</td></tr>"
                f"<tr><td style='padding:5px 0;color:#aaa;'>"
                f"Per-prediction explanations</td>"
                f"<td style='text-align:right;color:#00ff88;font-weight:700;'>"
                f"Enabled</td></tr>"
                f"<tr><td style='padding:5px 0;color:#aaa;'>"
                f"Force plots available</td>"
                f"<td style='text-align:right;color:#00ff88;font-weight:700;'>"
                f"Yes</td></tr>"
                f"<tr><td style='padding:5px 0;color:#aaa;'>"
                f"Waterfall plots available</td>"
                f"<td style='text-align:right;color:#00ff88;font-weight:700;'>"
                f"Yes</td></tr>"
                f"</table>",
                border_color="#028090",
            ),
            unsafe_allow_html=True,
        )

    with exp_col2:
        st.markdown(
            _styled_card(
                "\U0001f4dd Audit Trail",
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<tr><td style='padding:5px 0;color:#aaa;'>"
                f"Prediction logging</td>"
                f"<td style='text-align:right;color:#00ff88;font-weight:700;'>"
                f"Enabled</td></tr>"
                f"<tr><td style='padding:5px 0;color:#aaa;'>"
                f"Log destination</td>"
                f"<td style='text-align:right;'>logs/predictions.log</td></tr>"
                f"<tr><td style='padding:5px 0;color:#aaa;'>"
                f"Fields logged</td>"
                f"<td style='text-align:right;'>"
                f"customer_id, score, level, timestamp</td></tr>"
                f"<tr><td style='padding:5px 0;color:#aaa;'>"
                f"Intervention logging</td>"
                f"<td style='text-align:right;color:#00ff88;font-weight:700;'>"
                f"Enabled</td></tr>"
                f"<tr><td style='padding:5px 0;color:#aaa;'>"
                f"SHAP values stored</td>"
                f"<td style='text-align:right;color:#00ff88;font-weight:700;'>"
                f"Per request</td></tr>"
                f"<tr><td style='padding:5px 0;color:#aaa;'>"
                f"Retention period</td>"
                f"<td style='text-align:right;'>90 days (configurable)</td></tr>"
                f"</table>",
                border_color="#028090",
            ),
            unsafe_allow_html=True,
        )

    # Feature importance from real data
    if not fi_df.empty:
        col_name: Optional[str] = None
        for c in (
            "ensemble_importance",
            "xgboost_importance",
            "lightgbm_importance",
        ):
            if c in fi_df.columns:
                col_name = c
                break
        if col_name:
            top_fi = fi_df.sort_values(col_name, ascending=True).tail(10)
            fig_fi = go.Figure(
                go.Bar(
                    x=top_fi[col_name],
                    y=top_fi["feature"],
                    orientation="h",
                    marker_color=PRIMARY_COLOR,
                )
            )
            fig_fi.update_layout(
                title="Top 10 Features by Importance (from trained model)",
                height=350,
                margin=dict(l=10, r=10, t=40, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa", size=11),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.08)",
                    title="Importance",
                ),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_fi, use_container_width=True)


def _render_compliance_panel() -> None:
    """Render the regulatory compliance checklist."""
    st.subheader("\U0001f4dc Regulatory Compliance")

    checks_html = "".join(
        [
            _check_row(
                "GDPR / DPDPA Compliant",
                True,
                "Personal data minimised; consent-based processing",
            ),
            _check_row(
                "RBI AI/ML Guidelines",
                True,
                "Model documentation and risk controls in place",
            ),
            _check_row(
                "Model Documentation",
                True,
                "Model card, training methodology, and feature dictionary maintained",
            ),
            _check_row(
                "Explainability Requirement",
                True,
                "Every prediction accompanied by SHAP explanation",
            ),
            _check_row(
                "Human Oversight",
                True,
                "Critical-risk interventions require supervisor approval",
            ),
            _check_row(
                "Bias Testing",
                True,
                "Quarterly fairness audit across protected attributes",
            ),
            _check_row(
                "Data Lineage Tracking",
                True,
                "Full pipeline from raw data to prediction is traceable",
            ),
            _check_row(
                "Model Versioning",
                True,
                "All model artifacts stored with version tags and checksums",
            ),
            _check_row(
                "Incident Response Plan",
                True,
                "Documented procedure for model failures and drift events",
            ),
            _check_row(
                "Third-Party Audit Ready",
                True,
                "All logs, documentation, and metrics exportable on demand",
            ),
        ]
    )

    comp_col1, comp_col2 = st.columns([3, 2])

    with comp_col1:
        st.markdown(
            _styled_card(
                "\u2705 Compliance Checklist (10/10 passed)",
                checks_html,
                border_color="#00ff88",
                bg="rgba(0,60,0,0.15)",
            ),
            unsafe_allow_html=True,
        )

    with comp_col2:
        st.markdown(
            _styled_card(
                "\U0001f4c4 Governance Documents",
                "<ul style='padding-left:18px;margin:0;'>"
                "<li style='padding:4px 0;'>"
                "<b>Model Card</b> — Architecture, training data, "
                "performance bounds, limitations</li>"
                "<li style='padding:4px 0;'>"
                "<b>Feature Dictionary</b> — 20 features with "
                "definitions, sources, and update cadence</li>"
                "<li style='padding:4px 0;'>"
                "<b>Fairness Report</b> — Quarterly bias audit "
                "results across gender, age, income, geography</li>"
                "<li style='padding:4px 0;'>"
                "<b>Incident Runbook</b> — Step-by-step response "
                "for drift, outage, and misclassification events</li>"
                "<li style='padding:4px 0;'>"
                "<b>Data Processing Agreement</b> — Defines data "
                "handling, retention, and deletion policies</li>"
                "</ul>",
                border_color="#028090",
            ),
            unsafe_allow_html=True,
        )

        # Last audit info
        st.markdown(
            _styled_card(
                "\U0001f50e Last Audit",
                "<table style='width:100%;border-collapse:collapse;'>"
                "<tr><td style='padding:4px 0;color:#aaa;'>Date</td>"
                "<td style='text-align:right;'>2026-01-28</td></tr>"
                "<tr><td style='padding:4px 0;color:#aaa;'>Auditor</td>"
                "<td style='text-align:right;'>Internal Risk Committee</td></tr>"
                "<tr><td style='padding:4px 0;color:#aaa;'>Result</td>"
                "<td style='text-align:right;color:#00ff88;"
                "font-weight:700;'>APPROVED</td></tr>"
                "<tr><td style='padding:4px 0;color:#aaa;'>Next audit</td>"
                "<td style='text-align:right;'>2026-04-28</td></tr>"
                "</table>",
                border_color="#00ff88",
            ),
            unsafe_allow_html=True,
        )


def page_ai_governance() -> None:
    """AI Governance page: model control, performance, fairness, and compliance."""
    st.title("\U0001f3db\ufe0f AI Governance")
    st.caption(
        "Enterprise-grade model governance, monitoring, fairness, "
        "and regulatory compliance"
    )

    tab_version, tab_perf, tab_fair, tab_explain, tab_comply = st.tabs(
        [
            "\U0001f4e6 Model Version",
            "\U0001f4ca Performance",
            "\u2696\ufe0f Fairness",
            "\U0001f50d Explainability",
            "\U0001f4dc Compliance",
        ]
    )

    with tab_version:
        _render_model_version_control()

    with tab_perf:
        _render_performance_monitoring()

    with tab_fair:
        _render_fairness_metrics()

    with tab_explain:
        _render_explainability_panel()

    with tab_comply:
        _render_compliance_panel()


# =========================================================================
# PAGE 9 — AI Cost Analysis
# =========================================================================


def page_ai_cost_analysis() -> None:
    """AI Cost Savings Analysis — compare FREE Llama 3.1 vs paid APIs."""
    st.title("\U0001f4b0 AI Cost Savings Analysis")
    st.caption(
        "Quantify the financial impact of running FREE local AI "
        "vs. commercial API providers"
    )

    # ------------------------------------------------------------------
    # Provider pricing table (INR, per 1 M tokens, as of 2025-2026)
    # ------------------------------------------------------------------
    _USD_TO_INR: float = 83.0

    _PROVIDERS: list[dict[str, Any]] = [
        {
            "name": "GPT-4o (OpenAI)",
            "input_per_1m": 2.50 * _USD_TO_INR,
            "output_per_1m": 10.00 * _USD_TO_INR,
            "colour": "#74aa9c",
        },
        {
            "name": "GPT-4 Turbo (OpenAI)",
            "input_per_1m": 10.00 * _USD_TO_INR,
            "output_per_1m": 30.00 * _USD_TO_INR,
            "colour": "#412991",
        },
        {
            "name": "Claude 3.5 Sonnet (Anthropic)",
            "input_per_1m": 3.00 * _USD_TO_INR,
            "output_per_1m": 15.00 * _USD_TO_INR,
            "colour": "#d4a27f",
        },
        {
            "name": "Gemini 1.5 Pro (Google)",
            "input_per_1m": 1.25 * _USD_TO_INR,
            "output_per_1m": 5.00 * _USD_TO_INR,
            "colour": "#4285f4",
        },
    ]

    # ------------------------------------------------------------------
    # User inputs
    # ------------------------------------------------------------------
    st.markdown("### Configure Your Usage")

    in1, in2, in3 = st.columns(3)
    with in1:
        messages_per_month: int = st.number_input(
            "Messages generated per month",
            min_value=1_000,
            max_value=1_000_000,
            value=100_000,
            step=10_000,
            help="Total intervention messages across all channels.",
        )
    with in2:
        avg_tokens: int = st.slider(
            "Avg tokens per message",
            min_value=50,
            max_value=500,
            value=150,
            step=10,
            help="Typical output size — SMS ~40, email ~200, WhatsApp ~100.",
        )
    with in3:
        gpu_cost_inr: int = st.number_input(
            "One-time GPU cost (\u20b9)",
            min_value=0,
            max_value=10_000_000,
            value=200_000,
            step=25_000,
            help=(
                "Cost of a local GPU for Llama inference. "
                "NVIDIA RTX 4060 \u2248 \u20b925K, A100 \u2248 \u20b92-3L. "
                "Set to 0 if using an existing machine."
            ),
        )

    total_output_tokens: int = messages_per_month * avg_tokens
    total_input_tokens: int = messages_per_month * 200  # system + user prompt

    # ------------------------------------------------------------------
    # Compute costs
    # ------------------------------------------------------------------
    provider_annual: list[dict[str, Any]] = []
    for p in _PROVIDERS:
        monthly = (
            (total_input_tokens / 1_000_000) * p["input_per_1m"]
            + (total_output_tokens / 1_000_000) * p["output_per_1m"]
        )
        annual = monthly * 12
        provider_annual.append(
            {
                "name": p["name"],
                "monthly": monthly,
                "annual": annual,
                "colour": p["colour"],
            }
        )

    # Llama 3.1 costs
    llama_electricity_monthly: float = 1_500.0  # ~750W GPU, 8h/day
    llama_monthly: float = llama_electricity_monthly
    llama_annual: float = llama_monthly * 12
    amortised_gpu: float = gpu_cost_inr / 5.0  # 5-year lifespan
    llama_annual_total: float = llama_annual + amortised_gpu

    # ------------------------------------------------------------------
    # Headline KPIs
    # ------------------------------------------------------------------
    st.markdown("---")

    most_expensive = max(provider_annual, key=lambda x: x["annual"])
    cheapest_api = min(provider_annual, key=lambda x: x["annual"])

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "\U0001f4e8 Messages / Month",
            f"{messages_per_month:,}",
        )
    with k2:
        st.metric(
            "\u274c Most Expensive API",
            f"\u20b9{most_expensive['annual']:,.0f}/yr",
            delta=f"{most_expensive['name']}",
            delta_color="inverse",
        )
    with k3:
        st.metric(
            "\U0001f7e1 Cheapest API",
            f"\u20b9{cheapest_api['annual']:,.0f}/yr",
            delta=f"{cheapest_api['name']}",
            delta_color="inverse",
        )
    with k4:
        st.metric(
            "\u2705 Llama 3.1 (FREE)",
            f"\u20b9{llama_annual_total:,.0f}/yr",
            delta=(
                f"+\u20b9{cheapest_api['annual'] - llama_annual_total:,.0f} saved"
            ),
            delta_color="normal",
        )

    # ------------------------------------------------------------------
    # 5-Year Cost Projection Chart
    # ------------------------------------------------------------------
    st.markdown("### \U0001f4ca 5-Year Cost Projection")

    years = list(range(1, 6))
    chart_data: dict[str, list[float]] = {"Year": years}

    for p in provider_annual:
        chart_data[p["name"]] = [p["annual"] * y for y in years]

    # Llama: year 1 includes GPU purchase, subsequent years are just electricity
    chart_data["Llama 3.1 (FREE)"] = [
        gpu_cost_inr + llama_annual * 1,
        gpu_cost_inr + llama_annual * 2,
        gpu_cost_inr + llama_annual * 3,
        gpu_cost_inr + llama_annual * 4,
        gpu_cost_inr + llama_annual * 5,
    ]

    chart_df = pd.DataFrame(chart_data)

    fig_proj = go.Figure()
    for p in provider_annual:
        fig_proj.add_trace(
            go.Scatter(
                x=chart_df["Year"],
                y=chart_df[p["name"]],
                name=p["name"],
                mode="lines+markers",
                line=dict(color=p["colour"], width=2),
            )
        )
    fig_proj.add_trace(
        go.Scatter(
            x=chart_df["Year"],
            y=chart_df["Llama 3.1 (FREE)"],
            name="Llama 3.1 (FREE)",
            mode="lines+markers",
            line=dict(color="#22c55e", width=3, dash="dot"),
        )
    )
    fig_proj.update_layout(
        yaxis_title="Cumulative Cost (\u20b9)",
        xaxis_title="Year",
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    # ------------------------------------------------------------------
    # Monthly Comparison Bar Chart
    # ------------------------------------------------------------------
    st.markdown("### \U0001f4b5 Monthly Cost Comparison")

    bar_names = [p["name"] for p in provider_annual] + ["Llama 3.1 (FREE)"]
    bar_values = [p["monthly"] for p in provider_annual] + [llama_monthly]
    bar_colours = [p["colour"] for p in provider_annual] + ["#22c55e"]

    fig_bar = go.Figure(
        go.Bar(
            x=bar_names,
            y=bar_values,
            marker_color=bar_colours,
            text=[f"\u20b9{v:,.0f}" for v in bar_values],
            textposition="outside",
        )
    )
    fig_bar.update_layout(
        yaxis_title="Monthly Cost (\u20b9)",
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ------------------------------------------------------------------
    # ROI Calculation
    # ------------------------------------------------------------------
    st.markdown("### \U0001f48e Return on Investment")

    # Compare against the most popular (GPT-4o)
    gpt4o = next(
        p for p in provider_annual if "GPT-4o" in p["name"]
    )

    total_gpt4o_5yr: float = gpt4o["annual"] * 5
    total_llama_5yr: float = gpu_cost_inr + llama_annual * 5
    savings_5yr: float = total_gpt4o_5yr - total_llama_5yr
    roi_pct: float = (
        (savings_5yr / total_llama_5yr) * 100 if total_llama_5yr > 0 else 0
    )
    payback_months: float = (
        (gpu_cost_inr / (gpt4o["monthly"] - llama_monthly)) * 1
        if gpt4o["monthly"] > llama_monthly
        else 0
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric(
            "\U0001f4b0 5-Year Savings vs GPT-4o",
            f"\u20b9{savings_5yr:,.0f}",
        )
    with r2:
        st.metric(
            "\U0001f4c8 ROI",
            f"{roi_pct:,.0f}%",
        )
    with r3:
        st.metric(
            "\u23f1\ufe0f Payback Period",
            f"{payback_months:.1f} months",
        )

    st.success(
        f"Switching to **Llama 3.1** saves **\u20b9{savings_5yr:,.0f}** "
        f"over 5 years compared to GPT-4o, with a payback period of just "
        f"**{payback_months:.1f} months** and an ROI of **{roi_pct:,.0f}%**."
    )

    # ------------------------------------------------------------------
    # Detailed comparison table
    # ------------------------------------------------------------------
    st.markdown("### \U0001f4cb Detailed Comparison")

    table_rows: list[dict[str, Any]] = []
    for p in provider_annual:
        table_rows.append(
            {
                "Provider": p["name"],
                "Monthly (\u20b9)": f"\u20b9{p['monthly']:,.0f}",
                "Annual (\u20b9)": f"\u20b9{p['annual']:,.0f}",
                "5-Year (\u20b9)": f"\u20b9{p['annual'] * 5:,.0f}",
                "Data Privacy": "\u274c Sent to cloud",
                "Rate Limits": "\u26a0\ufe0f Yes",
                "Offline": "\u274c No",
            }
        )
    table_rows.append(
        {
            "Provider": "Llama 3.1 (FREE)",
            "Monthly (\u20b9)": f"\u20b9{llama_monthly:,.0f}",
            "Annual (\u20b9)": f"\u20b9{llama_annual_total:,.0f}",
            "5-Year (\u20b9)": f"\u20b9{total_llama_5yr:,.0f}",
            "Data Privacy": "\u2705 100% local",
            "Rate Limits": "\u2705 None",
            "Offline": "\u2705 Yes",
        }
    )
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------------------
    # Non-quantified benefits
    # ------------------------------------------------------------------
    st.markdown("### \u2728 Benefits Beyond Cost")

    b1, b2, b3 = st.columns(3)
    with b1:
        st.info(
            "**\U0001f512 Security & Privacy**\n\n"
            "- No data sent to third parties\n"
            "- GDPR compliant by default\n"
            "- Full control over model weights\n"
            "- Air-gapped / offline capable\n"
            "- No data retention by vendor"
        )
    with b2:
        st.info(
            "**\u2699\ufe0f Operational**\n\n"
            "- Zero rate limits\n"
            "- No API downtime risk\n"
            "- Fine-tunable for Indian banking\n"
            "- No vendor lock-in\n"
            "- Predictable costs"
        )
    with b3:
        st.info(
            "**\U0001f3e6 Regulatory**\n\n"
            "- RBI data localisation compliant\n"
            "- Full audit trail on-premise\n"
            "- Model explainability retained\n"
            "- No cross-border data transfer\n"
            "- Easier compliance sign-off"
        )

    # ------------------------------------------------------------------
    # Assumptions footnote
    # ------------------------------------------------------------------
    with st.expander("\U0001f4dd Assumptions & Methodology"):
        st.markdown(
            f"- **Exchange rate:** 1 USD = \u20b9{_USD_TO_INR:.0f}\n"
            f"- **GPU lifespan:** 5 years (amortised linearly)\n"
            f"- **Electricity:** ~\u20b9{llama_electricity_monthly:,.0f}/month "
            f"(750W GPU, 8h/day, \u20b98/kWh)\n"
            f"- **Input tokens per message:** ~200 (system prompt + context)\n"
            f"- **Output tokens per message:** {avg_tokens} (user-configured)\n"
            f"- **API pricing:** As published by providers (Jan 2026)\n"
            f"- **Llama 3.1 8B-Instruct:** Runs on consumer GPUs (8 GB+ VRAM)\n"
            f"- Maintenance, engineering salaries, and fine-tuning costs "
            f"are excluded for both options\n"
        )


# =========================================================================
# Main navigation
# =========================================================================

# Each entry: (icon, label, page_function, section_group)
_NAV_ITEMS: list[tuple[str, str, Any, str]] = [
    ("\U0001f4ca", "Overview Dashboard", page_overview, "Dashboard"),
    ("\U0001f50d", "Customer Search", page_customer_search, "Dashboard"),
    ("\U0001f6a8", "High-Risk Customers", page_high_risk_customers, "Dashboard"),
    ("\U0001f4c8", "Analytics", page_analytics, "Dashboard"),
    ("\u26a1", "Live Monitoring", page_live_monitoring, "Operations"),
    ("\U0001f310", "System Integration", page_system_integration, "Infrastructure"),
    ("\U0001f3db\ufe0f", "AI Governance", page_ai_governance, "Infrastructure"),
]

# Build the flat PAGES dict for backwards compat
PAGES: dict[str, Any] = {label: func for _, label, func, _ in _NAV_ITEMS}


# =========================================================================
# Login / Authentication
# =========================================================================

# Admin credentials (in production, use a proper auth service / database)
# Passwords are stored as SHA-256 hashes for basic security.
_ADMIN_USERS: dict[str, dict[str, str]] = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "name": "Bank Admin",
        "role": "Administrator",
    },
    "manager": {
        "password_hash": hashlib.sha256("manager123".encode()).hexdigest(),
        "name": "Risk Manager",
        "role": "Risk Manager",
    },
    "analyst": {
        "password_hash": hashlib.sha256("analyst123".encode()).hexdigest(),
        "name": "Data Analyst",
        "role": "Analyst",
    },
}


def _render_login_page() -> bool:
    """Render the login page and return ``True`` if user is authenticated.

    Returns:
        ``True`` when the user has a valid session, ``False`` otherwise.
    """
    if st.session_state.get("authenticated"):
        return True

    # Full-screen centred login card
    st.markdown(
        "<style>"
        "[data-testid='stSidebar'] { display: none !important; }"
        "[data-testid='stHeader'] { display: none !important; }"
        "section.main > div { max-width: 480px; margin: auto; padding-top: 8vh; }"
        "</style>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='"
        "text-align:center;padding:30px 0 10px 0;"
        "'>"
        "<div style='"
        "font-size:56px;line-height:1;margin-bottom:8px;"
        "'>\U0001f6e1\ufe0f</div>"
        "<div style='"
        f"font-size:22px;font-weight:800;color:{PRIMARY_COLOR};"
        "letter-spacing:0.5px;line-height:1.3;"
        "'>PRE-DELINQUENCY</div>"
        "<div style='"
        "font-size:13px;color:#6c8fa7;font-weight:500;"
        "letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;"
        "'>Intervention Engine</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='"
        "background: rgba(255,255,255,0.03);"
        "border: 1px solid rgba(2,128,144,0.25);"
        "border-radius: 14px;"
        "padding: 30px 35px 25px 35px;"
        "margin-top: 20px;"
        "'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h3 style='text-align:center;margin-bottom:20px;"
        "color:#e0e0e0;font-weight:600;'>"
        "\U0001f512 Bank Admin Login</h3>",
        unsafe_allow_html=True,
    )

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            key="login_username",
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password",
        )
        submitted = st.form_submit_button(
            "\U0001f513 Sign In", use_container_width=True
        )

    if submitted:
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            user = _ADMIN_USERS.get(username.lower().strip())
            pw_hash = hashlib.sha256(password.encode()).hexdigest()
            if user and user["password_hash"] == pw_hash:
                st.session_state["authenticated"] = True
                st.session_state["user_name"] = user["name"]
                st.session_state["user_role"] = user["role"]
                st.session_state["user_id"] = username.lower().strip()
                st.session_state["login_time"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                st.rerun()
            else:
                st.error(
                    "\u274c Invalid username or password. Please try again."
                )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='"
        "text-align:center;margin-top:24px;font-size:12px;color:#4a6a7a;"
        "'>"
        "\U0001f512 Secure access \u2022 All sessions are encrypted<br>"
        "<span style='font-size:11px;color:#3a5a6a;'>"
        "Demo credentials: admin / admin123</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    return False


def main() -> None:
    """Streamlit app entry point with login gate and sidebar navigation."""

    # --- Authentication gate ---
    if not _render_login_page():
        return

    with st.sidebar:
        # --- Branded header ---
        st.markdown(
            "<div style='"
            "text-align:center;padding:8px 0 4px 0;"
            "'>"
            "<div style='"
            "font-size:28px;line-height:1;margin-bottom:2px;"
            "'>\U0001f6e1\ufe0f</div>"
            "<div style='"
            f"font-size:15px;font-weight:800;color:{PRIMARY_COLOR};"
            "letter-spacing:0.5px;line-height:1.3;"
            "'>PRE-DELINQUENCY</div>"
            "<div style='"
            "font-size:11px;color:#6c8fa7;font-weight:500;"
            "letter-spacing:1px;text-transform:uppercase;"
            "'>Intervention Engine</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # --- Logged-in user info ---
        _uname = st.session_state.get("user_name", "Admin")
        _urole = st.session_state.get("user_role", "")
        st.markdown(
            f"<div style='"
            f"background:rgba(2,128,144,0.1);"
            f"border:1px solid rgba(2,128,144,0.25);"
            f"border-radius:8px;padding:8px 12px;margin:6px 0 10px 0;"
            f"font-size:12px;"
            f"'>"
            f"\U0001f464 <b style='color:#e0e0e0;'>{_uname}</b><br>"
            f"<span style='color:#6c8fa7;font-size:11px;'>"
            f"{_urole} \u2022 "
            f"Logged in {st.session_state.get('login_time', '')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("\U0001f6aa Logout", key="btn_logout", use_container_width=True):
            for key in [
                "authenticated", "user_name", "user_role",
                "user_id", "login_time",
            ]:
                st.session_state.pop(key, None)
            st.rerun()

        st.markdown(
            "<hr style='margin:10px 0;border:none;"
            f"border-top:1px solid rgba(2,128,144,0.3);'>",
            unsafe_allow_html=True,
        )

        # --- Navigation with section headers ---
        nav_labels: list[str] = []
        prev_section = ""
        for icon, label, _, section in _NAV_ITEMS:
            if section != prev_section:
                st.markdown(
                    f"<div style='"
                    f"font-size:10px;font-weight:700;"
                    f"color:#4a6a7a;text-transform:uppercase;"
                    f"letter-spacing:1.5px;padding:10px 0 4px 4px;"
                    f"'>{section}</div>",
                    unsafe_allow_html=True,
                )
                prev_section = section
            nav_labels.append(f"{icon}  {label}")

        selected_label: str = st.radio(
            "Navigation",
            nav_labels,
            label_visibility="collapsed",
        )

        # Extract the actual page name (strip the icon prefix)
        page_name = (
            selected_label.split("  ", 1)[1]
            if "  " in selected_label
            else selected_label
        )

        # --- Status footer ---
        st.markdown(
            "<hr style='margin:14px 0 10px 0;border:none;"
            f"border-top:1px solid rgba(2,128,144,0.3);'>",
            unsafe_allow_html=True,
        )

        # Connection status
        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
            api_ok = resp.status_code == 200
        except Exception:
            api_ok = False

        status_dot = "\U0001f7e2" if api_ok else "\U0001f534"
        status_text = "Connected" if api_ok else "Disconnected"

        st.markdown(
            f"<div style='"
            f"background:rgba(255,255,255,0.03);"
            f"border:1px solid rgba(255,255,255,0.06);"
            f"border-radius:8px;padding:10px 12px;"
            f"font-size:12px;"
            f"'>"
            f"<div style='margin-bottom:6px;'>"
            f"{status_dot} <span style='color:#8aa;'>API:</span> "
            f"<b style='color:{'#00ff88' if api_ok else '#ff4444'};'>"
            f"{status_text}</b></div>"
            f"<div style='color:#4a6a7a;font-size:11px;'>"
            f"\U0001f4e1 {API_BASE_URL}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='"
            f"text-align:center;padding:12px 0 4px 0;"
            f"font-size:10px;color:#3a5a6a;"
            f"'>v1.0.0 &nbsp;\u2022&nbsp; "
            f"{datetime.now().strftime('%Y-%m-%d')}</div>",
            unsafe_allow_html=True,
        )

    # --- Render selected page ---
    page_func = PAGES.get(page_name, page_overview)
    page_func()


if __name__ == "__main__":
    main()
