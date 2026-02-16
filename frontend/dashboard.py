"""Streamlit dashboard for the Pre-Delinquency Intervention Engine.

Run with::

    streamlit run frontend/dashboard.py

Environment variables:

    ``API_BASE_URL``
        Backend base URL (default: ``http://localhost:8000/api/v1``).
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

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
</style>
""",
    unsafe_allow_html=True,
)


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
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        intervention_type: str = st.selectbox(
            "Intervention Type",
            ["reminder", "restructuring_offer", "collection_call"],
        )
    with col_b:
        channel: str = st.selectbox("Channel", ["sms", "email", "app"])
    with col_c:
        language: str = st.selectbox(
            "Language",
            ["en", "hi"],
            format_func=lambda x: "English" if x == "en" else "Hindi",
        )
    with col_d:
        variant_choice: str = st.selectbox("A/B Variant", ["auto", "A", "B"])

    customer_name_for_msg = ""
    if (
        not customers_df.empty
        and selected_customer_id in customers_df["customer_id"].values
    ):
        customer_name_for_msg = customers_df.loc[
            customers_df["customer_id"] == selected_customer_id,
            "customer_name",
        ].iloc[0]

    gen_payload: dict[str, Any] = {
        "customer_id": selected_customer_id,
        "customer_name": customer_name_for_msg or selected_customer_id,
        "risk_score": float(risk_data.get("risk_score", 0)) if risk_data else 0.0,
        "risk_level": str(risk_data.get("risk_level", "low")) if risk_data else "low",
        "top_risk_factors": (
            risk_data.get("top_risk_factors", []) if risk_data else []
        ),
        "channel": channel,
        "language": language,
    }
    if variant_choice in ("A", "B"):
        gen_payload["variant"] = variant_choice

    with st.spinner("Generating personalised message..."):
        generated = api_post("/generate-message", gen_payload)

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

    message: str = st.text_area("Message (editable)", value=default_msg, height=100)

    # A/B preview
    with st.expander("Preview A/B variants", expanded=False):
        with st.spinner("Loading variants..."):
            preview = api_post("/generate-message/preview", gen_payload)
        if isinstance(preview, dict):
            pcol1, pcol2 = st.columns(2)
            for col, key in [(pcol1, "A"), (pcol2, "B")]:
                with col:
                    v = preview.get(key, {})
                    st.markdown(f"**Variant {key}** ({v.get('scenario', '')})")
                    if v.get("subject"):
                        st.caption(f"Subject: {v['subject']}")
                    st.text(v.get("body", ""))

    if st.button("Trigger Intervention", type="primary"):
        with st.spinner("Sending intervention..."):
            payload: dict[str, Any] = {
                "customer_id": selected_customer_id,
                "intervention_type": intervention_type,
                "channel": channel,
                "message": message,
            }
            resp = api_post("/intervention/trigger", payload)
        if resp:
            st.success(resp.get("message", "Intervention triggered successfully."))
        else:
            st.error("Failed to trigger intervention. Check backend logs.")


# =========================================================================
# PAGE 3 — High-Risk Customers
# =========================================================================


def page_high_risk_customers() -> None:
    """Listing of all customers above a configurable risk threshold."""
    st.title("High-Risk Customers")

    threshold: int = st.slider(
        "Risk score threshold", min_value=0, max_value=100, value=80, step=1
    )
    limit: int = st.number_input(
        "Max customers", min_value=1, max_value=1000, value=100, step=10
    )
    search_term: str = st.text_input("Search (Customer ID or Name)", "")

    with st.spinner("Loading high-risk customers..."):
        data = api_get(
            "/high-risk-customers",
            {"threshold": threshold, "limit": int(limit)},
        )

    if not isinstance(data, list) or not data:
        st.info("No high-risk customers found for the current threshold.")
        return

    risk_df = pd.DataFrame(data)
    customers_df = load_customers_csv()
    features_df = load_features_csv()

    if not customers_df.empty:
        risk_df = risk_df.merge(
            customers_df[["customer_id", "customer_name"]],
            on="customer_id",
            how="left",
        )
    if not features_df.empty and "avg_balance" in features_df.columns:
        risk_df = risk_df.merge(
            features_df[["customer_id", "avg_balance"]],
            on="customer_id",
            how="left",
        )

    tx_df = load_transactions_csv()
    if not tx_df.empty:
        last_tx = (
            tx_df.groupby("customer_id")["transaction_date"]
            .max()
            .rename("last_transaction_date")
        )
        risk_df = risk_df.merge(last_tx, on="customer_id", how="left")

    if search_term:
        mask = risk_df["customer_id"].astype(str).str.contains(search_term, case=False)
        if "customer_name" in risk_df:
            mask |= (
                risk_df["customer_name"]
                .astype(str)
                .str.contains(search_term, case=False)
            )
        risk_df = risk_df[mask]

    risk_df = risk_df.sort_values("risk_score", ascending=False)

    display_cols = [
        "customer_id",
        "customer_name",
        "risk_score",
        "risk_level",
        "last_transaction_date",
        "avg_balance",
    ]
    display_df = risk_df.reindex(
        columns=[c for c in display_cols if c in risk_df.columns]
    )

    st.subheader(f"High-Risk Customers ({len(display_df)})")

    selected_rows: list[str] = st.multiselect(
        "Select customers to send alerts to",
        options=display_df["customer_id"].tolist(),
        default=[],
    )

    st.dataframe(display_df, use_container_width=True)

    # Download as CSV
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download High-Risk Customers (CSV)",
        data=csv_bytes,
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )

    # PDF report (optional — requires reportlab)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as rl_canvas

        pdf_buffer = io.BytesIO()
        c = rl_canvas.Canvas(pdf_buffer, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, height - 40, "High-Risk Customers Report")
        c.setFont("Helvetica", 10)
        c.drawString(
            40,
            height - 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )
        c.drawString(40, height - 75, f"Threshold: {threshold}")
        c.drawString(40, height - 90, f"Total customers: {len(display_df)}")

        y = height - 120
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "Top 10 Customers")
        y -= 15
        c.setFont("Helvetica", 9)

        for _, row in display_df.head(10).iterrows():
            line = (
                f"{row.get('customer_id')}  "
                f"{row.get('customer_name', '')}  |  "
                f"Score: {row.get('risk_score', 0):.1f}  "
                f"Level: {row.get('risk_level', '')}"
            )
            c.drawString(40, y, line[:110])
            y -= 12
            if y < 60:
                c.showPage()
                y = height - 60

        c.showPage()
        c.save()
        pdf_buffer.seek(0)

        st.download_button(
            "Download Risk Report (PDF)",
            data=pdf_buffer,
            file_name="risk_report.pdf",
            mime="application/pdf",
        )
    except ImportError:
        st.info("PDF report unavailable (install ``reportlab`` to enable).")
    except Exception:
        st.warning("Could not generate PDF report.")

    # Bulk alerts
    if st.button("Send Alerts to Selected", type="primary") and selected_rows:
        progress = st.progress(0, text="Sending alerts...")
        for i, cid in enumerate(selected_rows):
            payload: dict[str, Any] = {
                "customer_id": cid,
                "intervention_type": "reminder",
                "channel": "sms",
                "message": "Your EMI is due soon. Please pay to avoid late fees.",
            }
            api_post("/intervention/trigger", payload)
            progress.progress(
                (i + 1) / len(selected_rows),
                text=f"Sent {i + 1}/{len(selected_rows)}...",
            )
        st.success(f"Alerts sent to {len(selected_rows)} customers.")

    # Detail view
    st.subheader("Customer Details")
    if not display_df.empty:
        detail_id: str = st.selectbox(
            "Select customer to view details",
            options=display_df["customer_id"].tolist(),
        )
        detail = display_df[display_df["customer_id"] == detail_id].iloc[0]
        st.write(detail)
    else:
        st.info("No customers to display.")


# =========================================================================
# PAGE 4 — Analytics
# =========================================================================


def page_analytics() -> None:
    """Model performance, feature importance, and score distributions."""
    st.title("Analytics")

    perf = load_model_performance()
    fi_df = load_feature_importance()

    # --- Model performance ---
    st.subheader("Model Performance")
    if perf:
        test_metrics: dict[str, float] = perf.get("test_metrics") or {}
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.3f}")
        with col2:
            st.metric("Precision", f"{test_metrics.get('precision', 0):.3f}")
        with col3:
            st.metric("Recall", f"{test_metrics.get('recall', 0):.3f}")
        with col4:
            st.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}")
        st.success("Model metrics loaded successfully.")
    else:
        st.warning(
            "Model performance file not found. " "Run the training pipeline first."
        )

    # --- Feature importance ---
    st.subheader("Feature Importance")
    if not fi_df.empty:
        col: Optional[str] = None
        for candidate in (
            "ensemble_importance",
            "xgboost_importance",
            "lightgbm_importance",
        ):
            if candidate in fi_df.columns:
                col = candidate
                break
        if col:
            top_fi = fi_df.sort_values(col, ascending=False).head(15)
            fig_fi = px.bar(
                top_fi,
                x=col,
                y="feature",
                orientation="h",
                title=f"Top Features by {col}",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importance file not found.")

    # --- Confusion matrix & ROC ---
    st.subheader("Confusion Matrix & ROC Curve")
    cm_path = _PROJECT_ROOT / "ml" / "reports" / "confusion_matrix_heatmap.png"
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


# =========================================================================
# Main navigation
# =========================================================================

PAGES: dict[str, Any] = {
    "Overview Dashboard": page_overview,
    "Customer Search": page_customer_search,
    "High-Risk Customers": page_high_risk_customers,
    "Analytics": page_analytics,
}


def main() -> None:
    """Streamlit app entry point with sidebar navigation."""
    with st.sidebar:
        st.header("Navigation")
        page_name: str = st.radio("Go to", list(PAGES.keys()))
        st.markdown("---")
        st.markdown(
            f"<span style='color:{PRIMARY_COLOR};font-weight:bold;'>"
            f"Backend:</span> `{API_BASE_URL}`",
            unsafe_allow_html=True,
        )

    page_func = PAGES[page_name]
    page_func()


if __name__ == "__main__":
    main()
