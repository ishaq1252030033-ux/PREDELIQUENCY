"""Tests for the FeatureEngineer in ml/feature_engineering.py."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_customers_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"customer_id": "C1"},
            {"customer_id": "C2"},
            {"customer_id": "C3"},
        ]
    )


@pytest.fixture
def sample_transactions_df() -> pd.DataFrame:
    """Create a small, controlled transaction set to test feature logic."""
    rows = [
        # C1: salary normally on 25th, last month delayed to 30th
        {"customer_id": "C1", "transaction_date": "2024-05-25", "category": "salary", "amount_inr": 50_000, "type": "credit"},
        {"customer_id": "C1", "transaction_date": "2024-06-30", "category": "salary", "amount_inr": 50_000, "type": "credit"},
        # C1: utilities and discretionary in June
        {"customer_id": "C1", "transaction_date": "2024-06-05", "category": "utility_electricity", "amount_inr": 3_000, "type": "debit"},
        {"customer_id": "C1", "transaction_date": "2024-06-10", "category": "discretionary_restaurant", "amount_inr": 2_000, "type": "debit"},
        {"customer_id": "C1", "transaction_date": "2024-06-12", "category": "discretionary_entertainment", "amount_inr": 1_000, "type": "debit"},
        # C2: only one debit transaction (edge case)
        {"customer_id": "C2", "transaction_date": "2024-06-15", "category": "discretionary_restaurant", "amount_inr": 1_500, "type": "debit"},
        # C3: no transactions in the window (will be covered by fixture)
    ]
    df = pd.DataFrame(rows)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


@pytest.fixture
def feature_engineer(sample_customers_df, sample_transactions_df) -> FeatureEngineer:
    fe = FeatureEngineer(window_days=30)
    # Use snapshot at end of June so we can reason about expectations
    snapshot = datetime(2024, 6, 30)
    fe.fit(sample_customers_df, sample_transactions_df, snapshot_date=snapshot.isoformat())
    return fe


def test_salary_delay_calculation(feature_engineer: FeatureEngineer):
    """Salary delay should reflect difference from usual salary day."""
    features_df = feature_engineer.transform()
    c1 = features_df[features_df["customer_id"] == "C1"].iloc[0]

    # C1 had salary on 25th (May) and 30th (June), so usual_day ~ 25.
    # With snapshot 2024-06-30 and last salary 2024-06-30:
    # expected_salary_date is around 25th, so delay ~ 5 days.
    assert 4 <= c1["salary_delay"] <= 7


def test_balance_trend_calculation(feature_engineer: FeatureEngineer):
    """Balance trend should be computed as percentage change over the window."""
    features_df = feature_engineer.transform()
    c1 = features_df[features_df["customer_id"] == "C1"].iloc[0]

    # At least ensure it's a finite float and not NaN
    assert np.isfinite(c1["balance_trend"])


def test_spending_categorization(feature_engineer: FeatureEngineer):
    """Discretionary vs essential spending should sum appropriate categories."""
    features_df = feature_engineer.transform()
    c1 = features_df[features_df["customer_id"] == "C1"].iloc[0]

    # For C1 in June: discretionary = 2000 + 1000, essential (utility) = 3000
    assert c1["discretionary_spending"] == 3_000
    assert c1["essential_spending"] == 3_000
    # total_spending is total debits in window
    assert c1["total_spending"] >= c1["discretionary_spending"] + c1["essential_spending"]


def test_customer_with_no_transactions():
    """Customer with no transactions should get default/zeroed features."""
    customers = pd.DataFrame([{"customer_id": "C_NO_TX"}])
    transactions = pd.DataFrame(
        [
            {"customer_id": "C_OTHER", "transaction_date": "2024-06-01", "category": "salary", "amount_inr": 10_000, "type": "credit"}
        ]
    )
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    fe = FeatureEngineer(window_days=30).fit(customers, transactions, snapshot_date="2024-06-30")
    features_df = fe.transform()
    row = features_df.iloc[0]

    # Should match _empty_features defaults
    assert row["days_since_salary"] == 999
    assert row["total_spending"] == 0.0
    assert row["discretionary_spending"] == 0.0
    assert row["essential_spending"] == 0.0


def test_customer_with_single_transaction():
    """Single-transaction customer should be handled without errors."""
    customers = pd.DataFrame([{"customer_id": "C_SINGLE"}])
    tx = pd.DataFrame(
        [
            {
                "customer_id": "C_SINGLE",
                "transaction_date": "2024-06-15",
                "category": "discretionary_restaurant",
                "amount_inr": 1_000,
                "type": "debit",
            }
        ]
    )
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])

    fe = FeatureEngineer(window_days=30).fit(customers, tx, snapshot_date="2024-06-30")
    features_df = fe.transform()
    row = features_df.iloc[0]

    # total_spending should equal the only debit
    assert row["total_spending"] == 1_000
    # days_to_zero should be positive and finite
    assert row["days_to_zero"] > 0


def test_no_null_values_in_output(feature_engineer: FeatureEngineer):
    """Output feature frame should not contain nulls."""
    features_df = feature_engineer.transform()
    assert not features_df.isna().any().any()


def test_feature_value_ranges(feature_engineer: FeatureEngineer):
    """Simple sanity checks on ranges for key features."""
    features_df = feature_engineer.transform()

    assert (features_df["days_since_salary"] >= 0).all()
    assert (features_df["total_spending"] >= 0).all()
    assert (features_df["discretionary_spending"] >= 0).all()
    assert (features_df["essential_spending"] >= 0).all()

