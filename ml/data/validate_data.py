"""
Validate generated banking CSV data: load files, run quality checks, print summary report.
Usage: python ml/data/validate_data.py [--data-dir PATH]
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

# Expected date range (from generate_synthetic_data.py)
EXPECTED_START = "2024-01-01"
EXPECTED_END = "2024-06-30"

# Critical columns that must have no missing values
CRITICAL_COLUMNS = {
    "customers": ["customer_id", "age", "income_bracket", "credit_score"],
    "transactions": ["customer_id", "transaction_date", "category", "amount_inr", "type"],
    "labels": ["customer_id", "default"],
}


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load customers, transactions, and labels from CSV. Raises FileNotFoundError if missing."""
    data_dir = Path(data_dir)
    customers_path = data_dir / "customers.csv"
    transactions_path = data_dir / "transactions.csv"
    labels_path = data_dir / "labels.csv"

    missing = [p for p in (customers_path, transactions_path, labels_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {[str(p) for p in missing]}")

    customers = pd.read_csv(customers_path)
    transactions = pd.read_csv(transactions_path, parse_dates=["transaction_date"])
    labels = pd.read_csv(labels_path)

    return customers, transactions, labels


def check_missing(df: pd.DataFrame, name: str, critical: list[str]) -> list[str]:
    """Return list of warning messages for missing values in critical columns."""
    warnings = []
    for col in critical:
        if col not in df.columns:
            warnings.append(f"[{name}] Missing column: {col}")
            continue
        n = df[col].isna().sum()
        if n > 0:
            warnings.append(f"[{name}] {n} missing value(s) in '{col}'")
    return warnings


def check_date_range(transactions: pd.DataFrame) -> list[str]:
    """Check that transaction dates fall within expected range."""
    warnings = []
    if "transaction_date" not in transactions.columns:
        return warnings
    min_d = transactions["transaction_date"].min()
    max_d = transactions["transaction_date"].max()
    if pd.isna(min_d) or pd.isna(max_d):
        warnings.append("[transactions] transaction_date has missing values")
        return warnings
    min_s = pd.Timestamp(min_d).strftime("%Y-%m-%d")
    max_s = pd.Timestamp(max_d).strftime("%Y-%m-%d")
    if min_s < EXPECTED_START:
        warnings.append(f"[transactions] Date range starts before expected: {min_s} < {EXPECTED_START}")
    if max_s > EXPECTED_END:
        warnings.append(f"[transactions] Date range ends after expected: {max_s} > {EXPECTED_END}")
    return warnings


def check_amounts_positive(transactions: pd.DataFrame) -> list[str]:
    """Check that all transaction amounts are positive."""
    warnings = []
    if "amount_inr" not in transactions.columns:
        return warnings
    non_positive = (transactions["amount_inr"] <= 0).sum()
    if non_positive > 0:
        warnings.append(f"[transactions] {non_positive} transaction(s) with amount_inr <= 0")
    return warnings


def check_customer_id_consistency(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
) -> list[str]:
    """Check that customer_id is consistent across all files."""
    warnings = []
    cust_ids = set(customers["customer_id"].dropna().astype(str))
    tx_ids = set(transactions["customer_id"].dropna().astype(str))
    label_ids = set(labels["customer_id"].dropna().astype(str))

    if not cust_ids:
        warnings.append("[consistency] customers has no customer_id values")
        return warnings

    only_in_tx = tx_ids - cust_ids
    only_in_labels = label_ids - cust_ids
    only_in_customers = cust_ids - label_ids
    missing_labels = cust_ids - label_ids

    if only_in_tx:
        warnings.append(f"[consistency] {len(only_in_tx)} customer_id(s) in transactions not in customers: e.g. {list(only_in_tx)[:3]}")
    if only_in_labels:
        warnings.append(f"[consistency] {len(only_in_labels)} customer_id(s) in labels not in customers: e.g. {list(only_in_labels)[:3]}")
    if missing_labels:
        warnings.append(f"[consistency] {len(missing_labels)} customer_id(s) in customers have no label (expected 0)")

    return warnings


def run_quality_checks(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
) -> list[str]:
    """Run all data quality checks; return list of warning messages."""
    all_warnings = []

    for name, cols in CRITICAL_COLUMNS.items():
        df = {"customers": customers, "transactions": transactions, "labels": labels}[name]
        all_warnings.extend(check_missing(df, name, cols))

    all_warnings.extend(check_date_range(transactions))
    all_warnings.extend(check_amounts_positive(transactions))
    all_warnings.extend(check_customer_id_consistency(customers, transactions, labels))

    return all_warnings


def print_summary_report(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    """Print summary statistics and sample stats."""
    n_customers = len(customers)
    n_transactions = len(transactions)
    default_count = labels["default"].sum()
    default_rate = default_count / n_customers if n_customers else 0

    if "transaction_date" in transactions.columns:
        min_date = transactions["transaction_date"].min()
        max_date = transactions["transaction_date"].max()
        date_range = f"{pd.Timestamp(min_date).strftime('%Y-%m-%d')} to {pd.Timestamp(max_date).strftime('%Y-%m-%d')}"
    else:
        date_range = "N/A"

    print("=" * 60)
    print("DATA VALIDATION SUMMARY REPORT")
    print("=" * 60)
    print(f"  Total customers:     {n_customers:,}")
    print(f"  Total transactions: {n_transactions:,}")
    print(f"  Default count:      {default_count:,}")
    print(f"  Default rate:       {default_rate:.2%}")
    print(f"  Date range:         {date_range}")
    print()

    print("--- Sample statistics ---")
    if n_customers:
        tx_per_customer = transactions.groupby("customer_id").size()
        print(f"  Transactions per customer: min={tx_per_customer.min()}, max={tx_per_customer.max()}, mean={tx_per_customer.mean():.1f}")
    if "amount_inr" in transactions.columns:
        print(f"  amount_inr: min={transactions['amount_inr'].min():,.0f}, max={transactions['amount_inr'].max():,.0f}, mean={transactions['amount_inr'].mean():,.0f} INR")
    if "age" in customers.columns:
        print(f"  age: min={customers['age'].min()}, max={customers['age'].max()}, mean={customers['age'].mean():.1f}")
    if "credit_score" in customers.columns:
        print(f"  credit_score: min={customers['credit_score'].min()}, max={customers['credit_score'].max()}, mean={customers['credit_score'].mean():.1f}")
    if "income_bracket" in customers.columns:
        print(f"  income_bracket distribution:\n{customers['income_bracket'].value_counts().to_string()}")
    if "category" in transactions.columns:
        print(f"  Transaction categories: {transactions['category'].nunique()} unique")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated banking CSV data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing customers.csv, transactions.csv, labels.csv",
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    try:
        customers, transactions, labels = load_data(data_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    warnings = run_quality_checks(customers, transactions, labels)

    if warnings:
        print("DATA QUALITY WARNINGS:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
        print(file=sys.stderr)

    print_summary_report(customers, transactions, labels)

    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
