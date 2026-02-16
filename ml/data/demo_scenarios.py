"""
Generate specific customer scenarios for demonstration.

Creates transactions and customer records for 5 narrative scenarios,
saves them as separate CSVs under ml/data/demo/, and supports loading
into the demo database and feature pipeline.

Run:
  python -m ml.data.demo_scenarios          # generate CSVs only
  python -m ml.data.demo_scenarios --load  # generate and load into DB + features.csv
  python -m ml.data.demo_scenarios --load --merge  # add demo to existing data instead of overwriting
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Project root (parent of ml/data)
SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
ROOT = ML_DIR.parent

# Ensure project root on path for backend imports when running --load
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# Demo customer IDs (must be unique and consistent)
DEMO_IDS = {
    "salary_delay": "DEMO01",
    "lending_spiral": "DEMO02",
    "gradual_decline": "DEMO03",
    "false_alarm": "DEMO04",
    "healthy": "DEMO05",
}

# Snapshot date for feature window (30 days ending this date)
SNAPSHOT_DATE = "2024-06-15"
WINDOW_START = "2024-05-16"

# CSV columns
TX_COLS = ["customer_id", "transaction_date", "category", "amount_inr", "type", "description"]
CUSTOMER_COLS = ["customer_id", "customer_name", "age", "income_bracket", "credit_score"]


def _tx(
    customer_id: str,
    date: str,
    category: str,
    amount_inr: float,
    type_: str,
    description: str,
) -> dict:
    """Build a single transaction row dict for the demo scenario generator."""
    return {
        "customer_id": customer_id,
        "transaction_date": date,
        "category": category,
        "amount_inr": int(amount_inr),
        "type": type_,
        "description": description,
    }


def scenario_01_salary_delay_crisis() -> Tuple[pd.DataFrame, List[dict]]:
    """
    Scenario 1: Salary Delay Crisis
    - Customer: Priya Sharma
    - Normal salary: 1st of month; this month: 8th (7 days late)
    - Savings down 45%; upcoming EMI on 15th
    - Expected risk: 88 (Critical)
    """
    cid = DEMO_IDS["salary_delay"]
    rows = []

    # Establish pattern: salary on 1st in April and May
    rows.append(_tx(cid, "2024-04-01 10:00:00", "salary", 72_000, "credit", "Salary credit"))
    rows.append(_tx(cid, "2024-05-01 10:00:00", "salary", 72_000, "credit", "Salary credit"))

    # Late salary this month (8th = 7 days late)
    rows.append(_tx(cid, "2024-06-08 10:00:00", "salary", 72_000, "credit", "Salary credit (delayed)"))

    # Heavy debits in May (drain balance ~45% before June salary)
    for day in [5, 8, 12, 15, 18, 22]:
        rows.append(_tx(cid, f"2024-05-{day:02d} 11:00:00", "utility_electricity", 4_500, "debit", "BESCOM"))
        rows.append(_tx(cid, f"2024-05-{day:02d} 12:00:00", "discretionary_restaurant", 800, "debit", "Swiggy"))
    rows.append(_tx(cid, "2024-05-10 09:00:00", "utility_water", 600, "debit", "Water bill"))
    rows.append(_tx(cid, "2024-05-14 14:00:00", "discretionary_shopping", 12_000, "debit", "Shopping"))
    rows.append(_tx(cid, "2024-05-20 10:00:00", "atm_withdrawal", 15_000, "debit", "ATM withdrawal"))
    rows.append(_tx(cid, "2024-06-02 10:00:00", "utility_electricity", 4_200, "debit", "BESCOM"))
    rows.append(_tx(cid, "2024-06-05 12:00:00", "discretionary_restaurant", 1_200, "debit", "Restaurant"))
    rows.append(_tx(cid, "2024-06-10 09:00:00", "upi_lending_app", 5_000, "debit", "Loan repayment"))

    customers = [{
        "customer_id": cid,
        "customer_name": "Priya Sharma",
        "age": 34,
        "income_bracket": "medium",
        "credit_score": 648,
    }]
    return pd.DataFrame(rows), customers


def scenario_02_lending_app_spiral() -> Tuple[pd.DataFrame, List[dict]]:
    """
    Scenario 2: Lending App Spiral
    - Customer: Raj Kumar
    - Borrowed from 3 lending apps this month; discretionary -70%
    - Multiple failed payment attempts (very late utilities)
    - Expected risk: 92 (Critical)
    """
    cid = DEMO_IDS["lending_spiral"]
    rows = []

    # Salary on time in April and May
    rows.append(_tx(cid, "2024-04-01 10:00:00", "salary", 55_000, "credit", "Salary credit"))
    rows.append(_tx(cid, "2024-05-01 10:00:00", "salary", 55_000, "credit", "Salary credit"))
    rows.append(_tx(cid, "2024-06-01 10:00:00", "salary", 55_000, "credit", "Salary credit"))

    # 3 lending apps in window (high lending_app_amount and count)
    rows.append(_tx(cid, "2024-05-18 11:00:00", "upi_lending_app", 15_000, "debit", "MoneyTap repayment"))
    rows.append(_tx(cid, "2024-05-25 12:00:00", "upi_lending_app", 12_000, "debit", "Paytm Postpaid"))
    rows.append(_tx(cid, "2024-06-05 10:00:00", "upi_lending_app", 18_000, "debit", "Bajaj Finserv"))

    # Discretionary high in previous period, low in current (→ -70% discretionary_change)
    rows.append(_tx(cid, "2024-04-15 14:00:00", "discretionary_restaurant", 8_000, "debit", "Dining"))
    rows.append(_tx(cid, "2024-04-20 16:00:00", "discretionary_entertainment", 3_000, "debit", "OTT"))
    rows.append(_tx(cid, "2024-04-25 12:00:00", "discretionary_shopping", 10_000, "debit", "Shopping"))
    # Current window: minimal discretionary
    rows.append(_tx(cid, "2024-05-20 12:00:00", "discretionary_restaurant", 600, "debit", "Swiggy"))
    rows.append(_tx(cid, "2024-06-08 14:00:00", "discretionary_restaurant", 500, "debit", "Tea"))

    # Failed payments: utility paid > 15 days after 7th (proxy for failed; window ends 2024-06-15)
    rows.append(_tx(cid, "2024-05-25 10:00:00", "utility_electricity", 5_200, "debit", "BESCOM"))   # 18 days late
    rows.append(_tx(cid, "2024-06-02 10:00:00", "utility_water", 800, "debit", "Water"))            # late
    rows.append(_tx(cid, "2024-05-28 11:00:00", "utility_phone", 1_200, "debit", "Airtel"))        # 21 days late

    rows.append(_tx(cid, "2024-05-10 09:00:00", "atm_withdrawal", 20_000, "debit", "ATM"))
    rows.append(_tx(cid, "2024-06-03 09:00:00", "atm_withdrawal", 10_000, "debit", "ATM"))

    customers = [{
        "customer_id": cid,
        "customer_name": "Raj Kumar",
        "age": 29,
        "income_bracket": "low",
        "credit_score": 512,
    }]
    return pd.DataFrame(rows), customers


def scenario_03_gradual_decline() -> Tuple[pd.DataFrame, List[dict]]:
    """
    Scenario 3: Gradual Decline
    - Customer: Anita Verma
    - Slow savings depletion over 3 months; increasing utility delays
    - Expected risk: 76 (High)
    """
    cid = DEMO_IDS["gradual_decline"]
    rows = []

    rows.append(_tx(cid, "2024-04-01 10:00:00", "salary", 62_000, "credit", "Salary credit"))
    rows.append(_tx(cid, "2024-05-01 10:00:00", "salary", 62_000, "credit", "Salary credit"))
    rows.append(_tx(cid, "2024-06-01 10:00:00", "salary", 62_000, "credit", "Salary credit"))

    # Utility delays (paying 5–10 days late)
    rows.append(_tx(cid, "2024-05-12 10:00:00", "utility_electricity", 4_100, "debit", "BESCOM"))
    rows.append(_tx(cid, "2024-05-18 10:00:00", "utility_water", 550, "debit", "Water"))
    rows.append(_tx(cid, "2024-06-10 10:00:00", "utility_electricity", 4_300, "debit", "BESCOM"))
    rows.append(_tx(cid, "2024-06-14 10:00:00", "utility_phone", 1_100, "debit", "Airtel"))

    # Steady spending; balance slowly depleting (more debits than credits over time)
    for d in [5, 10, 15, 20, 25]:
        rows.append(_tx(cid, f"2024-05-{d:02d} 12:00:00", "discretionary_restaurant", 1_500, "debit", "Dining"))
    rows.append(_tx(cid, "2024-05-08 14:00:00", "discretionary_shopping", 8_000, "debit", "Shopping"))
    rows.append(_tx(cid, "2024-06-05 14:00:00", "discretionary_restaurant", 2_000, "debit", "Restaurant"))
    rows.append(_tx(cid, "2024-05-22 09:00:00", "atm_withdrawal", 18_000, "debit", "ATM"))
    rows.append(_tx(cid, "2024-06-08 09:00:00", "atm_withdrawal", 12_000, "debit", "ATM"))

    customers = [{
        "customer_id": cid,
        "customer_name": "Anita Verma",
        "age": 41,
        "income_bracket": "medium",
        "credit_score": 588,
    }]
    return pd.DataFrame(rows), customers


def scenario_04_false_alarm() -> Tuple[pd.DataFrame, List[dict]]:
    """
    Scenario 4: False Alarm
    - Customer: Rohit Patel
    - One-time large purchase (house down payment); salary stable, savings recovering
    - Expected risk: 45 (Medium, declining)
    """
    cid = DEMO_IDS["false_alarm"]
    rows = []

    rows.append(_tx(cid, "2024-04-01 10:00:00", "salary", 95_000, "credit", "Salary credit"))
    rows.append(_tx(cid, "2024-05-01 10:00:00", "salary", 95_000, "credit", "Salary credit"))
    rows.append(_tx(cid, "2024-06-01 10:00:00", "salary", 95_000, "credit", "Salary credit"))

    # One-time large debit (down payment)
    rows.append(_tx(cid, "2024-05-15 10:00:00", "discretionary_shopping", 450_000, "debit", "House down payment"))

    # Normal spending otherwise; balance recovers after
    rows.append(_tx(cid, "2024-05-08 10:00:00", "utility_electricity", 5_000, "debit", "BESCOM"))
    rows.append(_tx(cid, "2024-05-12 12:00:00", "discretionary_restaurant", 2_500, "debit", "Dining"))
    rows.append(_tx(cid, "2024-06-05 10:00:00", "utility_electricity", 4_800, "debit", "BESCOM"))
    rows.append(_tx(cid, "2024-06-10 12:00:00", "discretionary_restaurant", 1_800, "debit", "Restaurant"))
    rows.append(_tx(cid, "2024-06-12 14:00:00", "discretionary_entertainment", 999, "debit", "OTT"))

    customers = [{
        "customer_id": cid,
        "customer_name": "Rohit Patel",
        "age": 38,
        "income_bracket": "high",
        "credit_score": 742,
    }]
    return pd.DataFrame(rows), customers


def scenario_05_healthy_customer() -> Tuple[pd.DataFrame, List[dict]]:
    """
    Scenario 5: Healthy Customer
    - Customer: Meera Singh
    - Regular salary, stable spending
    - Expected risk: 18 (Low)
    """
    cid = DEMO_IDS["healthy"]
    rows = []

    # Salary on 1st every month
    rows.append(_tx(cid, "2024-04-01 10:00:00", "salary", 85_000, "credit", "Salary credit"))
    rows.append(_tx(cid, "2024-05-01 10:00:00", "salary", 85_000, "credit", "Salary credit"))
    rows.append(_tx(cid, "2024-06-01 10:00:00", "salary", 85_000, "credit", "Salary credit"))

    # Utilities on time (early in month)
    rows.append(_tx(cid, "2024-05-05 10:00:00", "utility_electricity", 3_800, "debit", "BESCOM"))
    rows.append(_tx(cid, "2024-05-06 10:00:00", "utility_water", 450, "debit", "Water"))
    rows.append(_tx(cid, "2024-06-05 10:00:00", "utility_electricity", 3_600, "debit", "BESCOM"))
    rows.append(_tx(cid, "2024-06-06 10:00:00", "utility_phone", 999, "debit", "Airtel"))

    # Stable discretionary
    rows.append(_tx(cid, "2024-05-10 12:00:00", "discretionary_restaurant", 1_200, "debit", "Dining"))
    rows.append(_tx(cid, "2024-05-18 14:00:00", "discretionary_entertainment", 599, "debit", "OTT"))
    rows.append(_tx(cid, "2024-06-10 12:00:00", "discretionary_restaurant", 1_100, "debit", "Dining"))
    rows.append(_tx(cid, "2024-06-14 14:00:00", "discretionary_shopping", 2_500, "debit", "Shopping"))

    rows.append(_tx(cid, "2024-05-15 09:00:00", "atm_withdrawal", 5_000, "debit", "ATM"))

    customers = [{
        "customer_id": cid,
        "customer_name": "Meera Singh",
        "age": 36,
        "income_bracket": "high",
        "credit_score": 798,
    }]
    return pd.DataFrame(rows), customers


def generate_all(
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate all scenario CSVs and return combined transactions and customers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("01_salary_delay_crisis", scenario_01_salary_delay_crisis),
        ("02_lending_app_spiral", scenario_02_lending_app_spiral),
        ("03_gradual_decline", scenario_03_gradual_decline),
        ("04_false_alarm", scenario_04_false_alarm),
        ("05_healthy_customer", scenario_05_healthy_customer),
    ]

    all_tx: List[pd.DataFrame] = []
    all_customers: List[dict] = []

    for name, fn in scenarios:
        df_tx, customers = fn()
        path_tx = output_dir / f"scenario_{name}_transactions.csv"
        df_tx.to_csv(path_tx, index=False)
        logger.info("Wrote %s (%d rows)", path_tx.name, len(df_tx))
        all_tx.append(df_tx)
        all_customers.extend(customers)

    transactions_df = pd.concat(all_tx, ignore_index=True)
    transactions_df = transactions_df.sort_values(["customer_id", "transaction_date"]).reset_index(drop=True)
    customers_df = pd.DataFrame(all_customers)

    path_tx_all = output_dir / "demo_transactions.csv"
    path_cust = output_dir / "demo_customers.csv"
    transactions_df.to_csv(path_tx_all, index=False)
    customers_df.to_csv(path_cust, index=False)
    logger.info("Wrote %s (%d rows) and %s (%d rows)", path_tx_all.name, len(transactions_df), path_cust.name, len(customers_df))

    return transactions_df, customers_df


def load_into_demo_database(
    demo_dir: Path,
    merge_into_main: bool = False,
) -> None:
    """
    Load demo scenario data into the database and feature pipeline.

    - If merge_into_main is False: appends demo customers and transactions to the
      existing customers/transactions CSVs, then runs feature engineering to
      update features.csv with demo customer rows, then runs init_db to load
      everything into the DB (existing data is cleared by init_db).
    - If merge_into_main is True: same but merges demo into main CSVs so that
      ml/data/customers.csv and ml/data/transactions.csv include demo data;
      then re-run feature engineering on the combined data and init_db.

    For a true "demo-only" DB you would use a separate DATABASE_URL; this script
    focuses on making the 5 demo customers available in features.csv and in the
    same DB as the rest of the app.
    """
    demo_customers = demo_dir / "demo_customers.csv"
    demo_transactions = demo_dir / "demo_transactions.csv"
    if not demo_customers.exists() or not demo_transactions.exists():
        raise FileNotFoundError(
            "Run generate_all() first so that demo_customers.csv and demo_transactions.csv exist."
        )

    customers_path = SCRIPT_DIR / "customers.csv"
    transactions_path = SCRIPT_DIR / "transactions.csv"
    processed_dir = SCRIPT_DIR / "processed"
    features_path = processed_dir / "features.csv"

    # 1) Merge or use demo data for feature engineering
    cust_df = pd.read_csv(demo_customers)
    tx_df = pd.read_csv(demo_transactions, parse_dates=["transaction_date"])

    if merge_into_main and customers_path.exists() and transactions_path.exists():
        main_cust = pd.read_csv(customers_path)
        main_tx = pd.read_csv(transactions_path, parse_dates=["transaction_date"])
        # Avoid duplicate customer_id
        existing_ids = set(main_cust["customer_id"])
        cust_df = pd.concat([main_cust, cust_df[~cust_df["customer_id"].isin(existing_ids)]], ignore_index=True)
        tx_df = pd.concat([main_tx, tx_df], ignore_index=True)
        cust_df.to_csv(customers_path, index=False)
        tx_df.to_csv(transactions_path, index=False)
        logger.info("Merged demo data into %s and %s", customers_path, transactions_path)
    # else: use only demo data for this run (we'll only add demo rows to features below)

    # 2) Run feature engineering on (customers + transactions) that include demo
    sys.path.insert(0, str(ROOT))
    from ml.feature_engineering import FeatureEngineer

    engineer = FeatureEngineer(window_days=30)
    engineer.fit(cust_df, tx_df, snapshot_date=SNAPSHOT_DATE)
    computed_features = engineer.transform()

    # 3) Update features.csv: merge demo rows in, or overwrite if we merged full data
    processed_dir.mkdir(parents=True, exist_ok=True)
    demo_ids = set(pd.read_csv(demo_customers)["customer_id"].unique())

    if merge_into_main:
        # Full merge: we have main + demo in cust_df/tx_df; use recomputed features for everyone
        combined = computed_features
    elif features_path.exists():
        # Demo-only: keep existing main features, add/update only the 5 demo rows
        main_features = pd.read_csv(features_path)
        main_features = main_features[~main_features["customer_id"].isin(demo_ids)]
        combined = pd.concat([main_features, computed_features], ignore_index=True)
    else:
        combined = computed_features

    combined.to_csv(features_path, index=False)
    logger.info("Updated %s (%d total rows)", features_path, len(combined))

    # 4) Load into database (init_db reads from ml/data/customers.csv and transactions.csv)
    if not merge_into_main:
        # Write demo-only data to main CSVs so init_db loads it (optional: backup originals first)
        cust_df.to_csv(customers_path, index=False)
        tx_df.to_csv(transactions_path, index=False)
        logger.info("Wrote demo-only customers and transactions to ml/data for init_db.")

    from backend.app.utils.init_db import load_customers_from_csv, load_transactions_from_csv
    from backend.app.models.database import init_db

    init_db()
    n_cust = load_customers_from_csv(customers_path)
    n_tx = load_transactions_from_csv(transactions_path)
    logger.info("Database loaded: %d customers, %d transactions.", n_cust, n_tx)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Generate demo scenario CSVs and optionally load into demo database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "demo",
        help="Directory for scenario CSV files (default: ml/data/demo)",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="After generating, load demo data into database and feature pipeline",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="When using --load: merge demo into main customers/transactions CSVs (default: demo-only overwrite for init_db)",
    )
    args = parser.parse_args()

    transactions_df, customers_df = generate_all(args.output_dir)

    if args.load:
        if not args.merge:
            print("Warning: --load without --merge will overwrite ml/data/customers.csv and transactions.csv with demo-only data.")
        load_into_demo_database(args.output_dir, merge_into_main=args.merge)
        print("Demo scenarios loaded. Use customer_ids: DEMO01, DEMO02, DEMO03, DEMO04, DEMO05 for testing.")
    else:
        print("Demo CSVs written to", args.output_dir)
        print("Run with --load to load into database and update features.csv.")


if __name__ == "__main__":
    main()
