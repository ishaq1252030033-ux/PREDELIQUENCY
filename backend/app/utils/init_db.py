"""Initialize the SQLite database and load synthetic data.

Run as:
    python -m backend.app.utils.init_db
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sqlalchemy.exc import IntegrityError
from tqdm.auto import tqdm

from backend.app.models.database import (
    Customer,
    Transaction,
    SessionLocal,
    init_db,
)
from backend.app.utils.logger import get_logger, setup_logging


logger = get_logger("backend.app.utils.init_db")


def _setup_logging() -> None:
    """Initialise application-wide structured logging."""
    setup_logging()


def get_project_paths() -> Tuple[Path, Path]:
    """Return paths to customers.csv and transactions.csv."""
    root = Path(__file__).resolve().parents[3]
    customers_path = root / "ml" / "data" / "customers.csv"
    transactions_path = root / "ml" / "data" / "transactions.csv"
    return customers_path, transactions_path


def load_customers_from_csv(csv_path: Path) -> int:
    """Load customers.csv into the Customer table."""
    if not csv_path.exists():
        logger.error("customers.csv not found at %s", csv_path)
        return 0

    logger.info("Loading customers from %s", csv_path)
    df = pd.read_csv(csv_path)
    required_cols = {"customer_id", "customer_name", "age", "income_bracket", "credit_score"}
    if not required_cols.issubset(df.columns):
        logger.error("customers.csv is missing required columns: %s", required_cols - set(df.columns))
        return 0

    session = SessionLocal()
    try:
        # Clear existing data for idempotent runs
        deleted = session.query(Customer).delete()
        if deleted:
            logger.info("Deleted %d existing customers", deleted)

        customers = []
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Inserting customers"):
            customers.append(
                Customer(
                    customer_id=str(row.customer_id),
                    name=str(row.customer_name),
                    age=int(row.age),
                    credit_score=int(row.credit_score),
                    income_bracket=str(row.income_bracket),
                )
            )

        session.bulk_save_objects(customers)
        session.commit()
        logger.info("Inserted %d customers", len(customers))
        return len(customers)
    except IntegrityError as exc:
        session.rollback()
        logger.error("IntegrityError while inserting customers: %s", exc)
        return 0
    finally:
        session.close()


def load_transactions_from_csv(csv_path: Path) -> int:
    """Load transactions.csv into the Transaction table."""
    if not csv_path.exists():
        logger.error("transactions.csv not found at %s", csv_path)
        return 0

    logger.info("Loading transactions from %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["transaction_date"])
    required_cols = {"customer_id", "transaction_date", "category", "amount_inr", "type", "description"}
    if not required_cols.issubset(df.columns):
        logger.error("transactions.csv is missing required columns: %s", required_cols - set(df.columns))
        return 0

    session = SessionLocal()
    try:
        deleted = session.query(Transaction).delete()
        if deleted:
            logger.info("Deleted %d existing transactions", deleted)

        transactions = []
        for idx, row in tqdm(
            enumerate(df.itertuples(index=False), start=1),
            total=len(df),
            desc="Inserting transactions",
        ):
            tx_id = f"tx_{idx:09d}"
            transactions.append(
                Transaction(
                    transaction_id=tx_id,
                    customer_id=str(row.customer_id),
                    date=row.transaction_date.to_pydatetime(),
                    amount=float(row.amount_inr),
                    transaction_type=str(row.type),
                    category=str(row.category),
                    merchant=str(row.description) if hasattr(row, "description") else str(row.category),
                )
            )

            # Flush in batches to avoid huge memory usage
            if len(transactions) >= 10_000:
                session.bulk_save_objects(transactions)
                session.commit()
                transactions.clear()

        if transactions:
            session.bulk_save_objects(transactions)
            session.commit()

        total = session.query(Transaction).count()
        logger.info("Total transactions in DB: %d", total)
        return total
    except IntegrityError as exc:
        session.rollback()
        logger.error("IntegrityError while inserting transactions: %s", exc)
        return 0
    finally:
        session.close()


def main() -> None:
    _setup_logging()
    logger.info("Initializing database and loading synthetic data")

    # 1. Create tables and indexes
    init_db()
    logger.info("Database tables created (including indexes defined in models).")

    # 2. Determine CSV paths
    customers_path, transactions_path = get_project_paths()
    logger.info("Expected customers.csv at %s", customers_path)
    logger.info("Expected transactions.csv at %s", transactions_path)

    # 3. Load data
    n_customers = load_customers_from_csv(customers_path)
    n_transactions = load_transactions_from_csv(transactions_path)

    # 4. Summary
    logger.info(
        "Database initialization complete. Customers: %d, Transactions: %d",
        n_customers,
        n_transactions,
    )
    print(
        f"Loaded {n_customers} customers and {n_transactions} transactions into the database."
    )


if __name__ == "__main__":
    main()

