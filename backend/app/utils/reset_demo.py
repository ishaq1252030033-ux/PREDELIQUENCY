"""Reset demo: clear DB, load demo scenarios, run predictions, save to DB, print summary.

Run as:
    python -m backend.app.utils.reset_demo

Use for quick reset before demonstrations.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root (backend/app/utils -> 3 parents up)
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models.database import (
    Customer,
    Intervention,
    RiskAssessment,
    SessionLocal,
    Transaction,
    init_db,
)
from backend.app.services.risk_predictor import RiskPredictor
from backend.app.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

DEMO_CUSTOMER_IDS = ["DEMO01", "DEMO02", "DEMO03", "DEMO04", "DEMO05"]
DEMO_DATA_DIR = ROOT / "ml" / "data" / "demo"


def clear_database(session) -> dict:
    """Delete all rows from risk_assessments, interventions, transactions, customers. Returns counts deleted."""
    counts = {}
    # Delete in dependency order (children before parents)
    for model, key in [
        (RiskAssessment, "risk_assessments"),
        (Intervention, "interventions"),
        (Transaction, "transactions"),
        (Customer, "customers"),
    ]:
        n = session.query(model).delete()
        counts[key] = n
    session.commit()
    return counts


def load_demo_scenarios() -> tuple[int, int]:
    """Generate demo CSVs and load into DB + features.csv. Returns (n_customers, n_transactions)."""
    from ml.data.demo_scenarios import generate_all, load_into_demo_database

    generate_all(DEMO_DATA_DIR)
    # Load demo-only (overwrites ml/data/customers.csv and transactions.csv)
    load_into_demo_database(DEMO_DATA_DIR, merge_into_main=False)
    # Counts are in DB now; read from session
    session = SessionLocal()
    try:
        n_cust = session.query(Customer).count()
        n_tx = session.query(Transaction).count()
        return n_cust, n_tx
    finally:
        session.close()


def run_predictions_and_save() -> int:
    """Run risk prediction for all demo customers and persist to risk_assessments. Returns count saved."""
    predictor = RiskPredictor()
    scores = predictor.batch_predict(DEMO_CUSTOMER_IDS)

    session = SessionLocal()
    try:
        for risk in scores:
            session.add(
                RiskAssessment(
                    customer_id=risk.customer_id,
                    risk_score=risk.risk_score,
                    risk_level=risk.risk_level.value,
                    prediction_date=risk.prediction_date,
                    top_risk_factors=risk.top_risk_factors or None,
                    model_version=None,
                )
            )
        session.commit()
        return len(scores)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def print_summary(deleted: dict, n_customers: int, n_transactions: int, n_assessments: int) -> None:
    """Print summary of demo data loaded and predictions."""
    session = SessionLocal()
    try:
        rows = (
            session.query(Customer.customer_id, Customer.name, RiskAssessment.risk_score, RiskAssessment.risk_level)
            .join(RiskAssessment, Customer.customer_id == RiskAssessment.customer_id)
            .order_by(RiskAssessment.risk_score.desc())
            .all()
        )
    finally:
        session.close()

    print()
    print("=" * 60)
    print("DEMO RESET COMPLETE")
    print("=" * 60)
    print("Database cleared (deleted):")
    for table, count in deleted.items():
        print(f"  {table}: {count}")
    print()
    print("Demo data loaded:")
    print(f"  Customers:    {n_customers}")
    print(f"  Transactions: {n_transactions}")
    print(f"  Predictions:  {n_assessments} (saved to risk_assessments)")
    print()
    print("Risk scores (demo customers):")
    print("-" * 60)
    for customer_id, name, score, level in rows:
        print(f"  {customer_id}  {name:<18}  score={score:5.1f}  level={level}")
    print("-" * 60)
    print("Ready for demonstration.")
    print()


def main() -> None:
    setup_logging()

    session = SessionLocal()
    try:
        init_db()
        deleted = clear_database(session)
    finally:
        session.close()

    logger.info("Loading demo scenarios...")
    n_customers, n_transactions = load_demo_scenarios()

    logger.info("Running predictions for demo customers...")
    n_assessments = run_predictions_and_save()

    print_summary(deleted, n_customers, n_transactions, n_assessments)


if __name__ == "__main__":
    main()
