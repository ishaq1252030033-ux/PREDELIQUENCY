"""Automatic intervention system using FREE Llama 3.1.

Runs as a background service that periodically scans for high-risk
customers, generates personalised messages via the local AI, and
dispatches interventions through the appropriate channel.

Usage (standalone)::

    python -m backend.app.services.auto_intervention

Or import and start programmatically::

    from backend.app.services.auto_intervention import AutoInterventionService
    service = AutoInterventionService(risk_threshold=80)
    service.start_scheduler()

Requirements:
    - Ollama running locally with ``llama3.1`` pulled.
    - The ML model trained and feature matrix available.
    - ``schedule`` package installed (``pip install schedule``).
"""

from __future__ import annotations

import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import schedule
from sqlalchemy import select, func as sa_func

# Ensure project root is importable
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.models.database import (
    Intervention,
    RiskAssessment,
    SessionLocal,
)
from backend.app.models.schemas import RiskScore
from backend.app.services.ai_message_generator import AIMessageGenerator
from backend.app.services.risk_predictor import RiskPredictor
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: load customer IDs from the feature matrix
# ---------------------------------------------------------------------------


def _load_all_customer_ids(predictor: RiskPredictor) -> List[str]:
    """Return every customer ID present in the feature matrix.

    Args:
        predictor: An initialised :class:`RiskPredictor`.

    Returns:
        Sorted list of customer ID strings.
    """
    return sorted(predictor.features_df["customer_id"].unique().tolist())


# ---------------------------------------------------------------------------
# AutoInterventionService
# ---------------------------------------------------------------------------


class AutoInterventionService:
    """Automatically generate and send interventions to high-risk customers.

    Uses **FREE** Llama 3.1 (via Ollama) for message generation and
    the trained ML model for risk scoring.  All inference is local —
    no API keys, no cloud calls.

    Attributes:
        ai_generator: The AI message generator instance.
        risk_predictor: The ML risk predictor instance.
        risk_threshold: Minimum risk score to trigger an intervention.
        interventions_sent: Running counter of interventions dispatched
            during this session.
        dry_run: If ``True``, messages are generated but **not** sent
            or persisted.  Useful for testing.
    """

    def __init__(
        self,
        risk_threshold: int = 80,
        dry_run: bool = False,
    ) -> None:
        """Initialise the service.

        Args:
            risk_threshold: Minimum risk score (0-100) that triggers
                an intervention.
            dry_run: When ``True`` messages are printed but not
                dispatched or logged to the database.
        """
        self.risk_threshold: int = risk_threshold
        self.dry_run: bool = dry_run
        self.interventions_sent: int = 0

        logger.info("Initialising AI generator (Llama 3.1) ...")
        self.ai_generator = AIMessageGenerator()

        logger.info("Initialising risk predictor ...")
        self.risk_predictor = RiskPredictor()

        logger.info(
            "AutoInterventionService ready — threshold=%d, dry_run=%s",
            self.risk_threshold,
            self.dry_run,
        )

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #

    def check_and_intervene(self) -> int:
        """Scan all customers, identify high-risk ones, and intervene.

        Returns:
            Number of interventions triggered in this run.
        """
        logger.info(
            "Checking for customers with risk > %d ...",
            self.risk_threshold,
        )

        customer_ids = _load_all_customer_ids(self.risk_predictor)
        logger.info("Scoring %d customers ...", len(customer_ids))

        # Batch predict
        scores: List[RiskScore] = self.risk_predictor.batch_predict(
            customer_ids
        )

        # Filter high-risk
        high_risk: List[RiskScore] = [
            s for s in scores if s.risk_score >= self.risk_threshold
        ]
        logger.info(
            "Found %d high-risk customers (threshold=%d)",
            len(high_risk),
            self.risk_threshold,
        )

        count = 0
        for risk in high_risk:
            # Skip if already contacted today
            if self._already_intervened_today(risk.customer_id):
                logger.debug(
                    "Skipping %s — already contacted today",
                    risk.customer_id,
                )
                continue

            try:
                self._process_customer(risk)
                count += 1
            except Exception as exc:
                logger.error(
                    "Failed to process %s: %s",
                    risk.customer_id,
                    exc,
                )

            # Rate-limit so we don't overwhelm downstream services
            time.sleep(2)

        self.interventions_sent += count
        logger.info(
            "Run complete — %d interventions this run, %d total",
            count,
            self.interventions_sent,
        )
        return count

    # ------------------------------------------------------------------ #
    # Per-customer processing
    # ------------------------------------------------------------------ #

    def _process_customer(self, risk: RiskScore) -> None:
        """Generate AI messages and dispatch for one customer.

        Args:
            risk: The computed :class:`RiskScore` for the customer.
        """
        channel = self._select_channel(risk.risk_score)

        # Resolve customer name from the feature matrix or fall back
        customer_name = self._resolve_customer_name(risk.customer_id)

        logger.info(
            "Generating AI messages for %s (score=%.1f, channel=%s)",
            risk.customer_id,
            risk.risk_score,
            channel,
        )

        messages: Dict[str, Any] = (
            self.ai_generator.generate_intervention_messages(
                customer_name=customer_name,
                risk_score=risk.risk_score,
                risk_factors=risk.top_risk_factors[:3],
                language="en",
            )
        )

        # Extract the message for the selected channel
        if channel == "email":
            email_data = messages.get("email", {})
            display_msg = (
                f"Subject: {email_data.get('subject', '')}\n"
                f"{email_data.get('body', '')}"
            )
        else:
            display_msg = str(messages.get(channel, ""))

        # Dispatch
        if channel == "sms":
            self._send_sms(risk.customer_id, messages.get("sms", ""))
        elif channel == "email":
            email_data = messages.get("email", {})
            self._send_email(
                risk.customer_id,
                email_data.get("subject", ""),
                email_data.get("body", ""),
            )
        elif channel == "whatsapp":
            self._send_whatsapp(
                risk.customer_id, messages.get("whatsapp", "")
            )
        else:
            self._send_app_push(
                risk.customer_id, messages.get("app", "")
            )

        # Persist to database
        if not self.dry_run:
            self._log_intervention(
                customer_id=risk.customer_id,
                channel=channel,
                message=display_msg[:1024],
                risk_score=risk.risk_score,
            )

        logger.info(
            "Intervention sent to %s (%s) via %s",
            risk.customer_id,
            customer_name,
            channel,
        )

    # ------------------------------------------------------------------ #
    # Channel selection
    # ------------------------------------------------------------------ #

    @staticmethod
    def _select_channel(risk_score: float) -> str:
        """Select communication channel based on risk severity.

        Args:
            risk_score: Numeric risk score (0-100).

        Returns:
            One of ``"sms"``, ``"whatsapp"``, ``"email"``, or ``"app"``.
        """
        if risk_score >= 90:
            return "sms"
        if risk_score >= 80:
            return "whatsapp"
        if risk_score >= 70:
            return "email"
        return "app"

    # ------------------------------------------------------------------ #
    # De-duplication
    # ------------------------------------------------------------------ #

    @staticmethod
    def _already_intervened_today(customer_id: str) -> bool:
        """Return ``True`` if an intervention was already sent today.

        Args:
            customer_id: Customer identifier to check.

        Returns:
            ``True`` when an intervention record exists for *today*.
        """
        try:
            db = SessionLocal()
            today_start = datetime.combine(date.today(), datetime.min.time())
            stmt = (
                select(sa_func.count())
                .select_from(Intervention)
                .where(
                    Intervention.customer_id == customer_id,
                    Intervention.created_at >= today_start,
                )
            )
            count: int = db.execute(stmt).scalar() or 0
            db.close()
            return count > 0
        except Exception as exc:
            logger.debug(
                "Could not check intervention history for %s: %s",
                customer_id,
                exc,
            )
            return False

    # ------------------------------------------------------------------ #
    # Name resolution
    # ------------------------------------------------------------------ #

    def _resolve_customer_name(self, customer_id: str) -> str:
        """Look up a customer's display name.

        Falls back to the customer ID if the CSV is unavailable.

        Args:
            customer_id: Customer identifier.

        Returns:
            Customer name string.
        """
        try:
            import pandas as pd

            csv_path = _PROJECT_ROOT / "ml" / "data" / "customers.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                match = df.loc[
                    df["customer_id"] == customer_id, "customer_name"
                ]
                if not match.empty:
                    return str(match.iloc[0])
        except Exception:
            pass
        return customer_id

    # ------------------------------------------------------------------ #
    # Dispatch stubs (wire to real services in production)
    # ------------------------------------------------------------------ #

    def _send_sms(self, customer_id: str, message: str) -> None:
        """Send SMS via gateway (stub — logs only).

        Args:
            customer_id: Target customer.
            message: SMS text.
        """
        logger.info(
            "[SMS] -> %s: %s",
            customer_id,
            message[:80],
        )

    def _send_email(
        self, customer_id: str, subject: str, body: str
    ) -> None:
        """Send email via SMTP (stub — logs only).

        Args:
            customer_id: Target customer.
            subject: Email subject line.
            body: Email body.
        """
        logger.info(
            "[EMAIL] -> %s | Subject: %s",
            customer_id,
            subject,
        )

    def _send_whatsapp(self, customer_id: str, message: str) -> None:
        """Send WhatsApp via Business API (stub — logs only).

        Args:
            customer_id: Target customer.
            message: WhatsApp text.
        """
        logger.info(
            "[WHATSAPP] -> %s: %s",
            customer_id,
            message[:80],
        )

    def _send_app_push(self, customer_id: str, message: str) -> None:
        """Send app push notification (stub — logs only).

        Args:
            customer_id: Target customer.
            message: Notification text.
        """
        logger.info(
            "[APP PUSH] -> %s: %s",
            customer_id,
            message[:80],
        )

    # ------------------------------------------------------------------ #
    # Database logging
    # ------------------------------------------------------------------ #

    @staticmethod
    def _log_intervention(
        customer_id: str,
        channel: str,
        message: str,
        risk_score: float,
    ) -> None:
        """Persist an intervention record to the database.

        Args:
            customer_id: Customer who was contacted.
            channel: Channel used (sms, email, whatsapp, app).
            message: The message that was sent.
            risk_score: Risk score at the time of intervention.
        """
        try:
            db = SessionLocal()
            intervention = Intervention(
                customer_id=customer_id,
                intervention_type="auto_ai_intervention",
                channel=channel,
                message=message[:1024],
                sent_at=datetime.now(),
                status="sent",
            )
            db.add(intervention)
            db.commit()
            db.close()
            logger.debug(
                "Intervention logged for %s (score=%.1f)",
                customer_id,
                risk_score,
            )
        except Exception as exc:
            logger.error(
                "Failed to log intervention for %s: %s",
                customer_id,
                exc,
            )

    # ------------------------------------------------------------------ #
    # Scheduler
    # ------------------------------------------------------------------ #

    def start_scheduler(self) -> None:
        """Start the hourly intervention check loop.

        Schedules ``check_and_intervene`` every hour from 9 AM to 6 PM
        (business hours).  The loop runs indefinitely; send SIGINT /
        Ctrl+C to stop.
        """
        for hour in range(9, 19):
            schedule.every().day.at(f"{hour:02d}:00").do(
                self.check_and_intervene
            )

        logger.info("Auto-intervention scheduler started")
        logger.info("Schedule: every hour, 09:00-18:00")
        logger.info("Risk threshold: %d", self.risk_threshold)
        logger.info("Dry run: %s", self.dry_run)
        logger.info("Press Ctrl+C to stop.\n")

        # Also run once immediately so we don't wait until the next hour
        self.check_and_intervene()

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info(
                "Scheduler stopped. Total interventions sent: %d",
                self.interventions_sent,
            )

    def run_once(self) -> int:
        """Run a single intervention scan (useful for cron / testing).

        Returns:
            Number of interventions triggered.
        """
        return self.check_and_intervene()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-intervention service using FREE Llama 3.1",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=80,
        help="Minimum risk score to trigger intervention (default: 80)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate messages but do NOT send or log them",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan and exit (no scheduler loop)",
    )
    args = parser.parse_args()

    service = AutoInterventionService(
        risk_threshold=args.threshold,
        dry_run=args.dry_run,
    )

    if args.once:
        n = service.run_once()
        print(f"\nDone — {n} interventions triggered.")
    else:
        service.start_scheduler()
