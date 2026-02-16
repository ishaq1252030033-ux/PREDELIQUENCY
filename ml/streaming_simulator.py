"""Simulate a real-time transaction stream and push events to the API.

Features:
1. Reads transactions from ml/data/transactions.csv
2. Sends one transaction every 1–2 seconds to /api/v1/stream/transaction
3. Each event triggers risk recalculation on the backend
4. Designed to work with WebSocket updates from FastAPI for real-time dashboards

Run:
    python -m ml.streaming_simulator
"""

from __future__ import annotations

import argparse
import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd


DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_TX_PATH = Path("ml/data/transactions.csv")


async def stream_transactions(
    api_base: str,
    csv_path: Path,
    interval_min: float = 1.0,
    interval_max: float = 2.0,
    customer_filter: Optional[str] = None,
) -> None:
    """Stream transactions from CSV to the API."""
    if not csv_path.exists():
        raise FileNotFoundError(f"transactions.csv not found at {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["transaction_date"])
    if customer_filter:
        df = df[df["customer_id"] == customer_filter]

    df = df.sort_values("transaction_date")
    if df.empty:
        print("No transactions to stream.")
        return

    async with httpx.AsyncClient(base_url=api_base, timeout=10) as client:
        print(f"Starting stream to {api_base}/api/v1/stream/transaction ...")
        try:
            for _, row in df.iterrows():
                payload = {
                    "customer_id": str(row["customer_id"]),
                    "date": row["transaction_date"].isoformat(),
                    "amount": float(row["amount_inr"]),
                    "transaction_type": str(row["type"]),
                    "category": str(row["category"]),
                    "merchant": str(row.get("description") or row.get("category")),
                }
                try:
                    resp = await client.post("/api/v1/stream/transaction", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    risk = data.get("risk", {})
                    print(
                        f"[{datetime.utcnow().isoformat()}] "
                        f"tx -> customer={payload['customer_id']} "
                        f"amount={payload['amount']} "
                        f"category={payload['category']} "
                        f"| risk={risk.get('risk_score')} "
                        f"level={risk.get('risk_level')}"
                    )
                except httpx.HTTPError as exc:
                    print(f"HTTP error sending transaction: {exc}")

                await asyncio.sleep(random.uniform(interval_min, interval_max))
        except KeyboardInterrupt:
            print("Streaming cancelled by user.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a real-time transaction stream to the API.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"API base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_TX_PATH,
        help="Path to transactions.csv (default: ml/data/transactions.csv)",
    )
    parser.add_argument(
        "--interval-min",
        type=float,
        default=1.0,
        help="Minimum interval between transactions in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--interval-max",
        type=float,
        default=2.0,
        help="Maximum interval between transactions in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--customer-id",
        type=str,
        default=None,
        help="If provided, stream only this customer's transactions.",
    )
    args = parser.parse_args()

    asyncio.run(
        stream_transactions(
            api_base=args.api_base,
            csv_path=args.csv,
            interval_min=args.interval_min,
            interval_max=args.interval_max,
            customer_filter=args.customer_id,
        )
    )


if __name__ == "__main__":
    main()

