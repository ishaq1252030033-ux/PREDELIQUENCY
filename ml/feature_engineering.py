"""
Feature engineering for Pre-Delinquency Intervention Engine.
Transforms raw transaction data into ML features over 30-day windows.
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Transform transaction data into ML features.
    Calculates features over a 30-day rolling window for each customer.
    """

    def __init__(self, window_days: int = 30):
        """
        Initialize feature engineer.

        Args:
            window_days: Number of days for rolling window (default: 30)
        """
        self.window_days = window_days
        self.customers_df: Optional[pd.DataFrame] = None
        self.transactions_df: Optional[pd.DataFrame] = None
        self.snapshot_date: Optional[pd.Timestamp] = None
        self._salary_patterns: dict = {}  # customer_id -> usual salary day of month

    def fit(self, customers_df: pd.DataFrame, transactions_df: pd.DataFrame, snapshot_date: Optional[str] = None) -> "FeatureEngineer":
        """
        Fit the feature engineer (learn patterns like usual salary dates).

        Args:
            customers_df: DataFrame with customer profiles
            transactions_df: DataFrame with transactions
            snapshot_date: Date to calculate features for (default: max transaction date)

        Returns:
            self
        """
        self.customers_df = customers_df.copy()
        self.transactions_df = transactions_df.copy()

        # Parse dates if needed
        if not pd.api.types.is_datetime64_any_dtype(self.transactions_df["transaction_date"]):
            self.transactions_df["transaction_date"] = pd.to_datetime(
                self.transactions_df["transaction_date"]
            )

        # Set snapshot date
        if snapshot_date is None:
            self.snapshot_date = self.transactions_df["transaction_date"].max()
        else:
            self.snapshot_date = pd.Timestamp(snapshot_date)

        # Learn salary patterns (usual day of month for salary)
        self._learn_salary_patterns()

        return self

    def _learn_salary_patterns(self) -> None:
        """Learn usual salary day of month for each customer."""
        salary_tx = self.transactions_df[
            self.transactions_df["category"] == "salary"
        ].copy()
        salary_tx["day_of_month"] = salary_tx["transaction_date"].dt.day

        for customer_id in self.customers_df["customer_id"]:
            cust_salaries = salary_tx[salary_tx["customer_id"] == customer_id]
            if len(cust_salaries) > 0:
                # Most common day of month
                mode_day = cust_salaries["day_of_month"].mode()
                if len(mode_day) > 0:
                    self._salary_patterns[customer_id] = int(mode_day[0])
                else:
                    # Fallback: median
                    self._salary_patterns[customer_id] = int(cust_salaries["day_of_month"].median())
            else:
                # Default: last day of month
                self._salary_patterns[customer_id] = 28

    def transform(self) -> pd.DataFrame:
        """
        Transform transactions into features for each customer.

        Returns:
            DataFrame with one row per customer and feature columns
        """
        if self.customers_df is None or self.transactions_df is None:
            raise ValueError("Must call fit() before transform()")

        features_list = []
        window_start = self.snapshot_date - pd.Timedelta(days=self.window_days)

        for _, customer_row in self.customers_df.iterrows():
            customer_id = customer_row["customer_id"]

            # Get transactions in window
            cust_tx = self.transactions_df[
                (self.transactions_df["customer_id"] == customer_id) &
                (self.transactions_df["transaction_date"] >= window_start) &
                (self.transactions_df["transaction_date"] <= self.snapshot_date)
            ].copy().sort_values("transaction_date")

            # Calculate all features
            features = self._calculate_features(customer_id, cust_tx, window_start)
            features["customer_id"] = customer_id
            features_list.append(features)

        features_df = pd.DataFrame(features_list)
        return features_df

    def _calculate_features(
        self,
        customer_id: str,
        window_tx: pd.DataFrame,
        window_start: pd.Timestamp,
    ) -> dict:
        """Calculate all features for one customer."""
        features = {}

        if len(window_tx) == 0:
            # Return zeros/NaNs for empty window
            return self._empty_features()

        # Calculate running balance
        balance_history = self._calculate_balance_history(customer_id, window_tx, window_start)

        # 1. days_since_salary
        last_salary = window_tx[window_tx["category"] == "salary"]["transaction_date"]
        if len(last_salary) > 0:
            days_since = (self.snapshot_date - last_salary.max()).days
        else:
            # Look before window
            all_salary = self.transactions_df[
                (self.transactions_df["customer_id"] == customer_id) &
                (self.transactions_df["category"] == "salary") &
                (self.transactions_df["transaction_date"] < window_start)
            ]["transaction_date"]
            if len(all_salary) > 0:
                days_since = (self.snapshot_date - all_salary.max()).days
            else:
                days_since = 999  # No salary found
        features["days_since_salary"] = days_since

        # 2. salary_delay
        usual_day = self._salary_patterns.get(customer_id, 28)
        expected_salary_date = pd.Timestamp(self.snapshot_date.year, self.snapshot_date.month, usual_day)
        if expected_salary_date > self.snapshot_date:
            expected_salary_date = pd.Timestamp(
                self.snapshot_date.year,
                self.snapshot_date.month - 1 if self.snapshot_date.month > 1 else 12,
                usual_day
            )
        last_salary_in_window = window_tx[window_tx["category"] == "salary"]["transaction_date"]
        if len(last_salary_in_window) > 0:
            actual_date = last_salary_in_window.max()
            delay = (actual_date - expected_salary_date).days
        else:
            delay = (self.snapshot_date - expected_salary_date).days
        features["salary_delay"] = max(0, delay)  # Only positive delays

        # 3. avg_balance
        if len(balance_history) > 0:
            features["avg_balance"] = balance_history["balance"].mean()
        else:
            features["avg_balance"] = 0.0

        # 4. balance_trend (% change)
        if len(balance_history) >= 2:
            start_bal = balance_history.iloc[0]["balance"]
            end_bal = balance_history.iloc[-1]["balance"]
            if start_bal > 0:
                features["balance_trend"] = ((end_bal - start_bal) / start_bal) * 100
            else:
                features["balance_trend"] = 0.0 if end_bal == 0 else 100.0
        else:
            features["balance_trend"] = 0.0

        # 5. days_to_zero (estimated days until balance hits zero)
        if len(balance_history) >= 2:
            recent_balances = balance_history.tail(7)["balance"].values
            if len(recent_balances) >= 2:
                daily_change = np.mean(np.diff(recent_balances))
                current_balance = balance_history.iloc[-1]["balance"]
                if daily_change < 0 and current_balance > 0:
                    features["days_to_zero"] = abs(current_balance / daily_change)
                else:
                    features["days_to_zero"] = 999.0  # Not depleting
            else:
                features["days_to_zero"] = 999.0
        else:
            features["days_to_zero"] = 999.0

        # 6. total_spending
        debits = window_tx[window_tx["type"] == "debit"]
        features["total_spending"] = debits["amount_inr"].sum()

        # 7. discretionary_spending
        discretionary = window_tx[
            window_tx["category"].str.contains("discretionary", case=False, na=False)
        ]
        features["discretionary_spending"] = discretionary["amount_inr"].sum()

        # 8. discretionary_change (% change vs previous month)
        current_month_start = pd.Timestamp(self.snapshot_date.year, self.snapshot_date.month, 1)
        prev_month_start = current_month_start - pd.DateOffset(months=1)
        prev_month_end = current_month_start - pd.Timedelta(days=1)

        current_disc = window_tx[
            (window_tx["category"].str.contains("discretionary", case=False, na=False)) &
            (window_tx["transaction_date"] >= current_month_start)
        ]["amount_inr"].sum()

        prev_disc = self.transactions_df[
            (self.transactions_df["customer_id"] == customer_id) &
            (self.transactions_df["category"].str.contains("discretionary", case=False, na=False)) &
            (self.transactions_df["transaction_date"] >= prev_month_start) &
            (self.transactions_df["transaction_date"] <= prev_month_end)
        ]["amount_inr"].sum()

        if prev_disc > 0:
            features["discretionary_change"] = ((current_disc - prev_disc) / prev_disc) * 100
        else:
            features["discretionary_change"] = 0.0 if current_disc == 0 else 100.0

        # 9. essential_spending
        essential = window_tx[
            window_tx["category"].str.contains("utility", case=False, na=False)
        ]
        features["essential_spending"] = essential["amount_inr"].sum()

        # 10. spending_velocity (daily average)
        if len(window_tx) > 0:
            days_in_window = (self.snapshot_date - window_start).days + 1
            features["spending_velocity"] = features["total_spending"] / max(days_in_window, 1)
        else:
            features["spending_velocity"] = 0.0

        # 11. lending_app_transactions (count)
        lending = window_tx[window_tx["category"] == "upi_lending_app"]
        features["lending_app_transactions"] = len(lending)

        # 12. lending_app_amount
        features["lending_app_amount"] = lending["amount_inr"].sum()

        # 13. atm_withdrawal_count
        atm = window_tx[window_tx["category"] == "atm_withdrawal"]
        features["atm_withdrawal_count"] = len(atm)

        # 14. atm_amount
        features["atm_amount"] = atm["amount_inr"].sum()

        # 15. utility_payment_delay (days late from usual)
        utility_tx = window_tx[
            window_tx["category"].str.contains("utility", case=False, na=False)
        ]
        if len(utility_tx) > 0:
            # Assume utilities usually paid around 5-10th of month
            usual_day = 7
            delays = []
            for _, tx in utility_tx.iterrows():
                expected_date = pd.Timestamp(
                    tx["transaction_date"].year,
                    tx["transaction_date"].month,
                    usual_day
                )
                delay = (tx["transaction_date"] - expected_date).days
                if delay > 0:
                    delays.append(delay)
            features["utility_payment_delay"] = np.mean(delays) if delays else 0.0
        else:
            features["utility_payment_delay"] = 0.0

        # 16. failed_payments_count (proxy: very late utility payments > 15 days)
        if len(utility_tx) > 0:
            failed = 0
            for _, tx in utility_tx.iterrows():
                expected_date = pd.Timestamp(
                    tx["transaction_date"].year,
                    tx["transaction_date"].month,
                    7
                )
                if (tx["transaction_date"] - expected_date).days > 15:
                    failed += 1
            features["failed_payments_count"] = failed
        else:
            features["failed_payments_count"] = 0

        # 17. payment_timing_variance (std dev of days of month for utility payments)
        if len(utility_tx) > 0:
            days_of_month = utility_tx["transaction_date"].dt.day
            var = days_of_month.std()
            # Guard against NaN when there is only one payment in the window
            features["payment_timing_variance"] = float(var) if not np.isnan(var) else 0.0
        else:
            features["payment_timing_variance"] = 0.0

        # 18. savings_depletion_rate (% decrease)
        savings_tx = window_tx[
            window_tx["category"].isin(["savings_deposit", "savings_withdrawal"])
        ]
        if len(balance_history) >= 2:
            start_bal = balance_history.iloc[0]["balance"]
            end_bal = balance_history.iloc[-1]["balance"]
            if start_bal > 0:
                depletion = ((start_bal - end_bal) / start_bal) * 100
                features["savings_depletion_rate"] = max(0, depletion)
            else:
                features["savings_depletion_rate"] = 0.0
        else:
            features["savings_depletion_rate"] = 0.0

        # 19. transaction_diversity (unique merchant categories)
        features["transaction_diversity"] = window_tx["category"].nunique()

        # 20. cash_hoarding_score (ATM withdrawals / total spending)
        if features["total_spending"] > 0:
            features["cash_hoarding_score"] = features["atm_amount"] / features["total_spending"]
        else:
            features["cash_hoarding_score"] = 0.0

        return features

    def _calculate_balance_history(
        self,
        customer_id: str,
        window_tx: pd.DataFrame,
        window_start: pd.Timestamp,
    ) -> pd.DataFrame:
        """Calculate running balance over time."""
        # Get starting balance (balance before window)
        pre_window_tx = self.transactions_df[
            (self.transactions_df["customer_id"] == customer_id) &
            (self.transactions_df["transaction_date"] < window_start)
        ].sort_values("transaction_date")

        # Estimate starting balance (simplified: assume 50k starting + net of pre-window transactions)
        starting_balance = 50_000
        if len(pre_window_tx) > 0:
            credits = pre_window_tx[pre_window_tx["type"] == "credit"]["amount_inr"].sum()
            debits = pre_window_tx[pre_window_tx["type"] == "debit"]["amount_inr"].sum()
            starting_balance = max(0, starting_balance + credits - debits)

        # Calculate balance over window
        balance = starting_balance
        balance_records = [{"date": window_start, "balance": balance}]

        for _, tx in window_tx.iterrows():
            if tx["type"] == "credit":
                balance += tx["amount_inr"]
            else:
                balance = max(0, balance - tx["amount_inr"])
            balance_records.append({"date": tx["transaction_date"], "balance": balance})

        return pd.DataFrame(balance_records)

    def _empty_features(self) -> dict:
        """Return feature dict with zeros/NaNs for empty transaction window."""
        return {
            "days_since_salary": 999,
            "salary_delay": 0.0,
            "avg_balance": 0.0,
            "balance_trend": 0.0,
            "days_to_zero": 999.0,
            "total_spending": 0.0,
            "discretionary_spending": 0.0,
            "discretionary_change": 0.0,
            "essential_spending": 0.0,
            "spending_velocity": 0.0,
            "lending_app_transactions": 0,
            "lending_app_amount": 0.0,
            "atm_withdrawal_count": 0,
            "atm_amount": 0.0,
            "utility_payment_delay": 0.0,
            "failed_payments_count": 0,
            "payment_timing_variance": 0.0,
            "savings_depletion_rate": 0.0,
            "transaction_diversity": 0,
            "cash_hoarding_score": 0.0,
        }

    def fit_transform(
        self,
        customers_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        snapshot_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(customers_df, transactions_df, snapshot_date).transform()


def main():
    """Main function to run feature engineering."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate ML features from transaction data")
    parser.add_argument(
        "--customers",
        type=Path,
        default=Path(__file__).parent / "data" / "customers.csv",
        help="Path to customers.csv",
    )
    parser.add_argument(
        "--transactions",
        type=Path,
        default=Path(__file__).parent / "data" / "transactions.csv",
        help="Path to transactions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "processed" / "features.csv",
        help="Output path for features.csv",
    )
    parser.add_argument(
        "--snapshot-date",
        type=str,
        default=None,
        help="Snapshot date for feature calculation (YYYY-MM-DD, default: max transaction date)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Rolling window size in days (default: 30)",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading customers from {args.customers}...")
    customers_df = pd.read_csv(args.customers)

    print(f"Loading transactions from {args.transactions}...")
    transactions_df = pd.read_csv(args.transactions, parse_dates=["transaction_date"])

    # Generate features
    print("Generating features...")
    engineer = FeatureEngineer(window_days=args.window_days)
    features_df = engineer.fit_transform(customers_df, transactions_df, args.snapshot_date)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.output, index=False)
    print(f"Features saved to {args.output}")
    print(f"  Shape: {features_df.shape}")
    print(f"  Columns: {', '.join(features_df.columns)}")


if __name__ == "__main__":
    main()
