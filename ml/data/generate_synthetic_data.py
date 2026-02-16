"""
Generate realistic synthetic banking transaction data for 10,000 customers over 6 months.
Jan 2024 - Jun 2024, amounts in INR. Outputs: customers.csv, transactions.csv, labels.csv.

Usage:
  python ml/data/generate_synthetic_data.py                    # full 10k customers
  python ml/data/generate_synthetic_data.py -n 1000            # quick run
  python ml/data/generate_synthetic_data.py -o data/sample    # save to data/sample
"""

import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Constants (set to 100 for quick test, 10_000 for full run)
N_CUSTOMERS = 10_000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 6, 30)
DEFAULT_RATE_TARGET = 0.175  # 17.5% (within 15-20%)
OUTPUT_DIR = Path(__file__).resolve().parent

# Indian first names (mix of common names across regions)
FIRST_NAMES = [
    "Aarav", "Aditya", "Amit", "Anil", "Arjun", "Rahul", "Raj", "Ramesh", "Suresh", "Vikram",
    "Karthik", "Kumar", "Manoj", "Priya", "Anita", "Kavita", "Neha", "Pooja", "Sunita", "Deepak",
    "Ravi", "Sanjay", "Vijay", "Ashok", "Gopal", "Lakshmi", "Meera", "Divya", "Shreya", "Preeti",
    "Sandeep", "Naveen", "Rohit", "Akash", "Vivek", "Kiran", "Swati", "Pallavi", "Anjali", "Ritu",
    "Mohit", "Gaurav", "Nitin", "Sachin", "Rajat", "Abhishek", "Varun", "Harsh", "Yash", "Arun",
]

# Indian surnames
SURNAMES = [
    "Sharma", "Singh", "Kumar", "Patel", "Reddy", "Nair", "Iyer", "Gupta", "Joshi", "Mehta",
    "Shah", "Desai", "Rao", "Pillai", "Menon", "Nambiar", "Pandey", "Verma", "Jain", "Agarwal",
    "Kapoor", "Malhotra", "Chopra", "Sethi", "Khanna", "Bhatia", "Saxena", "Dubey", "Tiwari", "Mishra",
]

# Income brackets (monthly, INR) and approximate salary
INCOME_BRACKETS = [
    ("low", 25_000, 45_000),
    ("medium", 50_000, 120_000),
    ("high", 130_000, 350_000),
]

# Lending app UPI IDs (common BNPL / instant loan apps in India)
LENDING_APPS = [
    "MoneyTap", "PayTM Postpaid", "PhonePe Credit", "Lazypay", "ZestMoney",
    "Simpl", "Amazon Pay Later", "Flipkart Pay Later", "MobiKwik Postpaid",
]

# Transaction category config: (category, is_debit, amount_range_inr, monthly_freq_range)
TX_CATEGORIES = {
    "salary": ("credit", (1, 1), None),  # amount from income
    "utility_electricity": ("debit", (1, 1), (2_000, 8_000)),
    "utility_water": ("debit", (1, 1), (300, 1_500)),
    "utility_phone": ("debit", (1, 1), (500, 2_500)),
    "discretionary_restaurant": ("debit", (4, 15), (200, 3_000)),
    "discretionary_entertainment": ("debit", (2, 8), (500, 5_000)),
    "discretionary_shopping": ("debit", (2, 10), (1_000, 15_000)),
    "atm_withdrawal": ("debit", (8, 25), (2_000, 25_000)),
    "upi_lending_app": ("debit", (0, 4), (1_000, 25_000)),
    "savings_deposit": ("credit", (1, 4), (2_000, 20_000)),
    "savings_withdrawal": ("debit", (0, 3), (1_000, 50_000)),
}


def generate_customers(n: int) -> pd.DataFrame:
    """Generate customer profiles with Indian names, age, income_bracket, credit_score."""
    customer_ids = [f"C{i:06d}" for i in range(1, n + 1)]
    names = [
        f"{random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}"
        for _ in range(n)
    ]
    ages = np.random.randint(22, 58, size=n)
    bracket_choices = ["low", "medium", "high"]
    weights = [0.35, 0.45, 0.20]
    income_brackets = random.choices(bracket_choices, weights=weights, k=n)
    # Credit score 300-900, slight bias by income
    credit_scores = np.clip(
        np.random.normal(650, 120, n) + np.where(
            np.array(income_brackets) == "low", -30,
            np.where(np.array(income_brackets) == "high", 40, 0)
        ),
        300,
        900,
    ).astype(int)

    return pd.DataFrame({
        "customer_id": customer_ids,
        "customer_name": names,
        "age": ages,
        "income_bracket": income_brackets,
        "credit_score": credit_scores,
    })


def get_salary_amount(income_bracket: str) -> int:
    """Return monthly salary in INR for bracket."""
    low, high = next(
        (lo, hi) for name, lo, hi in INCOME_BRACKETS if name == income_bracket
    )
    return random.randint(low, high)


def month_end_or_first(year: int, month: int, prefer_last: bool = True) -> datetime:
    """Last working day of month or 1st."""
    if prefer_last:
        d = datetime(year, month, 1) + timedelta(days=32)
        d = d.replace(day=1) - timedelta(days=1)
        while d.weekday() >= 5:  # Sat=5, Sun=6
            d -= timedelta(days=1)
        return d.replace(hour=9, minute=0, second=0)
    return datetime(year, month, 1, 9, 0, 0)


def generate_transactions_for_customer(
    customer_id: str,
    income_bracket: str,
    is_defaulter: bool,
    salary_delay_months: list,
    savings_depletion_months: list,
    lending_app_boost: bool,
    discretionary_cut_months: list,
) -> list[dict]:
    """Generate 6 months of transactions for one customer."""
    rows = []
    salary_amount = get_salary_amount(income_bracket)
    date = START_DATE
    running_savings = random.randint(30_000, 200_000)  # starting balance

    while date <= END_DATE:
        y, m = date.year, date.month

        # ---- Salary (with optional delay for defaulters) ----
        salary_date = month_end_or_first(y, m)
        delay_days = 0
        if is_defaulter and (y, m) in salary_delay_months:
            delay_days = random.randint(5, 14)
        salary_date += timedelta(days=delay_days)
        if salary_date <= END_DATE:
            rows.append({
                "customer_id": customer_id,
                "transaction_date": salary_date,
                "category": "salary",
                "amount_inr": salary_amount,
                "type": "credit",
                "description": "Salary credit",
            })
            running_savings += salary_amount

        # ---- Utilities (monthly) ----
        for cat, (_, _, amt_range) in TX_CATEGORIES.items():
            if not cat.startswith("utility_") or amt_range is None:
                continue
            amt = random.randint(amt_range[0], amt_range[1])
            pay_date = datetime(y, m, random.randint(5, 15), 10, 0, 0)
            if pay_date <= END_DATE:
                rows.append({
                    "customer_id": customer_id,
                    "transaction_date": pay_date,
                    "category": cat,
                    "amount_inr": amt,
                    "type": "debit",
                    "description": f"{cat.replace('_', ' ').title()} payment",
                })
                running_savings -= amt

        # ---- Discretionary (multiple per month) ----
        for cat in ["discretionary_restaurant", "discretionary_entertainment", "discretionary_shopping"]:
            _, (freq_lo, freq_hi), amt_range = TX_CATEGORIES[cat]
            n_tx = random.randint(freq_lo, freq_hi)
            if is_defaulter and (y, m) in discretionary_cut_months:
                n_tx = max(0, n_tx - random.randint(2, 4))
            for _ in range(n_tx):
                day = random.randint(1, 28)
                try:
                    tx_date = datetime(y, m, day, random.randint(8, 22), random.randint(0, 59), 0)
                except ValueError:
                    continue
                if tx_date <= END_DATE:
                    amt = random.randint(amt_range[0], amt_range[1])
                    rows.append({
                        "customer_id": customer_id,
                        "transaction_date": tx_date,
                        "category": cat,
                        "amount_inr": amt,
                        "type": "debit",
                        "description": cat.replace("_", " ").title(),
                    })
                    running_savings -= amt

        # ---- ATM ----
        n_atm = random.randint(8, 25)
        for _ in range(n_atm):
            day = random.randint(1, 28)
            try:
                tx_date = datetime(y, m, day, random.randint(6, 23), 0, 0)
            except ValueError:
                continue
            if tx_date <= END_DATE:
                amt = random.randint(2_000, min(25_000, salary_amount // 2))
                rows.append({
                    "customer_id": customer_id,
                    "transaction_date": tx_date,
                    "category": "atm_withdrawal",
                    "amount_inr": amt,
                    "type": "debit",
                    "description": "ATM withdrawal",
                })
                running_savings -= amt

        # ---- UPI to lending apps ----
        n_lending = random.randint(0, 4)
        if lending_app_boost and is_defaulter:
            n_lending += random.randint(3, 8)
        for _ in range(n_lending):
            day = random.randint(1, 28)
            try:
                tx_date = datetime(y, m, day, random.randint(9, 21), random.randint(0, 59), 0)
            except ValueError:
                continue
            if tx_date <= END_DATE:
                app = random.choice(LENDING_APPS)
                amt = random.randint(1_000, min(25_000, salary_amount // 3))
                rows.append({
                    "customer_id": customer_id,
                    "transaction_date": tx_date,
                    "category": "upi_lending_app",
                    "amount_inr": amt,
                    "type": "debit",
                    "description": f"UPI to {app}",
                })
                running_savings -= amt

        # ---- Savings movements ----
        n_dep = random.randint(1, 4)
        for _ in range(n_dep):
            day = random.randint(2, 27)
            try:
                tx_date = datetime(y, m, day, 11, 0, 0)
            except ValueError:
                continue
            if tx_date <= END_DATE:
                amt = random.randint(2_000, 20_000)
                rows.append({
                    "customer_id": customer_id,
                    "transaction_date": tx_date,
                    "category": "savings_deposit",
                    "amount_inr": amt,
                    "type": "credit",
                    "description": "Savings deposit",
                })
                running_savings += amt

        n_wd = random.randint(0, 3)
        if is_defaulter and (y, m) in savings_depletion_months:
            n_wd += random.randint(2, 5)
        for _ in range(n_wd):
            day = random.randint(1, 28)
            try:
                tx_date = datetime(y, m, day, 14, 0, 0)
            except ValueError:
                continue
            if tx_date <= END_DATE and running_savings > 5_000:
                amt = random.randint(1_000, min(50_000, running_savings - 2_000))
                rows.append({
                    "customer_id": customer_id,
                    "transaction_date": tx_date,
                    "category": "savings_withdrawal",
                    "amount_inr": amt,
                    "type": "debit",
                    "description": "Savings withdrawal",
                })
                running_savings -= amt

        # Next month
        if m == 12:
            date = datetime(y + 1, 1, 1)
        else:
            date = datetime(y, m + 1, 1)

    return rows


def assign_defaulter_risk_factors(
    n_customers: int,
    defaulter_indices: set,
) -> dict:
    """
    For each defaulter, assign which months have salary delay, savings depletion,
    lending app boost, and discretionary cut. Non-defaulters get empty/False.
    """
    out = {
        "salary_delay_months": {},
        "savings_depletion_months": {},
        "lending_app_boost": {},
        "discretionary_cut_months": {},
    }
    months = [(2024, m) for m in range(1, 7)]

    for i in range(n_customers):
        cid = f"C{i + 1:06d}"
        if i not in defaulter_indices:
            out["salary_delay_months"][cid] = []
            out["savings_depletion_months"][cid] = []
            out["lending_app_boost"][cid] = False
            out["discretionary_cut_months"][cid] = []
            continue

        # Defaulters: inject risk factors (concentrate in later months)
        n_delay = random.randint(1, 3)
        out["salary_delay_months"][cid] = list(random.sample(months[-3:], min(n_delay, 3)))
        n_deplete = random.randint(2, 4)
        out["savings_depletion_months"][cid] = list(random.sample(months[-4:], min(n_deplete, 4)))
        out["lending_app_boost"][cid] = True
        n_cut = random.randint(2, 5)
        out["discretionary_cut_months"][cid] = list(random.sample(months, min(n_cut, 6)))

    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic banking data")
    parser.add_argument("-n", "--customers", type=int, default=N_CUSTOMERS, help="Number of customers")
    parser.add_argument("-o", "--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory for CSVs")
    args = parser.parse_args()
    n_customers = args.customers
    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_customers} customers...")
    customers_df = generate_customers(n_customers)

    # Decide who defaults (15-20%); slightly bias by income and credit
    n_default = int(n_customers * DEFAULT_RATE_TARGET)
    n_default = max(1, min(n_default, n_customers - 1))
    # Higher default probability for low income / lower credit
    weights = np.zeros(n_customers)
    for i in range(n_customers):
        b = customers_df.iloc[i]["income_bracket"]
        cs = customers_df.iloc[i]["credit_score"]
        w = 1.0
        if b == "low":
            w *= 1.8
        elif b == "high":
            w *= 0.4
        if cs < 550:
            w *= 1.5
        elif cs > 750:
            w *= 0.6
        weights[i] = w
    weights /= weights.sum()
    defaulter_indices = set(
        np.random.choice(n_customers, size=n_default, replace=False, p=weights)
    )

    risk_factors = assign_defaulter_risk_factors(n_customers, defaulter_indices)

    print("Generating transactions...")
    all_tx = []
    for i in range(n_customers):
        cid = customers_df.iloc[i]["customer_id"]
        income_bracket = customers_df.iloc[i]["income_bracket"]
        is_def = i in defaulter_indices
        tx_rows = generate_transactions_for_customer(
            cid,
            income_bracket,
            is_def,
            risk_factors["salary_delay_months"][cid],
            risk_factors["savings_depletion_months"][cid],
            risk_factors["lending_app_boost"][cid],
            risk_factors["discretionary_cut_months"][cid],
        )
        all_tx.extend(tx_rows)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_customers} customers...")

    transactions_df = pd.DataFrame(all_tx)
    transactions_df = transactions_df.sort_values(["customer_id", "transaction_date"]).reset_index(drop=True)

    # Labels: default = 1 if customer missed payment in next 30 days (we simulated as defaulter)
    labels_df = pd.DataFrame({
        "customer_id": customers_df["customer_id"],
        "default": [1 if i in defaulter_indices else 0 for i in range(n_customers)],
    })

    # Save
    customers_path = output_dir / "customers.csv"
    transactions_path = output_dir / "transactions.csv"
    labels_path = output_dir / "labels.csv"

    customers_df.to_csv(customers_path, index=False)
    transactions_df.to_csv(transactions_path, index=False)
    labels_df.to_csv(labels_path, index=False)

    default_rate = labels_df["default"].mean()
    print(f"\nDone. Default rate: {default_rate:.2%} (target 15-20%)")
    print(f"  {customers_path}")
    print(f"  {transactions_path}")
    print(f"  {labels_path}")
    print(f"  Customers: {len(customers_df)}, Transactions: {len(transactions_df)}")


if __name__ == "__main__":
    main()
