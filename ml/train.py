"""
Training script for the Pre-Delinquency risk prediction model.
Run from project root: python -m ml.train
"""

import argparse
from pathlib import Path


def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train Pre-Delinquency risk model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed training data",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("ml/models"),
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline via ml/train_model.py."""
    args = parse_args()
    print(f"Training with data from {args.data_dir}, saving to {args.model_dir}")
    print(f"Epochs: {args.epochs}")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    print("Delegating to ml.train_model — run: python -m ml.train_model")


if __name__ == "__main__":
    main()
