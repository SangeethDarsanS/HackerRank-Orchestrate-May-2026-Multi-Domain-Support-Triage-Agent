#!/usr/bin/env python3
"""
Backtesting / evaluation script.

Compares model output against sample_support_tickets.csv to compute
accuracy, precision, recall, F1, escalation rate, and latency metrics.

Usage:
    python evaluate.py
    python evaluate.py --predictions ../support_tickets/output.csv \
                       --ground-truth ../support_tickets/sample_support_tickets.csv
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.config import SUPPORT_TICKETS_DIR
from modules.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate triage agent predictions")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=SUPPORT_TICKETS_DIR / "output.csv",
        help="Path to model output CSV",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=SUPPORT_TICKETS_DIR / "sample_support_tickets.csv",
        help="Path to ground-truth CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.predictions.exists():
        print(f"[error] Predictions file not found: {args.predictions}", file=sys.stderr)
        print("Run:  python main.py --tickets ../support_tickets/support_tickets.csv", file=sys.stderr)
        sys.exit(1)

    if not args.ground_truth.exists():
        print(f"[error] Ground-truth file not found: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    evaluator = Evaluator()
    results = evaluator.evaluate(args.predictions, args.ground_truth)
    evaluator.print_report(results)


if __name__ == "__main__":
    main()
