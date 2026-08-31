"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Emergency SOS anomaly detection pipeline")
    parser.add_argument("--rows", type=int, default=100_000, help="synthetic event count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args.output, args.rows, args.seed), indent=2, default=str))


if __name__ == "__main__":
    main()
