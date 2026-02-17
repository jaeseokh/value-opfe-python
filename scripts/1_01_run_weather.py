#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Make project importable if user runs from repo root (recommended)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.fetch_weather import run_all_fields, run_one_field


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1: build Daymet weather features")
    ap.add_argument("--ffy-id", default=None, help="Run only one ffy_id (e.g., 10_1_2022)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    if args.ffy_id:
        run_one_field(args.ffy_id, overwrite=args.overwrite)
    else:
        run_all_fields(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
