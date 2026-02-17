# scripts/1_04_run_build_model_ready.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.build_model_ready import run, run_one_field


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="rebuild even if outputs exist")
    ap.add_argument("--ffy-id", type=str, default=None, help="run only one ffy_id (optional)")
    args = ap.parse_args()

    if args.ffy_id:
        run_one_field(args.ffy_id, overwrite=args.overwrite)
    else:
        run(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
