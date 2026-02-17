#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.fetch_topo import run_all_fields, run_one_field, TopoPhase1Config


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 1 TOPO: build dem/slope/aspect/twi rasters + parquet manifest"
    )
    ap.add_argument("--ffy-id", default=None, help="Run only one field-year, e.g. 10_1_2022")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if outputs exist")
    args = ap.parse_args()

    if args.ffy_id:
        cfg = TopoPhase1Config(overwrite=args.overwrite)
        run_one_field(args.ffy_id, cfg=cfg)
    else:
        run_all_fields(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
