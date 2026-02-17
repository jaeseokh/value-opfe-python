# scripts/1_03_run_ssurgo.py
from __future__ import annotations

import argparse

from src.etl.fetch_ssurgo import run_all_fields, run_one_field, SsurgoPhase1Config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffy-id", type=str, default=None, help="Run one field only")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist")
    ap.add_argument("--vars", type=str, default=None, help="CSV list of SSURGO vars (optional)")
    args = ap.parse_args()

    if args.vars:
        vars_tuple = tuple(v.strip() for v in args.vars.split(",") if v.strip())
        cfg = SsurgoPhase1Config(overwrite=args.overwrite, vars=vars_tuple)
    else:
        cfg = SsurgoPhase1Config(overwrite=args.overwrite)

    if args.ffy_id:
        run_one_field(args.ffy_id, cfg=cfg)
    else:
        run_all_fields(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
