#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.etl.build_spatial_features import (
    Phase2Config,          # keep as-is unless you also rename the class
    build_all_fields,
    build_one_field_features,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build obs-level spatial features: exp + topo rasters + SSURGO (area-weighted) + weather"
    )
    ap.add_argument("--ffy-id", default=None, help="Run only one field-year, e.g. 10_1_2022")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if outputs exist")
    args = ap.parse_args()

    cfg = Phase2Config(overwrite=args.overwrite)

    if args.ffy_id:
        build_one_field_features(args.ffy_id, cfg=cfg)
    else:
        build_all_fields(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
