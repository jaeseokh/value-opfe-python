#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
from src.modeling.phase3.config import Phase3Config
from src.modeling.phase3.data import add_group_ids, basic_clean

def main() -> None:
    cfg = Phase3Config()
    df = pd.read_parquet(cfg.data_path)
    df = add_group_ids(df, lolo_by_field_only=cfg.lolo_by_field_only)
    df = basic_clean(df, y_col=cfg.y_col, n_col=cfg.n_col)

    out = cfg.out_dir / "prepared.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print("Wrote:", out, "shape=", df.shape)

if __name__ == "__main__":
    main()
