#!/usr/bin/env python
from __future__ import annotations

import pandas as pd

from src.modeling.phase3.config import Phase3Config
from src.modeling.phase3.reporting import save_df


def main() -> None:
    cfg = Phase3Config()
    p_ranks = cfg.out_dir / "step3_focus_feature_ranks.csv"
    if not p_ranks.exists():
        raise FileNotFoundError(f"Missing Step3 ranks: {p_ranks}")

    ranks = pd.read_csv(p_ranks)
    if ranks.empty:
        out = pd.DataFrame(
            columns=["feature", "mean", "median", "std", "min", "max", "count"]
        )
    else:
        out = (
            ranks.groupby("feature")["rank"]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .reset_index()
        )

    p_out = cfg.out_dir / "table_summary_step3_ranks.csv"
    save_df(out, p_out)
    print("Wrote:", p_out)


if __name__ == "__main__":
    main()

