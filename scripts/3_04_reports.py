#!/usr/bin/env python
from __future__ import annotations

import pandas as pd
from src.modeling.phase3.config import Phase3Config
from src.modeling.phase3.reporting import save_df

def main() -> None:
    cfg = Phase3Config()

    p_eonr = cfg.out_dir / "step2_eonr_gap.csv"
    p_ranks = cfg.out_dir / "step3_focus_feature_ranks.csv"
    if not p_eonr.exists():
        raise FileNotFoundError(f"Missing Step2 output: {p_eonr}")
    eonr = pd.read_csv(p_eonr)
    ranks = pd.read_csv(p_ranks) if p_ranks.exists() else pd.DataFrame()

    summary = pd.DataFrame([{
        "n_holdouts": int(eonr.shape[0]),
        "mean_abs_delta_eonr": float(eonr["abs_delta_eonr"].mean()),
        "median_abs_delta_eonr": float(eonr["abs_delta_eonr"].median()),
        "mean_loss_$per_acre_proxy": float(eonr["loss_$per_acre_proxy"].mean()),
    }])

    save_df(summary, cfg.out_dir / "table_summary_step2.csv")
    print("Wrote:", cfg.out_dir / "table_summary_step2.csv")

    if not ranks.empty:
        rank_sum = (
            ranks.groupby("feature")["rank"]
            .agg(["mean","median","std","min","max","count"])
            .reset_index()
        )
        save_df(rank_sum, cfg.out_dir / "table_summary_step3_ranks.csv")
        print("Wrote:", cfg.out_dir / "table_summary_step3_ranks.csv")

    print("Wrote summary tables to", cfg.out_dir)

if __name__ == "__main__":
    main()
