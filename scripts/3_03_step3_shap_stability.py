#!/usr/bin/env python
from __future__ import annotations

import os
import pandas as pd

from src.modeling.phase3.config import Phase3Config
from src.modeling.phase3.data import select_features, make_interactions_for_diagnostics
from src.modeling.phase3.models import fit_xgboost
from src.modeling.phase3.shap_stability import shap_rank_table_xgb
from src.modeling.phase3.reporting import save_df, fig_focus_rank_boxplot

def main() -> None:
    cfg = Phase3Config()
    df = pd.read_parquet(cfg.out_dir / "prepared.parquet")

    Xraw, _ = select_features(df)
    Xraw = make_interactions_for_diagnostics(Xraw)
    y = df[cfg.y_col]
    group = df["lolo_group"].astype(str)
    groups = sorted(group.unique())
    max_holdouts = int(os.getenv("PHASE3_MAX_HOLDOUTS", "0"))
    if max_holdouts > 0:
        groups = groups[:max_holdouts]
        print(f"[INFO] limiting holdouts to first {max_holdouts} groups")

    focus = [c for c in ["n_x_clay", "n_x_twi"] if c in Xraw.columns]

    all_tabs = []
    for g in groups:
        tr = group != g
        if tr.sum() == 0:
            continue

        # Train on others (use constrained or unconstrained; either is fine for stability test)
        model = fit_xgboost(Xraw.loc[tr], y.loc[tr], monotone_on_n=True)

        sample_n = min(5000, int(tr.sum()))
        if sample_n <= 0:
            continue
        tab = shap_rank_table_xgb(
            model,
            Xraw.loc[tr].sample(sample_n, random_state=1),
            focus=focus,
        )
        tab["lolo_holdout"] = g
        all_tabs.append(tab[tab["is_focus"]].copy())

        print("done SHAP ranks for holdout:", g)

    if not all_tabs:
        raise RuntimeError("Step3 produced no SHAP tables.")
    out = pd.concat(all_tabs, axis=0, ignore_index=True)
    save_df(out, cfg.out_dir / "step3_focus_feature_ranks.csv")

    # plot boxplot across holdouts
    fig_focus_rank_boxplot(out[["feature","rank"]], cfg.out_dir / "fig_step3_focus_rank_boxplot.png")
    print("Wrote step3 outputs to", cfg.out_dir)

if __name__ == "__main__":
    main()
