#!/usr/bin/env python
from __future__ import annotations

import pandas as pd

from src.modeling.phase3.config import Phase3Config
from src.modeling.phase3.reporting import fig_delta_eonr, fig_focus_rank_boxplot


def main() -> None:
    cfg = Phase3Config()

    p_step2 = cfg.out_dir / "step2_eonr_gap.csv"
    p_step3 = cfg.out_dir / "step3_focus_feature_ranks.csv"

    if not p_step2.exists():
        raise FileNotFoundError(f"Missing Step2 table: {p_step2}")
    if not p_step3.exists():
        raise FileNotFoundError(f"Missing Step3 table: {p_step3}")

    eonr = pd.read_csv(p_step2)
    ranks = pd.read_csv(p_step3)

    p_fig2 = cfg.out_dir / "fig_step2_delta_eonr.png"
    p_fig3 = cfg.out_dir / "fig_step3_focus_rank_boxplot.png"

    fig_delta_eonr(eonr, p_fig2)
    fig_focus_rank_boxplot(ranks[["feature", "rank"]], p_fig3)

    print("Wrote:", p_fig2)
    print("Wrote:", p_fig3)


if __name__ == "__main__":
    main()

