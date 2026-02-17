#!/usr/bin/env python
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.modeling.phase3.config import Phase3Config
from src.modeling.phase3.data import (
    build_test_field_reference_row,
    make_interactions_for_diagnostics,
    select_features,
)
from src.modeling.phase3.models import fit_xgboost, predict_over_n_grid
from src.modeling.phase3.eonr import fit_true_field_gam_and_eonr, eonr_from_curve
from src.modeling.phase3.reporting import save_df, fig_delta_eonr

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

    n_grid_points = int(os.getenv("PHASE3_N_GRID_POINTS", "100"))

    rows = []
    for g in groups:
        tr = group != g
        te = group == g
        if tr.sum() == 0 or te.sum() == 0:
            continue

        # Train on all other fields
        xgb_c = fit_xgboost(Xraw.loc[tr], y.loc[tr], monotone_on_n=True)

        # True EONR from observed within-field yield~N
        df_field = df.loc[te].copy()
        n_obs = pd.to_numeric(df_field[cfg.n_col], errors="coerce").dropna()
        if n_obs.empty:
            print("skip:", g, "(no observed n_rate in test field)")
            continue
        n_min = float(n_obs.min())
        n_max = float(n_obs.max())
        if not np.isfinite(n_min) or not np.isfinite(n_max) or n_min >= n_max:
            print("skip:", g, "(invalid or degenerate N range in test field)")
            continue

        # User-specified Step2 rule:
        # use test-field N range and split into 100 equal intervals.
        n_grid = np.linspace(n_min, n_max, n_grid_points)

        true_eonr, true_ygrid = fit_true_field_gam_and_eonr(
            df_field, cfg.y_col, cfg.n_col, n_grid, cfg.price_yield_per_bu, cfg.price_n_per_lb
        )
        if not np.isfinite(true_eonr):
            print("skip:", g, "(true EONR could not be estimated)")
            continue

        # Pred EONR: one response curve using fixed test-field profile
        # (median field covariates + single weather value).
        X_field = Xraw.loc[te].copy()
        X_ref = build_test_field_reference_row(X_field, n_col=cfg.n_col)
        pred_long = predict_over_n_grid(xgb_c, X_ref, n_grid, n_col=cfg.n_col)
        pred_curve = pred_long["yhat"].to_numpy(dtype=float)
        pred_eonr = eonr_from_curve(n_grid, pred_curve, cfg.price_yield_per_bu, cfg.price_n_per_lb)

        delta = float(pred_eonr - true_eonr)
        abs_delta = float(abs(delta))

        # economic loss proxy: profit(true_eonr) - profit(pred_eonr) using TRUE curve
        # (evaluate both on the true GAM curve for comparability)
        from src.modeling.phase3.eonr import profit
        true_pi = profit(n_grid, true_ygrid, cfg.price_yield_per_bu, cfg.price_n_per_lb)
        i_true = int(np.nanargmin(np.abs(n_grid - true_eonr)))
        i_pred = int(np.nanargmin(np.abs(n_grid - pred_eonr)))
        pi_true = float(true_pi[i_true])
        pi_pred = float(true_pi[i_pred])
        loss = float(pi_true - pi_pred)

        rows.append({
            "lolo_group": g,
            "n_grid_min": n_min,
            "n_grid_max": n_max,
            "n_grid_points": n_grid_points,
            "true_eonr": true_eonr,
            "pred_eonr_xgb_monotone": pred_eonr,
            "delta_eonr": delta,
            "abs_delta_eonr": abs_delta,
            "loss_$per_acre_proxy": loss,
        })

        print("done:", g, "true=", true_eonr, "pred=", pred_eonr, "absΔ=", abs_delta)

    if not rows:
        raise RuntimeError("Step2 produced no rows. Check LOLO groups and data coverage.")
    out = pd.DataFrame(rows)
    save_df(out, cfg.out_dir / "step2_eonr_gap.csv")
    fig_delta_eonr(out, cfg.out_dir / "fig_step2_delta_eonr.png")
    print("Wrote step2 outputs to", cfg.out_dir)

if __name__ == "__main__":
    main()
