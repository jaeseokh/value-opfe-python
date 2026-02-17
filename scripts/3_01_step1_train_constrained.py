#!/usr/bin/env python
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.modeling.phase3.config import Phase3Config
from src.modeling.phase3.data import select_features, make_interactions_for_diagnostics
from src.modeling.phase3.models import fit_xgboost, fit_random_forest

def main() -> None:
    cfg = Phase3Config()
    df = pd.read_parquet(cfg.out_dir / "prepared.parquet")

    Xraw, feat_cols = select_features(df)
    Xraw = make_interactions_for_diagnostics(Xraw)
    y = df[cfg.y_col]
    group = df["lolo_group"].astype(str)
    groups = sorted(group.unique())
    max_holdouts = int(os.getenv("PHASE3_MAX_HOLDOUTS", "0"))
    if max_holdouts > 0:
        groups = groups[:max_holdouts]
        print(f"[INFO] limiting holdouts to first {max_holdouts} groups")

    out_rows = []
    for g in groups:
        tr = group != g
        te = group == g
        if tr.sum() == 0 or te.sum() == 0:
            continue

        Xtr, ytr = Xraw.loc[tr], y.loc[tr]
        Xte, yte = Xraw.loc[te], y.loc[te]

        # RF
        rf = fit_random_forest(Xtr, ytr)
        yhat_rf = rf.predict(Xte)

        # XGB unconstrained
        xgb_u = fit_xgboost(Xtr, ytr, monotone_on_n=False)
        yhat_xgb_u = xgb_u.predict(Xte)

        # XGB constrained (monotone in N)
        xgb_c = fit_xgboost(Xtr, ytr, monotone_on_n=True)
        yhat_xgb_c = xgb_c.predict(Xte)

        obs_id = (
            df.loc[te, "obs_id"].values
            if "obs_id" in df.columns
            else df.loc[te].index.astype(str).to_numpy()
        )
        out_rows.append(pd.DataFrame({
            "lolo_group": g,
            "ffy_id": df.loc[te, "ffy_id"].values,
            "obs_id": obs_id,
            "y_true": yte.values,
            "yhat_rf": yhat_rf,
            "yhat_xgb_uncon": yhat_xgb_u,
            "yhat_xgb_monotone": yhat_xgb_c,
        }))

        print("done group:", g, "n_test=", te.sum())

    if not out_rows:
        raise RuntimeError("No LOLO folds were produced in Step1.")
    out = pd.concat(out_rows, axis=0, ignore_index=True)
    p = cfg.out_dir / "step1_predictions_obs.parquet"
    out.to_parquet(p, index=False)
    print("Wrote:", p, "shape=", out.shape)

if __name__ == "__main__":
    main()
