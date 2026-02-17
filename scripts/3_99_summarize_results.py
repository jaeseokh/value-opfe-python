#!/usr/bin/env python
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(os.getenv("PHASE3_OUT_DIR", "data/export/phase3"))
PHASE2_ALL = Path(
    os.getenv("PHASE3_DATA_PATH", "data/export/parquet/phase2_features/all_fields_features.parquet")
)
EONR_GAM_SUMMARY = Path("data/export/figures/eonr_gam/eonr_gam_checks_summary.csv")


def _fmt(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    if isinstance(x, int):
        return str(x)
    return f"{x:,.3f}"


def step0_data_diagnostics() -> None:
    print("\n=== Step 0 / Data Diagnostics ===")
    p = OUT_DIR / "prepared.parquet"
    if not p.exists():
        print(f"[MISSING] {p}")
        if PHASE2_ALL.exists():
            print(f"[HINT] run: .venv/bin/python scripts/3_00_prepare.py (input exists: {PHASE2_ALL})")
        return

    df = pd.read_parquet(p)
    print(f"rows={len(df):,} cols={len(df.columns)}")
    if {"ffy_id", "farm", "field"}.issubset(df.columns):
        n_ffy = df["ffy_id"].nunique()
        n_field = df[["farm", "field"]].drop_duplicates().shape[0]
        print(f"field-years={n_ffy} fields={n_field}")

    if "lolo_group" in df.columns:
        cnt = df.groupby("lolo_group").size()
        q = cnt.quantile([0, 0.25, 0.5, 0.75, 0.9, 1.0]).to_dict()
        print("obs/LOLO-group quantiles:", {k: int(v) for k, v in q.items()})

    key_cov = [
        "s_rate",
        "topo_dem_mean",
        "topo_twi_mean",
        "claytotal_r",
        "awc_r",
        "prcp_t",
        "gdd_t",
        "edd_t",
    ]
    miss = {}
    for c in key_cov:
        if c in df.columns:
            miss[c] = float(df[c].isna().mean())
    print("missing rate (selected):", {k: round(v, 4) for k, v in miss.items()})


def step1_model_fit() -> None:
    print("\n=== Step 1 / Constrained vs Unconstrained Fit ===")
    p = OUT_DIR / "step1_predictions_obs.parquet"
    if not p.exists():
        print(f"[MISSING] {p}")
        print("[HINT] run: .venv/bin/python scripts/3_01_step1_train_constrained.py")
        return

    df = pd.read_parquet(p)
    need = {"y_true", "yhat_rf", "yhat_xgb_uncon", "yhat_xgb_monotone"}
    miss = need - set(df.columns)
    if miss:
        print(f"[ERROR] missing cols in Step1 output: {sorted(miss)}")
        return

    def rmse(a: pd.Series, b: pd.Series) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    out = {
        "rmse_rf": rmse(df["y_true"], df["yhat_rf"]),
        "rmse_xgb_uncon": rmse(df["y_true"], df["yhat_xgb_uncon"]),
        "rmse_xgb_monotone": rmse(df["y_true"], df["yhat_xgb_monotone"]),
    }
    print("overall RMSE:", {k: round(v, 4) for k, v in out.items()})

    if np.isfinite(out["rmse_xgb_uncon"]) and out["rmse_xgb_uncon"] > 0:
        ratio = out["rmse_xgb_monotone"] / out["rmse_xgb_uncon"]
        print(f"mono/uncon RMSE ratio={ratio:.3f}")
        if ratio <= 1.10:
            print("interpretation: monotone constraint is not very costly (supports agronomic plausibility).")
        else:
            print("interpretation: monotone constraint costs substantial fit (data may violate smooth monotone response).")


def step2_eonr_gap() -> None:
    print("\n=== Step 2 / EONR Transfer Gap ===")
    p = OUT_DIR / "step2_eonr_gap.csv"
    if not p.exists():
        print(f"[MISSING] {p}")
        print("[HINT] run: .venv/bin/python scripts/3_02_step2_eonr_gap.py")
        return

    df = pd.read_csv(p)
    if df.empty:
        print("[EMPTY] step2_eonr_gap.csv")
        return

    m_abs = float(df["abs_delta_eonr"].mean())
    med_abs = float(df["abs_delta_eonr"].median())
    p90_abs = float(df["abs_delta_eonr"].quantile(0.9))
    m_loss = float(df["loss_$per_acre_proxy"].mean()) if "loss_$per_acre_proxy" in df.columns else np.nan
    print(
        "abs_delta_eonr:",
        {"mean": round(m_abs, 3), "median": round(med_abs, 3), "p90": round(p90_abs, 3)},
    )
    print("loss_$per_acre_proxy mean:", round(m_loss, 3) if np.isfinite(m_loss) else "NA")

    if m_abs <= 35:
        print("interpretation: transfer recommendations are in potentially usable range on average.")
    else:
        print("interpretation: transfer failure is economically/materially large on average.")


def step3_mechanism_stability() -> None:
    print("\n=== Step 3 / Mechanism Stability ===")
    p = OUT_DIR / "step3_focus_feature_ranks.csv"
    if not p.exists():
        print(f"[MISSING] {p}")
        print("[HINT] run: .venv/bin/python scripts/3_03_step3_shap_stability.py (requires shap)")
        return

    df = pd.read_csv(p)
    if df.empty:
        print("[EMPTY] step3_focus_feature_ranks.csv")
        return
    if not {"feature", "rank"}.issubset(df.columns):
        print("[ERROR] step3 table missing required columns feature/rank")
        return

    tab = (
        df.groupby("feature")["rank"]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
        .sort_values("mean")
    )
    print(tab.to_string(index=False))
    print("interpretation: lower mean rank + lower std implies more stable mechanism across holdouts.")


def step4_reports() -> None:
    print("\n=== Step 4 / Report Tables ===")
    p2 = OUT_DIR / "table_summary_step2.csv"
    p3 = OUT_DIR / "table_summary_step3_ranks.csv"

    if p2.exists():
        print(f"found: {p2}")
        print(pd.read_csv(p2).to_string(index=False))
    else:
        print(f"[MISSING] {p2}")

    if p3.exists():
        print(f"found: {p3}")
        print(pd.read_csv(p3).to_string(index=False))
    else:
        print(f"[MISSING] {p3}")


def optional_true_eonr_baseline() -> None:
    print("\n=== Optional / In-Field True EONR Baseline (existing) ===")
    if not EONR_GAM_SUMMARY.exists():
        print(f"[MISSING] {EONR_GAM_SUMMARY}")
        return
    df = pd.read_csv(EONR_GAM_SUMMARY)
    if "eonr" not in df.columns:
        print("[ERROR] eonr column not found in GAM summary")
        return
    q = df["eonr"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).to_dict()
    print("true EONR quantiles:", {k: round(v, 3) for k, v in q.items()})
    print(f"n_field_year={df['ffy_id'].nunique() if 'ffy_id' in df.columns else len(df)}")


def main() -> None:
    step0_data_diagnostics()
    step1_model_fit()
    step2_eonr_gap()
    step3_mechanism_stability()
    step4_reports()
    optional_true_eonr_baseline()


if __name__ == "__main__":
    main()
