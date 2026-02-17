#!/usr/bin/env python
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fig_step1_rmse(table: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(table["model"], table["rmse"])
    ax.set_title("Step1: LOLO RMSE by Model")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fig_step2_delta_hist(step2: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    x = pd.to_numeric(step2["abs_delta_eonr"], errors="coerce").dropna()
    ax.hist(x, bins=15)
    ax.set_title("Step2: Distribution of |Delta EONR|")
    ax.set_xlabel("|Delta EONR| (lbs/acre)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fig_step2_delta_bar(step2: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    d = step2.sort_values("abs_delta_eonr", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(d["lolo_group"].astype(str), d["abs_delta_eonr"])
    ax.set_title("Step2: |Delta EONR| by Holdout")
    ax.set_xlabel("LOLO holdout")
    ax.set_ylabel("|Delta EONR| (lbs/acre)")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir = Path(os.getenv("PHASE3_OUT_DIR", "data/export/phase3"))
    if not out_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {out_dir}")

    out_tbl = out_dir / "paper_tables"
    out_fig = out_dir / "paper_figures"
    _ensure(out_tbl)
    _ensure(out_fig)

    # -------------------------
    # Step 1
    # -------------------------
    p1 = out_dir / "step1_predictions_obs.parquet"
    if p1.exists():
        s1 = pd.read_parquet(p1)
        t1 = pd.DataFrame(
            [
                {"model": "rf", "rmse": _rmse(s1["y_true"], s1["yhat_rf"])},
                {"model": "xgb_uncon", "rmse": _rmse(s1["y_true"], s1["yhat_xgb_uncon"])},
                {"model": "xgb_monotone", "rmse": _rmse(s1["y_true"], s1["yhat_xgb_monotone"])},
            ]
        )
        t1.to_csv(out_tbl / "table_step1_model_rmse.csv", index=False)

        rows = []
        for g, d in s1.groupby("lolo_group"):
            rows.append(
                {
                    "lolo_group": g,
                    "n_obs": len(d),
                    "rmse_rf": _rmse(d["y_true"], d["yhat_rf"]),
                    "rmse_xgb_uncon": _rmse(d["y_true"], d["yhat_xgb_uncon"]),
                    "rmse_xgb_monotone": _rmse(d["y_true"], d["yhat_xgb_monotone"]),
                }
            )
        pd.DataFrame(rows).to_csv(out_tbl / "table_step1_holdout_rmse.csv", index=False)
        _fig_step1_rmse(t1, out_fig / "fig_step1_model_rmse.png")

    # -------------------------
    # Step 2
    # -------------------------
    p2 = out_dir / "step2_eonr_gap.csv"
    if p2.exists():
        s2 = pd.read_csv(p2)
        t2 = pd.DataFrame(
            [
                {
                    "n_holdouts": int(len(s2)),
                    "mean_abs_delta_eonr": float(s2["abs_delta_eonr"].mean()),
                    "median_abs_delta_eonr": float(s2["abs_delta_eonr"].median()),
                    "p90_abs_delta_eonr": float(s2["abs_delta_eonr"].quantile(0.9)),
                    "mean_loss_$per_acre_proxy": float(s2["loss_$per_acre_proxy"].mean()),
                    "share_abs_delta_ge35": float((s2["abs_delta_eonr"] >= 35).mean()),
                    "share_abs_delta_ge40": float((s2["abs_delta_eonr"] >= 40).mean()),
                }
            ]
        )
        t2.to_csv(out_tbl / "table_step2_summary.csv", index=False)
        s2.sort_values("abs_delta_eonr", ascending=False).to_csv(
            out_tbl / "table_step2_by_holdout.csv", index=False
        )
        _fig_step2_delta_hist(s2, out_fig / "fig_step2_delta_hist.png")
        _fig_step2_delta_bar(s2, out_fig / "fig_step2_delta_by_holdout.png")

    # -------------------------
    # Step 3
    # -------------------------
    p3 = out_dir / "step3_focus_feature_ranks.csv"
    if p3.exists():
        s3 = pd.read_csv(p3)
        t3 = (
            s3.groupby("feature")["rank"]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .reset_index()
        )
        t3.to_csv(out_tbl / "table_step3_rank_summary.csv", index=False)

    # -------------------------
    # Step 4 assumptions table (from available step outputs)
    # -------------------------
    rows = []
    if p1.exists():
        mono_ratio = (
            float(t1.loc[t1["model"] == "xgb_monotone", "rmse"].iloc[0])
            / float(t1.loc[t1["model"] == "xgb_uncon", "rmse"].iloc[0])
        )
        rows.append(
            {
                "assumption": "Agronomic monotonicity",
                "required_condition": "RMSE(xgb_monotone) ~= RMSE(xgb_unconstrained)",
                "evidence": f"mono/uncon ratio={mono_ratio:.3f}",
                "verdict": "OK" if mono_ratio <= 1.10 else "Violated",
            }
        )

    if p2.exists():
        mean_abs = float(s2["abs_delta_eonr"].mean())
        rows.append(
            {
                "assumption": "Transferability",
                "required_condition": "Mean |Delta EONR| <= 35-40 lbs/acre",
                "evidence": f"mean |Delta EONR|={mean_abs:.2f}",
                "verdict": "OK" if mean_abs <= 40 else "Violated",
            }
        )

    if p3.exists():
        # rank stability proxy: lower rank std means more stable.
        mean_std = float(t3["std"].mean()) if len(t3) else np.nan
        rows.append(
            {
                "assumption": "Mechanism stability",
                "required_condition": "Consistent interaction importance across holdouts",
                "evidence": f"mean rank std={mean_std:.2f}",
                "verdict": "OK" if np.isfinite(mean_std) and mean_std <= 2.0 else "Violated",
            }
        )

    if rows:
        tab = pd.DataFrame(rows)
        # Covariate adequacy: derived
        v_transfer = tab.loc[tab["assumption"] == "Transferability", "verdict"]
        v_mech = tab.loc[tab["assumption"] == "Mechanism stability", "verdict"]
        if len(v_transfer) and len(v_mech):
            cov_ok = (v_transfer.iloc[0] == "OK") and (v_mech.iloc[0] == "OK")
            tab = pd.concat(
                [
                    tab,
                    pd.DataFrame(
                        [
                            {
                                "assumption": "Covariate adequacy",
                                "required_condition": "Public covariates encode transferable dynamics",
                                "evidence": f"derived from Transferability={v_transfer.iloc[0]}, Mechanism={v_mech.iloc[0]}",
                                "verdict": "OK" if cov_ok else "Violated",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        tab.to_csv(out_tbl / "table_step4_assumptions.csv", index=False)

    print("Wrote paper tables to:", out_tbl)
    print("Wrote paper figures to:", out_fig)


if __name__ == "__main__":
    main()

