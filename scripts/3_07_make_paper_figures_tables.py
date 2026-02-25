#!/usr/bin/env python
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def _r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    m = y_true.mean()
    denom = float(np.sum((y_true - m) ** 2))
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    sse = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - sse / denom)


def _ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fmt_mean_sd(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return "NA"
    # Render as two lines in PDF tables: mean on top, (sd) below.
    return r"\shortstack{" + f"{x.mean():.1f}" + r"\\" + f"({x.std(ddof=1):.1f})" + r"}"


def _write_markdown_table(df: pd.DataFrame, out_path: Path) -> None:
    cols = list(df.columns)
    lines = [
        "|" + "|".join(cols) + "|",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("|" + "|".join(str(row[c]) for c in cols) + "|")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bootstrap_ci_mean(values: pd.Series, *, n_boot: int = 1000, seed: int = 123) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(x[idx]))
    lo = float(np.quantile(means, 0.025))
    hi = float(np.quantile(means, 0.975))
    return lo, hi


def _bootstrap_ci_stat(
    values: pd.Series,
    *,
    stat: str = "mean",
    n_boot: int = 1000,
    seed: int = 123,
) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0])
    stat = stat.lower()

    def _stat(a: np.ndarray) -> float:
        if stat == "mean":
            return float(np.mean(a))
        if stat == "median":
            return float(np.median(a))
        if stat == "p90":
            return float(np.quantile(a, 0.9))
        raise ValueError(f"Unsupported stat: {stat}")

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = _stat(x[idx])
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return lo, hi


def _fmt_mean_ci(values: pd.Series, digits: int = 2) -> str:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return "NA"
    mu = float(x.mean())
    lo, hi = _bootstrap_ci_mean(x)
    return f"{mu:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _fmt_scalar(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _fmt_stat_paren(values: pd.Series, *, stat: str = "mean", digits: int = 2) -> str:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return "NA"
    stat = stat.lower()
    if stat == "mean":
        point = float(x.mean())
    elif stat == "median":
        point = float(x.median())
    elif stat == "p90":
        point = float(x.quantile(0.9))
    else:
        raise ValueError(f"Unsupported stat: {stat}")
    spread = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return f"{point:.{digits}f} ({spread:.{digits}f})"


def _fmt_stat_ci_multiline(values: pd.Series, *, stat: str = "mean", digits: int = 2) -> str:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return "NA"
    stat = stat.lower()
    if stat == "mean":
        point = float(x.mean())
    elif stat == "median":
        point = float(x.median())
    elif stat == "p90":
        point = float(x.quantile(0.9))
    else:
        raise ValueError(f"Unsupported stat: {stat}")
    lo, hi = _bootstrap_ci_stat(x, stat=stat)
    ci_text = f"[{lo:.{digits}f}, {hi:.{digits}f}]"
    return r"\shortstack{" + f"{point:.{digits}f}" + r"\\\mbox{" + ci_text + r"}" + r"}"


def _first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _build_lolo_stylized_table(
    s1: pd.DataFrame,
    s2: pd.DataFrame | None,
) -> pd.DataFrame:
    model_meta = [
        ("RF", "rf", "yhat_rf", "No"),
        ("XGB (Uncon.)", "xgb_uncon", "yhat_xgb_uncon", "No"),
        ("XGB (Monotone)", "xgb_monotone", "yhat_xgb_monotone", "Yes"),
    ]

    cols_s1 = set(s1.columns)
    pred_stats: dict[str, dict[str, pd.Series | int]] = {}
    for _label, key, pred_col, _mono in model_meta:
        if pred_col not in cols_s1:
            continue
        per_holdout = []
        for g, d in s1.groupby("lolo_group"):
            y = pd.to_numeric(d["y_true"], errors="coerce")
            yp = pd.to_numeric(d[pred_col], errors="coerce")
            keep = y.notna() & yp.notna()
            if keep.sum() == 0:
                continue
            y = y[keep]
            yp = yp[keep]
            per_holdout.append(
                {
                    "lolo_group": g,
                    "rmse": _rmse(y, yp),
                    "mae": _mae(y, yp),
                    "r2": _r2(y, yp),
                    "bias": float(np.mean(yp - y)),
                }
            )
        ph = pd.DataFrame(per_holdout)
        pred_stats[key] = {
            "rmse": ph["rmse"] if "rmse" in ph.columns else pd.Series(dtype=float),
            "mae": ph["mae"] if "mae" in ph.columns else pd.Series(dtype=float),
            "r2": ph["r2"] if "r2" in ph.columns else pd.Series(dtype=float),
            "bias": ph["bias"] if "bias" in ph.columns else pd.Series(dtype=float),
            "n_holdouts": int(len(ph)),
            "n_obs": int(s1[pred_col].notna().sum()),
        }

    dec_stats: dict[str, dict[str, pd.Series | float]] = {}
    if s2 is not None:
        cols_s2 = set(s2.columns)
        dec_map = {
            "rf": (
                ["abs_delta_eonr_rf"],
                ["loss_rf_$per_acre_proxy"],
            ),
            "xgb_uncon": (
                ["abs_delta_eonr_xgb_uncon"],
                ["loss_xgb_uncon_$per_acre_proxy"],
            ),
            "xgb_monotone": (
                ["abs_delta_eonr_xgb_monotone", "abs_delta_eonr"],
                ["loss_xgb_monotone_$per_acre_proxy", "loss_$per_acre_proxy"],
            ),
        }
        for key, (abs_candidates, loss_candidates) in dec_map.items():
            abs_col = _first_existing(cols_s2, abs_candidates)
            loss_col = _first_existing(cols_s2, loss_candidates)
            if abs_col is None:
                dec_stats[key] = {
                    "abs": pd.Series(dtype=float),
                    "loss": pd.Series(dtype=float),
                    "median_abs": float("nan"),
                    "p90_abs": float("nan"),
                    "share_ge35": float("nan"),
                }
                continue
            abs_s = pd.to_numeric(s2[abs_col], errors="coerce").dropna()
            loss_s = (
                pd.to_numeric(s2[loss_col], errors="coerce").dropna()
                if loss_col is not None
                else pd.Series(dtype=float)
            )
            dec_stats[key] = {
                "abs": abs_s,
                "loss": loss_s,
                "median_abs": float(abs_s.median()) if len(abs_s) else float("nan"),
                "p90_abs": float(abs_s.quantile(0.9)) if len(abs_s) else float("nan"),
                "share_ge35": float((abs_s >= 35).mean()) if len(abs_s) else float("nan"),
            }

    def _get_pred_stat(key: str, metric: str) -> pd.Series:
        return pred_stats.get(key, {}).get(metric, pd.Series(dtype=float))  # type: ignore[return-value]

    def _get_dec_series(key: str, metric: str) -> pd.Series:
        return dec_stats.get(key, {}).get(metric, pd.Series(dtype=float))  # type: ignore[return-value]

    def _get_dec_scalar(key: str, metric: str) -> float:
        v = dec_stats.get(key, {}).get(metric, float("nan"))
        return float(v) if isinstance(v, (float, int, np.floating)) else float("nan")

    rows = []
    rows.append({"Statistic": r"\shortstack{Panel A.\\Model Performance}", "RF": "", "XGB (Uncon.)": "", "XGB (Monotone)": ""})
    rows.append({"Statistic": "RMSE", "RF": _fmt_stat_ci_multiline(_get_pred_stat("rf", "rmse"), stat="mean", digits=3), "XGB (Uncon.)": _fmt_stat_ci_multiline(_get_pred_stat("xgb_uncon", "rmse"), stat="mean", digits=3), "XGB (Monotone)": _fmt_stat_ci_multiline(_get_pred_stat("xgb_monotone", "rmse"), stat="mean", digits=3)})
    rows.append({"Statistic": "MAE", "RF": _fmt_stat_ci_multiline(_get_pred_stat("rf", "mae"), stat="mean", digits=3), "XGB (Uncon.)": _fmt_stat_ci_multiline(_get_pred_stat("xgb_uncon", "mae"), stat="mean", digits=3), "XGB (Monotone)": _fmt_stat_ci_multiline(_get_pred_stat("xgb_monotone", "mae"), stat="mean", digits=3)})
    rows.append({"Statistic": "R^2", "RF": _fmt_stat_ci_multiline(_get_pred_stat("rf", "r2"), stat="mean", digits=3), "XGB (Uncon.)": _fmt_stat_ci_multiline(_get_pred_stat("xgb_uncon", "r2"), stat="mean", digits=3), "XGB (Monotone)": _fmt_stat_ci_multiline(_get_pred_stat("xgb_monotone", "r2"), stat="mean", digits=3)})
    rows.append({"Statistic": "Bias", "RF": _fmt_stat_ci_multiline(_get_pred_stat("rf", "bias"), stat="mean", digits=3), "XGB (Uncon.)": _fmt_stat_ci_multiline(_get_pred_stat("xgb_uncon", "bias"), stat="mean", digits=3), "XGB (Monotone)": _fmt_stat_ci_multiline(_get_pred_stat("xgb_monotone", "bias"), stat="mean", digits=3)})

    rows.append({"Statistic": r"\shortstack{Panel B.\\EONR differences}", "RF": "", "XGB (Uncon.)": "", "XGB (Monotone)": ""})
    rows.append({"Statistic": "Mean", "RF": _fmt_stat_ci_multiline(_get_dec_series("rf", "abs"), stat="mean", digits=2), "XGB (Uncon.)": _fmt_stat_ci_multiline(_get_dec_series("xgb_uncon", "abs"), stat="mean", digits=2), "XGB (Monotone)": _fmt_stat_ci_multiline(_get_dec_series("xgb_monotone", "abs"), stat="mean", digits=2)})
    rows.append({"Statistic": "Median", "RF": _fmt_stat_ci_multiline(_get_dec_series("rf", "abs"), stat="median", digits=2), "XGB (Uncon.)": _fmt_stat_ci_multiline(_get_dec_series("xgb_uncon", "abs"), stat="median", digits=2), "XGB (Monotone)": _fmt_stat_ci_multiline(_get_dec_series("xgb_monotone", "abs"), stat="median", digits=2)})
    rows.append({"Statistic": "P90", "RF": _fmt_stat_ci_multiline(_get_dec_series("rf", "abs"), stat="p90", digits=2), "XGB (Uncon.)": _fmt_stat_ci_multiline(_get_dec_series("xgb_uncon", "abs"), stat="p90", digits=2), "XGB (Monotone)": _fmt_stat_ci_multiline(_get_dec_series("xgb_monotone", "abs"), stat="p90", digits=2)})

    return pd.DataFrame(rows)


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


def _fig_weather_ablation(table: pd.DataFrame, out_path: Path, title: str, y_label: str) -> None:
    import matplotlib.pyplot as plt

    if table.empty:
        return

    x = np.arange(len(table))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    y_w = pd.to_numeric(table["with_weather"], errors="coerce")
    y_nw = pd.to_numeric(table["no_weather"], errors="coerce")
    if y_nw.notna().any():
        ax.bar(x - w / 2, y_w, width=w, label="With weather")
        ax.bar(x + w / 2, y_nw, width=w, label="No weather")
    else:
        ax.bar(x, y_w, width=0.6, label="With weather")
    ax.set_xticks(x)
    ax.set_xticklabels(table["item"].astype(str))
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fig_step2_delta_hist(step2: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    x = pd.to_numeric(step2["abs_delta_eonr"], errors="coerce").dropna()
    ax.hist(x, bins=15)
    ax.set_title("Distribution of Absolute EONR Differences")
    ax.set_xlabel("Absolute EONR differences (lbs/acre)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fig_step2_delta_bar(step2: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    d = step2.copy()
    d["lolo_group"] = d["lolo_group"].astype(str)
    d["abs_delta_eonr"] = pd.to_numeric(d["abs_delta_eonr"], errors="coerce")
    d = d.dropna(subset=["abs_delta_eonr"])
    if d.empty:
        return

    def _year_from_id(s: str) -> int | None:
        try:
            return int(str(s).split("_")[-1])
        except Exception:
            return None

    d["year"] = d["lolo_group"].map(_year_from_id)
    d = d[d["year"].notna()].copy()
    d["year"] = d["year"].astype(int)
    years = sorted(d["year"].unique().tolist())
    if not years:
        return

    n = len(years)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, max(4, 2.8 * nrows)), squeeze=False)

    for i, yr in enumerate(years):
        ax = axes[i // ncols][i % ncols]
        sub = d[d["year"] == yr].sort_values("abs_delta_eonr", ascending=False)
        ax.bar(sub["lolo_group"], sub["abs_delta_eonr"])
        ax.set_title(str(yr))
        ax.set_ylabel("Absolute EONR differences (lbs/acre)")
        ax.tick_params(axis="x", rotation=90, labelsize=7)

    # Hide any unused axes
    for j in range(len(years), nrows * ncols):
        axes[j // ncols][j % ncols].set_axis_off()

    fig.suptitle("Absolute EONR differences by holdout ffy_id (sorted within year)", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _fig_methods_climate(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    def _ridgeline(
        ax,
        frame: pd.DataFrame,
        value_col: str,
        title: str,
        xlab: str,
        color: str,
        scale: float = 1.0,
    ) -> None:
        d = frame[["year", value_col]].copy()
        d["year"] = pd.to_numeric(d["year"], errors="coerce")
        d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
        if scale != 1.0:
            d[value_col] = d[value_col] * float(scale)
        d = d.dropna()
        years = sorted(d["year"].astype(int).unique().tolist())
        if not years:
            ax.text(0.5, 0.5, f"No data: {value_col}", ha="center", va="center")
            ax.set_axis_off()
            return

        all_vals = d[value_col].to_numpy(dtype=float)
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            ax.text(0.5, 0.5, f"Insufficient variation: {value_col}", ha="center", va="center")
            ax.set_axis_off()
            return

        bins = np.linspace(vmin, vmax, 120)
        mids = 0.5 * (bins[:-1] + bins[1:])
        for idx, yr in enumerate(years):
            vals = d.loc[d["year"].astype(int) == yr, value_col].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            hist, _ = np.histogram(vals, bins=bins, density=True)
            if np.nanmax(hist) > 0:
                hist = hist / np.nanmax(hist) * 0.8
            y0 = float(idx)
            y1 = y0 + hist
            ax.fill_between(mids, y0, y1, color=color, alpha=0.35, linewidth=0)
            ax.plot(mids, y1, color=color, linewidth=0.9)

        ax.set_yticks(range(len(years)))
        ax.set_yticklabels([str(y) for y in years], fontsize=9)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Year")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    _ridgeline(
        axes[0],
        frame=df,
        value_col="prcp_t",
        title="Distribution of Total In-season Precipitation by Year",
        xlab="Precipitation (mm)",
        color="#2C7FB8",
        scale=0.01,
    )
    _ridgeline(
        axes[1],
        frame=df,
        value_col="gdd_t",
        title="Distribution of Total In-season Growing Degree Days by Year",
        xlab="GDD (degC-day; base 10C, cap 30C)",
        color="#8C510A",
        scale=0.01,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _fig_methods_application_map(exp_dir: Path, out_path: Path, out_note_path: Path) -> None:
    import geopandas as gpd
    import matplotlib.pyplot as plt

    preferred_ffy = os.getenv("METHODS_APP_MAP_FFY", "10_1_2022")
    preferred_path = exp_dir / f"{preferred_ffy}_exp.gpkg"

    if preferred_path.exists():
        p = preferred_path
    else:
        candidates = sorted(exp_dir.glob("*_exp.gpkg"))
        if not candidates:
            raise FileNotFoundError(f"No exp gpkg found in {exp_dir}")
        p = candidates[0]

    ffy_id = p.name.replace("_exp.gpkg", "")
    gdf = gpd.read_file(p, layer="exp")
    if "n_rate" not in gdf.columns:
        raise KeyError(f"Expected n_rate in {p}")

    fig, ax = plt.subplots(figsize=(6, 8))
    gdf.plot(
        column="n_rate",
        cmap="viridis",
        linewidth=0.0,
        legend=True,
        legend_kwds={"label": "Nitrogen rate (lbs/ac)", "shrink": 0.6},
        ax=ax,
    )
    ax.set_title(f"Input (N) application map by designed pattern ({ffy_id})", fontsize=11)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    out_note_path.write_text(
        f"application_map_source_ffy={ffy_id}\nsource_file={p}\n",
        encoding="utf-8",
    )


def _build_methods_assets(out_tbl: Path, out_fig: Path) -> None:
    phase2_path = Path(
        os.getenv(
            "PHASE3_DATA_PATH",
            "data/export/parquet/phase2_features/all_fields_features.parquet",
        )
    )
    if not phase2_path.exists():
        print(f"[WARN] Skip methods assets. Missing phase2 data: {phase2_path}")
        return

    df = pd.read_parquet(phase2_path)
    need = {"ffy_id", "farm", "field", "year", "yield", "n_rate", "prcp_t", "gdd_t", "edd_t"}
    missing = sorted(need - set(df.columns))
    if missing:
        print(f"[WARN] Skip methods assets. Missing columns: {missing}")
        return

    d = df.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d.dropna(subset=["year"]).copy()
    d["year"] = d["year"].astype(int)

    rows_wide = []
    for yr, g in d.groupby("year", sort=True):
        n_trials = int(g[["farm", "field"]].drop_duplicates().shape[0])
        n_obs = int(len(g))
        yv = _fmt_mean_sd(g["yield"])
        nv = _fmt_mean_sd(g["n_rate"])
        pr = _fmt_mean_sd(pd.to_numeric(g["prcp_t"], errors="coerce") * 0.01)
        gd = _fmt_mean_sd(pd.to_numeric(g["gdd_t"], errors="coerce") * 0.01)
        ed = _fmt_mean_sd(pd.to_numeric(g["edd_t"], errors="coerce") * 0.01)

        rows_wide.append(
            {
                "year": int(yr),
                "n_trials": "\\makebox[0.6cm][c]{" + str(n_trials) + "}",
                "n_obs": n_obs,
                "yield_bu_ac": yv,
                "n_rate_lbs_ac": nv,
                "prcp_t_mm": "\\makebox[3.2cm][l]{" + pr + "}",
                "gdd_t_degC_day": gd,
                "edd_t_degC_day": ed,
            }
        )

    tab_wide = pd.DataFrame(rows_wide).sort_values("year")
    tab_disp = tab_wide.rename(
        columns={
            "year": "Year",
            "n_trials": r"\makebox[0.6cm][c]{\shortstack{Trials\\\#}}",
            "n_obs": "Obs",
            "yield_bu_ac": "Yield (bu/ac)",
            "n_rate_lbs_ac": r"\shortstack{N\\(lbs/ac)}",
            "prcp_t_mm": r"\makebox[3.2cm][l]{\shortstack{Precipitation\\(mm)}}",
            "gdd_t_degC_day": r"\shortstack{GDD\\(C)}",
            "edd_t_degC_day": r"\shortstack{EDD\\(C)}",
        }
    )
    tab_wide.to_csv(out_tbl / "table_methods_data_summary_by_year.csv", index=False)
    _write_markdown_table(tab_disp, out_tbl / "table_methods_data_summary_by_year.md")

    _fig_methods_climate(d, out_fig / "fig_methods_climate_distributions.png")
    _fig_methods_application_map(
        exp_dir=Path("data/export/gpkg/exp"),
        out_path=out_fig / "fig_methods_application_map_placeholder.png",
        out_note_path=out_tbl / "fig_methods_application_map_source.txt",
    )


def main() -> None:
    out_dir = Path(os.getenv("PHASE3_OUT_DIR", "data/export/phase3"))
    if not out_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {out_dir}")

    out_tbl = out_dir / "paper_tables"
    out_fig = out_dir / "paper_figures"
    _ensure(out_tbl)
    _ensure(out_fig)

    # -------------------------
    # Methods assets (dataset summary + climate + application map)
    # -------------------------
    _build_methods_assets(out_tbl, out_fig)
    s1_for_lolo: pd.DataFrame | None = None
    s2_for_lolo: pd.DataFrame | None = None

    # -------------------------
    # Step 1
    # -------------------------
    p1 = out_dir / "step1_predictions_obs.parquet"
    if p1.exists():
        s1 = pd.read_parquet(p1)
        s1_for_lolo = s1.copy()
        t1 = pd.DataFrame(
            [
                {"model": "rf", "rmse": _rmse(s1["y_true"], s1["yhat_rf"])},
                {"model": "xgb_uncon", "rmse": _rmse(s1["y_true"], s1["yhat_xgb_uncon"])},
                {"model": "xgb_monotone", "rmse": _rmse(s1["y_true"], s1["yhat_xgb_monotone"])},
            ]
        )
        t1.to_csv(out_tbl / "table_step1_model_rmse.csv", index=False)
        _write_markdown_table(t1.round(3), out_tbl / "table_step1_model_rmse.md")

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
        t1h = pd.DataFrame(rows)
        t1h.to_csv(out_tbl / "table_step1_holdout_rmse.csv", index=False)
        _write_markdown_table(t1h.round(3), out_tbl / "table_step1_holdout_rmse.md")
        _fig_step1_rmse(t1, out_fig / "fig_step1_model_rmse.png")

        # Weather ablation (supports both new and backward-compatible outputs)
        model_col_map = [
            ("rf", "yhat_rf_with_weather", "yhat_rf_no_weather", "yhat_rf"),
            ("xgb_uncon", "yhat_xgb_uncon_with_weather", "yhat_xgb_uncon_no_weather", "yhat_xgb_uncon"),
            ("xgb_monotone", "yhat_xgb_monotone_with_weather", "yhat_xgb_monotone_no_weather", "yhat_xgb_monotone"),
        ]
        rows_ab = []
        for model_name, with_col, no_col, fallback_col in model_col_map:
            if with_col in s1.columns:
                rmse_with = _rmse(s1["y_true"], s1[with_col])
            elif fallback_col in s1.columns:
                rmse_with = _rmse(s1["y_true"], s1[fallback_col])
            else:
                continue
            rmse_no = _rmse(s1["y_true"], s1[no_col]) if no_col in s1.columns else np.nan
            rows_ab.append(
                {
                    "model": model_name,
                    "rmse_with_weather": rmse_with,
                    "rmse_no_weather": rmse_no,
                }
            )
        if rows_ab:
            t1_ab = pd.DataFrame(rows_ab)
            t1_ab["rmse_gain_from_weather"] = t1_ab["rmse_no_weather"] - t1_ab["rmse_with_weather"]
            t1_ab.to_csv(out_tbl / "table_step1_weather_ablation.csv", index=False)
            _write_markdown_table(t1_ab.round(3), out_tbl / "table_step1_weather_ablation.md")
            _fig_weather_ablation(
                pd.DataFrame(
                    {
                        "item": t1_ab["model"],
                        "with_weather": t1_ab["rmse_with_weather"],
                        "no_weather": t1_ab["rmse_no_weather"],
                    }
                ),
                out_fig / "fig_step1_weather_ablation.png",
                title="Step1: RMSE with vs without weather",
                y_label="RMSE",
            )

    # -------------------------
    # Step 2
    # -------------------------
    p2 = out_dir / "step2_eonr_gap.csv"
    if p2.exists():
        s2 = pd.read_csv(p2)
        s2_for_lolo = s2.copy()
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
        _write_markdown_table(t2.round(3), out_tbl / "table_step2_summary.md")
        s2_sorted = s2.sort_values("abs_delta_eonr", ascending=False)
        s2_sorted.to_csv(
            out_tbl / "table_step2_by_holdout.csv", index=False
        )
        _write_markdown_table(s2_sorted.round(3), out_tbl / "table_step2_by_holdout.md")
        _fig_step2_delta_hist(s2, out_fig / "fig_step2_delta_hist.png")
        _fig_step2_delta_bar(s2, out_fig / "fig_step2_delta_by_holdout.png")

        # Merge + transpose style table for Step2 (readable economics layout)
        cols2 = set(s2.columns)
        abs_with_col = "abs_delta_eonr_with_weather" if "abs_delta_eonr_with_weather" in cols2 else "abs_delta_eonr"
        loss_with_col = (
            "loss_with_weather_$per_acre_proxy"
            if "loss_with_weather_$per_acre_proxy" in cols2
            else "loss_$per_acre_proxy"
        )
        abs_no_col = "abs_delta_eonr_no_weather" if "abs_delta_eonr_no_weather" in cols2 else None
        loss_no_col = "loss_no_weather_$per_acre_proxy" if "loss_no_weather_$per_acre_proxy" in cols2 else None

        with_abs = pd.to_numeric(s2[abs_with_col], errors="coerce")
        with_loss = pd.to_numeric(s2[loss_with_col], errors="coerce")
        no_abs = pd.to_numeric(s2[abs_no_col], errors="coerce") if abs_no_col else pd.Series(dtype=float)
        no_loss = pd.to_numeric(s2[loss_no_col], errors="coerce") if loss_no_col else pd.Series(dtype=float)

        def _m(x: pd.Series) -> float:
            x = pd.to_numeric(x, errors="coerce").dropna()
            return float(x.mean()) if len(x) else float("nan")

        def _med(x: pd.Series) -> float:
            x = pd.to_numeric(x, errors="coerce").dropna()
            return float(x.median()) if len(x) else float("nan")

        def _p90(x: pd.Series) -> float:
            x = pd.to_numeric(x, errors="coerce").dropna()
            return float(x.quantile(0.9)) if len(x) else float("nan")

        def _share_ge(x: pd.Series, th: float) -> float:
            x = pd.to_numeric(x, errors="coerce").dropna()
            return float((x >= th).mean()) if len(x) else float("nan")

        rows_m = [
            {
                "Statistic": "Holdouts",
                "Value": int(len(with_abs.dropna())),
            },
            {
                "Statistic": "Mean abs EONR differences",
                "Value": _m(with_abs),
            },
            {
                "Statistic": "Median abs EONR differences",
                "Value": _med(with_abs),
            },
            {
                "Statistic": "P90 abs EONR differences",
                "Value": _p90(with_abs),
            },
            {
                "Statistic": "Mean loss ($/ac proxy)",
                "Value": _m(with_loss),
            },
            {
                "Statistic": "Share abs EONR differences >= 35",
                "Value": _share_ge(with_abs, 35.0),
            },
            {
                "Statistic": "Share abs EONR differences >= 40",
                "Value": _share_ge(with_abs, 40.0),
            },
        ]
        t2m = pd.DataFrame(rows_m)
        t2m["Value"] = pd.to_numeric(t2m["Value"], errors="coerce").round(3)
        t2m.to_csv(out_tbl / "table_step2_merged_transposed.csv", index=False)
        _write_markdown_table(t2m.fillna("NA"), out_tbl / "table_step2_merged_transposed.md")

        has_with_abs = "abs_delta_eonr_with_weather" in s2.columns or "abs_delta_eonr" in s2.columns
        has_with_loss = "loss_with_weather_$per_acre_proxy" in s2.columns or "loss_$per_acre_proxy" in s2.columns
        if has_with_abs and has_with_loss:
            abs_with_col = "abs_delta_eonr_with_weather" if "abs_delta_eonr_with_weather" in s2.columns else "abs_delta_eonr"
            loss_with_col = (
                "loss_with_weather_$per_acre_proxy"
                if "loss_with_weather_$per_acre_proxy" in s2.columns
                else "loss_$per_acre_proxy"
            )
            abs_no_col = "abs_delta_eonr_no_weather" if "abs_delta_eonr_no_weather" in s2.columns else None
            loss_no_col = "loss_no_weather_$per_acre_proxy" if "loss_no_weather_$per_acre_proxy" in s2.columns else None

            t2_ab = pd.DataFrame(
                [
                    {
                        "metric": "mean_abs_delta_eonr",
                        "with_weather": float(s2[abs_with_col].mean()),
                        "no_weather": float(s2[abs_no_col].mean()) if abs_no_col else np.nan,
                    },
                    {
                        "metric": "mean_loss_$per_acre_proxy",
                        "with_weather": float(s2[loss_with_col].mean()),
                        "no_weather": float(s2[loss_no_col].mean()) if loss_no_col else np.nan,
                    },
                ]
            )
            t2_ab["gain_from_weather"] = t2_ab["no_weather"] - t2_ab["with_weather"]
            t2_ab.to_csv(out_tbl / "table_step2_weather_ablation.csv", index=False)
            _write_markdown_table(t2_ab.round(3), out_tbl / "table_step2_weather_ablation.md")
            _fig_weather_ablation(
                pd.DataFrame(
                    {
                        "item": t2_ab["metric"],
                        "with_weather": t2_ab["with_weather"],
                        "no_weather": t2_ab["no_weather"],
                    }
                ),
                out_fig / "fig_step2_weather_ablation.png",
                title="Step2: Economic error with vs without weather",
                y_label="Value",
            )

    # -------------------------
    # Stylized LOLO comparison table (economics-style panels)
    # -------------------------
    if s1_for_lolo is not None:
        t_lolo = _build_lolo_stylized_table(s1_for_lolo, s2_for_lolo)
        t_lolo.to_csv(out_tbl / "table_lolo_model_comparison.csv", index=False)
        _write_markdown_table(t_lolo, out_tbl / "table_lolo_model_comparison.md")

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
        _write_markdown_table(t3.round(3), out_tbl / "table_step3_rank_summary.md")

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
        _write_markdown_table(tab, out_tbl / "table_step4_assumptions.md")

    print("Wrote paper tables to:", out_tbl)
    print("Wrote paper figures to:", out_fig)


if __name__ == "__main__":
    main()
