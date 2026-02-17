from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.io import ProjectPaths
from src.utils import ensure_dir, get_logger


REQUIRED_BASE_COLS: tuple[str, ...] = (
    "yield",
    "n_rate",
    "ffy_id",
    "farm",
    "field",
    "year",
)

CANONICAL_FEATURE_CANDIDATES: tuple[str, ...] = (
    "n_rate",
    "s_rate",
    "topo_dem_mean",
    "topo_slope_mean",
    "topo_aspect_mean",
    "topo_twi_mean",
    "sandtotal_r",
    "silttotal_r",
    "claytotal_r",
    "awc_r",
    "om_r",
    "dbovendry_r",
    "prcp_t",
    "gdd_t",
    "edd_t",
    "lon",
    "lat",
)


@dataclass(frozen=True)
class Phase3Config:
    split_level: str = "field"  # {"field", "field_year"}
    seed: int = 42
    corn_price: float = 5.0
    n_price: float = 0.8
    n_grid_points: int = 101
    model_names: tuple[str, ...] = ("rf", "xgb", "xgb_mono")
    min_n_levels_for_true_eonr: int = 5


def _phase2_default_path(paths: ProjectPaths) -> Path:
    return paths.parquet_root / "phase2_features" / "all_fields_features.parquet"


def _phase3_out_dir(paths: ProjectPaths) -> Path:
    return paths.parquet_root / "phase3_diagnostics"


def _phase3_plot_dir(paths: ProjectPaths) -> Path:
    return paths.root / "outputs" / "phase3_diagnostics"


def _validate_phase2_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Phase2 table missing required columns: {missing}")


def _as_field_id(df: pd.DataFrame) -> pd.Series:
    farm = pd.to_numeric(df["farm"], errors="coerce").astype("Int64")
    field = pd.to_numeric(df["field"], errors="coerce").astype("Int64")
    return farm.astype(str) + "_" + field.astype(str)


def _prepare_dataset(raw: pd.DataFrame, split_level: str) -> pd.DataFrame:
    df = raw.copy()
    _validate_phase2_schema(df)

    # enforce numeric types where expected
    numeric_cols = list(dict.fromkeys(["yield", *CANONICAL_FEATURE_CANDIDATES]))
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["yield"].notna() & df["n_rate"].notna()].copy()

    df["location_field"] = _as_field_id(df)
    df["location_field_year"] = df["ffy_id"].astype(str)

    if split_level == "field":
        df["location_id"] = df["location_field"]
    elif split_level == "field_year":
        df["location_id"] = df["location_field_year"]
    else:
        raise ValueError(f"Unsupported split_level: {split_level}")

    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in CANONICAL_FEATURE_CANDIDATES if c in df.columns]
    if "n_rate" not in cols:
        raise ValueError("Feature columns must include n_rate")
    if len(cols) < 2:
        raise ValueError("Too few feature columns found. Need n_rate plus covariates.")
    return cols


def _monotone_constraint_for_nrate(feature_cols: list[str]) -> str:
    vals = ["1" if c == "n_rate" else "0" for c in feature_cols]
    return "(" + ",".join(vals) + ")"


def _make_model(name: str, feature_cols: list[str], seed: int) -> Pipeline:
    if name == "rf":
        est = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", est),
            ]
        )

    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError(
            "xgboost is required for model_name 'xgb' or 'xgb_mono'."
        ) from e

    kwargs: dict[str, Any] = dict(
        objective="reg:squarederror",
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
    )
    if name == "xgb_mono":
        kwargs["monotone_constraints"] = _monotone_constraint_for_nrate(feature_cols)

    if name not in {"xgb", "xgb_mono"}:
        raise ValueError(f"Unsupported model name: {name}")

    est = xgb.XGBRegressor(**kwargs)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", est),
        ]
    )


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _true_eonr_from_gam(
    df_ffy: pd.DataFrame,
    n_grid: np.ndarray,
    corn_price: float,
    n_price: float,
    min_n_levels: int,
) -> dict[str, float]:
    out = {
        "eonr_true": np.nan,
        "profit_true_at_true": np.nan,
        "profit_true_at_pred": np.nan,
    }

    if df_ffy["n_rate"].nunique(dropna=True) < min_n_levels:
        return out

    X = df_ffy[["n_rate"]].to_numpy()
    y = df_ffy["yield"].to_numpy()
    if len(y) < max(8, min_n_levels + 2):
        return out

    try:
        from pygam import LinearGAM, s

        gam = LinearGAM(s(0)).fit(X, y)
        y_hat = gam.predict(n_grid.reshape(-1, 1))
    except Exception:
        return out

    profit = (corn_price * y_hat) - (n_price * n_grid)
    idx = int(np.nanargmax(profit))
    out["eonr_true"] = float(n_grid[idx])
    out["profit_true_at_true"] = float(profit[idx])

    # stash arrays in dict for downstream interpolation
    out["_profit_grid"] = profit
    out["_n_grid"] = n_grid
    return out


def _predicted_eonr_from_transfer_model(
    fitted_model: Pipeline,
    df_ffy: pd.DataFrame,
    feature_cols: list[str],
    n_grid: np.ndarray,
    corn_price: float,
    n_price: float,
) -> dict[str, float]:
    X_base = df_ffy[feature_cols].copy()
    if X_base.empty:
        return {"eonr_pred": np.nan, "profit_pred_at_pred": np.nan}

    mean_y = []
    for n_val in n_grid:
        x = X_base.copy()
        x["n_rate"] = float(n_val)
        y_hat = fitted_model.predict(x)
        mean_y.append(float(np.nanmean(y_hat)))

    y_curve = np.asarray(mean_y, dtype=float)
    profit_curve = (corn_price * y_curve) - (n_price * n_grid)
    idx = int(np.nanargmax(profit_curve))
    return {
        "eonr_pred": float(n_grid[idx]),
        "profit_pred_at_pred": float(profit_curve[idx]),
    }


def _iter_lolo_splits(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    out: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for loc in sorted(df["location_id"].dropna().unique().tolist()):
        test = df[df["location_id"] == loc].copy()
        train = df[df["location_id"] != loc].copy()
        if test.empty or train.empty:
            continue
        out.append((str(loc), train, test))
    return out


def run_lolo_and_eonr(
    *,
    input_parquet: Path | None = None,
    cfg: Phase3Config | None = None,
) -> dict[str, Path]:
    logger = get_logger()
    cfg = cfg or Phase3Config()
    paths = ProjectPaths(Path("."))

    src_path = input_parquet or _phase2_default_path(paths)
    if not src_path.exists():
        raise FileNotFoundError(f"Missing Phase2 combined parquet: {src_path}")

    out_dir = _phase3_out_dir(paths)
    ensure_dir(out_dir)

    logger.info("[RUN] Phase3 Step1-2 | load dataset: %s", src_path)
    raw = pd.read_parquet(src_path)
    df = _prepare_dataset(raw, split_level=cfg.split_level)
    feature_cols = _feature_columns(df)
    logger.info(
        "Phase3 Step1-2 | rows=%d locations=%d ffy=%d features=%d",
        len(df),
        df["location_id"].nunique(),
        df["ffy_id"].nunique(),
        len(feature_cols),
    )

    split_rows: list[dict[str, Any]] = []
    eonr_rows: list[dict[str, Any]] = []

    for loc, train, test in _iter_lolo_splits(df):
        X_train = train[feature_cols]
        y_train = train["yield"]
        X_test = test[feature_cols]
        y_test = test["yield"]

        n_train = len(train)
        n_test = len(test)
        logger.info("LOLO location=%s | n_train=%d n_test=%d", loc, n_train, n_test)

        for model_name in cfg.model_names:
            try:
                model = _make_model(model_name, feature_cols, seed=cfg.seed)
                model.fit(X_train, y_train)
                yhat_train = model.predict(X_train)
                yhat_test = model.predict(X_test)
            except Exception as e:
                logger.exception("[FAIL] model fit | location=%s model=%s err=%s", loc, model_name, e)
                continue

            split_rows.append(
                {
                    "split_level": cfg.split_level,
                    "location_id": loc,
                    "model": model_name,
                    "rmse_train": _rmse(y_train, yhat_train),
                    "rmse_test": _rmse(y_test, yhat_test),
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_ffy_in_test": test["ffy_id"].nunique(),
                }
            )

            for ffy_id, g_ffy in test.groupby("ffy_id"):
                n_min = float(g_ffy["n_rate"].min())
                n_max = float(g_ffy["n_rate"].max())
                if not np.isfinite(n_min) or not np.isfinite(n_max) or n_min == n_max:
                    continue
                n_grid = np.linspace(n_min, n_max, cfg.n_grid_points)

                true_dict = _true_eonr_from_gam(
                    g_ffy,
                    n_grid=n_grid,
                    corn_price=cfg.corn_price,
                    n_price=cfg.n_price,
                    min_n_levels=cfg.min_n_levels_for_true_eonr,
                )
                pred_dict = _predicted_eonr_from_transfer_model(
                    model,
                    g_ffy,
                    feature_cols=feature_cols,
                    n_grid=n_grid,
                    corn_price=cfg.corn_price,
                    n_price=cfg.n_price,
                )

                eonr_true = true_dict["eonr_true"]
                eonr_pred = pred_dict["eonr_pred"]
                delta_n = (
                    float(abs(eonr_true - eonr_pred))
                    if np.isfinite(eonr_true) and np.isfinite(eonr_pred)
                    else np.nan
                )

                profit_loss = np.nan
                if np.isfinite(eonr_pred) and "_n_grid" in true_dict:
                    profit_curve = true_dict["_profit_grid"]
                    ng = true_dict["_n_grid"]
                    p_pred_on_true = float(np.interp(eonr_pred, ng, profit_curve))
                    true_dict["profit_true_at_pred"] = p_pred_on_true
                    if np.isfinite(true_dict["profit_true_at_true"]):
                        profit_loss = float(true_dict["profit_true_at_true"] - p_pred_on_true)

                eonr_rows.append(
                    {
                        "split_level": cfg.split_level,
                        "location_id": loc,
                        "ffy_id": str(ffy_id),
                        "model": model_name,
                        "n_obs_test_ffy": len(g_ffy),
                        "n_min": n_min,
                        "n_max": n_max,
                        "eonr_true": eonr_true,
                        "eonr_pred": eonr_pred,
                        "delta_n": delta_n,
                        "profit_true_at_true": true_dict["profit_true_at_true"],
                        "profit_true_at_pred": true_dict["profit_true_at_pred"],
                        "profit_loss": profit_loss,
                        "corn_price": cfg.corn_price,
                        "n_price": cfg.n_price,
                    }
                )

    split_df = pd.DataFrame(split_rows)
    eonr_df = pd.DataFrame(eonr_rows)

    split_summary = (
        split_df.groupby("model", as_index=False)
        .agg(
            n_splits=("location_id", "nunique"),
            rmse_train_mean=("rmse_train", "mean"),
            rmse_test_mean=("rmse_test", "mean"),
            rmse_test_median=("rmse_test", "median"),
        )
        if not split_df.empty
        else pd.DataFrame()
    )

    eonr_summary = (
        eonr_df.groupby("model", as_index=False)
        .agg(
            n_ffy=("ffy_id", "nunique"),
            delta_n_mean=("delta_n", "mean"),
            delta_n_median=("delta_n", "median"),
            delta_n_p90=("delta_n", lambda x: float(np.nanquantile(x, 0.9))),
            profit_loss_mean=("profit_loss", "mean"),
            profit_loss_median=("profit_loss", "median"),
        )
        if not eonr_df.empty
        else pd.DataFrame()
    )

    split_path = out_dir / "step1_lolo_rmse_by_location.parquet"
    split_summary_path = out_dir / "step1_lolo_rmse_summary.parquet"
    eonr_path = out_dir / "step2_eonr_by_field.parquet"
    eonr_summary_path = out_dir / "step2_eonr_summary.parquet"

    split_df.to_parquet(split_path, index=False)
    split_summary.to_parquet(split_summary_path, index=False)
    eonr_df.to_parquet(eonr_path, index=False)
    eonr_summary.to_parquet(eonr_summary_path, index=False)

    logger.info("[OK] Phase3 Step1-2 outputs saved: %s", out_dir)
    return {
        "step1_lolo_by_location": split_path,
        "step1_lolo_summary": split_summary_path,
        "step2_eonr_by_field": eonr_path,
        "step2_eonr_summary": eonr_summary_path,
    }


def _add_interaction_features(df: pd.DataFrame, moderators: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for m in moderators:
        if m in out.columns:
            out[f"n_rate_x_{m}"] = out["n_rate"] * out[m]
    return out


def _scenario_predict_for_interaction_sign(
    model: Pipeline,
    base_x: pd.DataFrame,
    moderator: str,
    n_value: float,
    z_value: float,
) -> float:
    x = base_x.copy()
    x["n_rate"] = n_value
    if moderator in x.columns:
        x[moderator] = z_value
    inter_col = f"n_rate_x_{moderator}"
    if inter_col in x.columns:
        x[inter_col] = x["n_rate"] * x[moderator]
    yhat = model.predict(x)
    return float(np.nanmean(yhat))


def run_mechanism_stability(
    *,
    input_parquet: Path | None = None,
    split_level: str = "field",
    seed: int = 42,
    moderators: tuple[str, ...] = ("claytotal_r", "topo_twi_mean"),
    min_rows_per_location: int = 30,
) -> dict[str, Path]:
    logger = get_logger()
    paths = ProjectPaths(Path("."))
    src_path = input_parquet or _phase2_default_path(paths)
    if not src_path.exists():
        raise FileNotFoundError(f"Missing Phase2 combined parquet: {src_path}")

    out_dir = _phase3_out_dir(paths)
    ensure_dir(out_dir)

    raw = pd.read_parquet(src_path)
    df = _prepare_dataset(raw, split_level=split_level)
    df = _add_interaction_features(df, moderators=moderators)
    feat_cols = _feature_columns(df)
    feat_cols = feat_cols + [c for c in df.columns if c.startswith("n_rate_x_")]

    rows: list[dict[str, Any]] = []

    for location_id, g in df.groupby("location_id"):
        if len(g) < min_rows_per_location:
            continue
        if g["n_rate"].nunique(dropna=True) < 4:
            continue

        g = g.copy()
        for c in feat_cols + ["yield"]:
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce")
        g = g.dropna(subset=["yield", "n_rate"])
        if len(g) < min_rows_per_location:
            continue

        X = g[feat_cols]
        y = g["yield"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=3,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        try:
            model.fit(X_train, y_train)
            pi = permutation_importance(
                model,
                X_test,
                y_test,
                scoring="neg_root_mean_squared_error",
                n_repeats=8,
                random_state=seed,
                n_jobs=-1,
            )
        except Exception as e:
            logger.warning("Mechanism fit failed for location=%s: %s", location_id, e)
            continue

        imp = pd.Series(pi.importances_mean, index=feat_cols).sort_values(ascending=False)
        ranks = imp.rank(ascending=False, method="min")

        for moderator in moderators:
            inter_col = f"n_rate_x_{moderator}"
            if moderator not in g.columns or inter_col not in imp.index:
                continue
            z_series = pd.to_numeric(g[moderator], errors="coerce")
            if z_series.notna().sum() < 5:
                continue

            n_lo = float(np.nanquantile(g["n_rate"], 0.2))
            n_hi = float(np.nanquantile(g["n_rate"], 0.8))
            z_lo = float(np.nanquantile(z_series, 0.2))
            z_hi = float(np.nanquantile(z_series, 0.8))
            if not np.isfinite([n_lo, n_hi, z_lo, z_hi]).all():
                continue

            base_x = g[feat_cols]
            y_nlo_zlo = _scenario_predict_for_interaction_sign(model, base_x, moderator, n_lo, z_lo)
            y_nhi_zlo = _scenario_predict_for_interaction_sign(model, base_x, moderator, n_hi, z_lo)
            y_nlo_zhi = _scenario_predict_for_interaction_sign(model, base_x, moderator, n_lo, z_hi)
            y_nhi_zhi = _scenario_predict_for_interaction_sign(model, base_x, moderator, n_hi, z_hi)

            effect_low_z = y_nhi_zlo - y_nlo_zlo
            effect_high_z = y_nhi_zhi - y_nlo_zhi
            cross_proxy = effect_high_z - effect_low_z

            rows.append(
                {
                    "split_level": split_level,
                    "location_id": str(location_id),
                    "n_obs": len(g),
                    "moderator": moderator,
                    "interaction_feature": inter_col,
                    "importance": float(imp.get(inter_col, np.nan)),
                    "rank": float(ranks.get(inter_col, np.nan)),
                    "cross_proxy": float(cross_proxy),
                    "cross_sign": int(np.sign(cross_proxy)),
                }
            )

    detail_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame()
    if not detail_df.empty:
        def _rank_var(x: pd.Series) -> float:
            return float(np.nanvar(x.to_numpy(dtype=float)))

        def _sign_consistency(x: pd.Series) -> float:
            x = x.dropna().astype(int)
            x = x[x != 0]
            if x.empty:
                return np.nan
            p_pos = (x == 1).mean()
            p_neg = (x == -1).mean()
            return float(max(p_pos, p_neg))

        summary_df = (
            detail_df.groupby("moderator", as_index=False)
            .agg(
                n_locations=("location_id", "nunique"),
                importance_mean=("importance", "mean"),
                importance_std=("importance", "std"),
                rank_variance=("rank", _rank_var),
                sign_consistency=("cross_sign", _sign_consistency),
                sign_positive_share=("cross_sign", lambda x: float((x == 1).mean())),
                sign_negative_share=("cross_sign", lambda x: float((x == -1).mean())),
            )
        )

    detail_path = out_dir / "step3_mechanism_detail.parquet"
    summary_path = out_dir / "step3_mechanism_summary.parquet"
    detail_df.to_parquet(detail_path, index=False)
    summary_df.to_parquet(summary_path, index=False)

    logger.info("[OK] Phase3 Step3 outputs saved: %s", out_dir)
    return {
        "step3_mechanism_detail": detail_path,
        "step3_mechanism_summary": summary_path,
    }


def build_assumptions_table(
    *,
    mono_rmse_tolerance: float = 1.10,
    delta_n_threshold: float = 35.0,
    sign_consistency_threshold: float = 0.70,
    rank_variance_threshold: float = 2.0,
) -> Path:
    logger = get_logger()
    paths = ProjectPaths(Path("."))
    out_dir = _phase3_out_dir(paths)
    ensure_dir(out_dir)

    p_step1 = out_dir / "step1_lolo_rmse_summary.parquet"
    p_step2 = out_dir / "step2_eonr_summary.parquet"
    p_step3 = out_dir / "step3_mechanism_summary.parquet"

    if not p_step1.exists():
        raise FileNotFoundError(f"Missing Step1 summary: {p_step1}")
    if not p_step2.exists():
        raise FileNotFoundError(f"Missing Step2 summary: {p_step2}")
    if not p_step3.exists():
        raise FileNotFoundError(f"Missing Step3 summary: {p_step3}")

    step1 = pd.read_parquet(p_step1)
    step2 = pd.read_parquet(p_step2)
    step3 = pd.read_parquet(p_step3)

    # Step 1: monotonicity check by constrained-vs-unconstrained RMSE
    mono_verdict = "Violated"
    mono_evidence = "xgb_mono or unconstrained baseline missing."
    if not step1.empty:
        mono_row = step1.loc[step1["model"] == "xgb_mono"]
        base_row = step1.loc[step1["model"].isin(["xgb", "rf"])].sort_values("rmse_test_mean")
        if not mono_row.empty and not base_row.empty:
            mono = float(mono_row.iloc[0]["rmse_test_mean"])
            base = float(base_row.iloc[0]["rmse_test_mean"])
            ratio = mono / base if base > 0 else np.inf
            mono_verdict = "OK" if ratio <= mono_rmse_tolerance else "Violated"
            mono_evidence = (
                f"xgb_mono RMSE={mono:.3f}, best_unconstrained RMSE={base:.3f}, "
                f"ratio={ratio:.3f}, threshold<={mono_rmse_tolerance:.2f}"
            )

    # Step 2: transferability by EONR error
    transfer_verdict = "Violated"
    transfer_evidence = "No EONR summary rows."
    if not step2.empty:
        best = step2.sort_values("delta_n_mean").iloc[0]
        best_delta = float(best["delta_n_mean"])
        transfer_verdict = "OK" if best_delta <= delta_n_threshold else "Violated"
        transfer_evidence = (
            f"best_model={best['model']}, mean_deltaN={best_delta:.2f}, "
            f"threshold<={delta_n_threshold:.1f}"
        )

    # Step 3: mechanism stability
    mech_verdict = "Violated"
    mech_evidence = "No mechanism summary rows."
    if not step3.empty:
        ok_sign = (step3["sign_consistency"] >= sign_consistency_threshold).fillna(False)
        ok_rank = (step3["rank_variance"] <= rank_variance_threshold).fillna(False)
        is_ok = bool((ok_sign & ok_rank).all())
        mech_verdict = "OK" if is_ok else "Violated"
        mech_evidence = (
            f"all moderators satisfy sign_consistency>={sign_consistency_threshold:.2f} "
            f"and rank_variance<={rank_variance_threshold:.2f}: {is_ok}"
        )

    # Step 4: covariate adequacy, derived from transferability + mechanism
    cov_verdict = "OK" if (transfer_verdict == "OK" and mech_verdict == "OK") else "Violated"
    cov_evidence = (
        f"Derived from transferability={transfer_verdict}, mechanism={mech_verdict}."
    )

    rows = [
        {
            "assumption": "Agronomic monotonicity",
            "required_condition": "Constrained ~= unconstrained fit quality",
            "empirical_evidence": mono_evidence,
            "verdict": mono_verdict,
        },
        {
            "assumption": "Transferability",
            "required_condition": "Low LOLO EONR error",
            "empirical_evidence": transfer_evidence,
            "verdict": transfer_verdict,
        },
        {
            "assumption": "Mechanism stability",
            "required_condition": "Stable interaction rank/sign across fields",
            "empirical_evidence": mech_evidence,
            "verdict": mech_verdict,
        },
        {
            "assumption": "Covariate adequacy",
            "required_condition": "Public covariates encode transferable dynamics",
            "empirical_evidence": cov_evidence,
            "verdict": cov_verdict,
        },
    ]

    out = pd.DataFrame(rows)
    out_path = out_dir / "step4_assumptions_table.parquet"
    out.to_parquet(out_path, index=False)
    out.to_csv(out_dir / "step4_assumptions_table.csv", index=False)
    logger.info("[OK] Phase3 Step4 outputs saved: %s", out_path)
    return out_path


def make_phase3_plots(
    *,
    model_for_scatter: str | None = None,
) -> dict[str, Path]:
    logger = get_logger()
    paths = ProjectPaths(Path("."))
    in_dir = _phase3_out_dir(paths)
    out_dir = _phase3_plot_dir(paths)
    ensure_dir(out_dir)

    eonr_path = in_dir / "step2_eonr_by_field.parquet"
    mech_path = in_dir / "step3_mechanism_summary.parquet"

    if not eonr_path.exists():
        raise FileNotFoundError(f"Missing Step2 detail: {eonr_path}")
    if not mech_path.exists():
        raise FileNotFoundError(f"Missing Step3 summary: {mech_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required to generate Phase3 plots.") from e

    eonr = pd.read_parquet(eonr_path)
    mech = pd.read_parquet(mech_path)

    out: dict[str, Path] = {}

    # Plot 1: distribution of Delta EONR by model
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, g in eonr.groupby("model"):
        vals = pd.to_numeric(g["delta_n"], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=30, alpha=0.45, label=str(model))
    ax.set_xlabel("Delta EONR (abs lbs/acre)")
    ax.set_ylabel("Count")
    ax.set_title("Step2: Delta EONR Distribution")
    ax.legend()
    p1 = out_dir / "01_delta_eonr_distribution.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    out["delta_distribution"] = p1

    # Plot 2: EONR true vs predicted scatter
    if model_for_scatter is None and not eonr.empty:
        model_for_scatter = str(
            eonr.groupby("model")["delta_n"].mean().sort_values().index[0]
        )
    gg = eonr.loc[eonr["model"] == model_for_scatter].copy() if model_for_scatter else eonr.copy()
    gg = gg.dropna(subset=["eonr_true", "eonr_pred"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gg["eonr_true"], gg["eonr_pred"], s=12, alpha=0.7)
    if not gg.empty:
        lo = float(np.nanmin(np.r_[gg["eonr_true"].to_numpy(), gg["eonr_pred"].to_numpy()]))
        hi = float(np.nanmax(np.r_[gg["eonr_true"].to_numpy(), gg["eonr_pred"].to_numpy()]))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel("True EONR")
    ax.set_ylabel("Predicted EONR")
    ax.set_title(f"Step2: True vs Predicted EONR ({model_for_scatter})")
    p2 = out_dir / "02_true_vs_pred_eonr.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    out["true_vs_pred_scatter"] = p2

    # Plot 3: mechanism sign consistency
    fig, ax = plt.subplots(figsize=(7, 4))
    if not mech.empty and "moderator" in mech.columns:
        ax.bar(mech["moderator"].astype(str), mech["sign_consistency"].astype(float))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Sign consistency")
    ax.set_title("Step3: Mechanism Sign Consistency")
    p3 = out_dir / "03_mechanism_sign_consistency.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    out["mechanism_sign_consistency"] = p3

    logger.info("[OK] Phase3 plots saved: %s", out_dir)
    return out
