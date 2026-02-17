from __future__ import annotations

import numpy as np
import pandas as pd


def add_group_ids(df: pd.DataFrame, lolo_by_field_only: bool = True) -> pd.DataFrame:
    out = df.copy()

    if "ffy_id" not in out.columns:
        needed = {"farm", "field", "year"}
        missing = needed - set(out.columns)
        if missing:
            raise KeyError(f"Need ffy_id or farm/field/year. Missing: {sorted(missing)}")
        out["ffy_id"] = (
            out["farm"].astype(str) + "_" + out["field"].astype(str) + "_" + out["year"].astype(str)
        )

    if lolo_by_field_only:
        needed = {"farm", "field"}
        missing = needed - set(out.columns)
        if missing:
            raise KeyError(f"Need farm/field for field-level LOLO. Missing: {sorted(missing)}")
        out["lolo_group"] = out["farm"].astype(str) + "_" + out["field"].astype(str)
    else:
        out["lolo_group"] = out["ffy_id"].astype(str)

    return out


def basic_clean(df: pd.DataFrame, y_col: str = "yield", n_col: str = "n_rate") -> pd.DataFrame:
    out = df.copy()
    for c in (y_col, n_col):
        if c not in out.columns:
            raise KeyError(f"Missing required column: {c}")
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[out[y_col].notna() & out[n_col].notna()].copy()
    return out


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Keep treatment + agronomic covariates used in Phase3.
    """
    drop = {"yield", "ffy_id", "farm", "field", "year", "lolo_group"}
    candidates = [c for c in df.columns if c not in drop]

    keep: list[str] = []
    for c in candidates:
        # n_app_stage is currently a constant string ("NA") in this dataset,
        # so we exclude it from numeric ML features until event-timing data exist.
        if c in ("n_rate", "s_rate", "prcp_t", "gdd_t", "edd_t"):
            keep.append(c)
        elif c.startswith("topo_"):
            keep.append(c)
        elif c in (
            "sandtotal_r",
            "silttotal_r",
            "claytotal_r",
            "awc_r",
            "om_r",
            "dbovendry_r",
            "lon",
            "lat",
        ):
            keep.append(c)

    if "n_rate" not in keep:
        keep.append("n_rate")

    out = df[keep].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop non-informative columns (all NaN after numeric conversion).
    keep2 = [c for c in out.columns if not out[c].isna().all()]
    out = out[keep2].copy()
    return out, keep2


def make_interactions_for_diagnostics(df_x: pd.DataFrame) -> pd.DataFrame:
    out = df_x.copy()
    if "n_rate" in out.columns and "claytotal_r" in out.columns:
        out["n_x_clay"] = out["n_rate"] * out["claytotal_r"]
    if "n_rate" in out.columns and "topo_twi_mean" in out.columns:
        out["n_x_twi"] = out["n_rate"] * out["topo_twi_mean"]
    return out


def build_test_field_reference_row(
    x_field: pd.DataFrame,
    *,
    n_col: str = "n_rate",
) -> pd.DataFrame:
    """
    Build a single fixed covariate profile for Step2 prediction:
      - n_rate is placeholder (overwritten on grid)
      - field covariates: median over test-field subplots
      - weather covariates: single value (first non-null, expected constant per field-year)
    """
    if x_field.empty:
        raise ValueError("x_field is empty")
    if n_col not in x_field.columns:
        raise KeyError(f"Missing N column in x_field: {n_col}")

    weather_prefixes = ("prcp_", "gdd_", "edd_", "tmax_", "tmin_", "heat_days")
    row: dict[str, float] = {}

    for c in x_field.columns:
        s = pd.to_numeric(x_field[c], errors="coerce")
        non_na = s.dropna()

        if c == n_col:
            row[c] = float(non_na.median()) if len(non_na) else np.nan
            continue

        if c.startswith(weather_prefixes):
            # Weather should already be a single scalar per field-year.
            row[c] = float(non_na.iloc[0]) if len(non_na) else np.nan
        else:
            row[c] = float(non_na.median()) if len(non_na) else np.nan

    return pd.DataFrame([row], columns=list(x_field.columns))
