from __future__ import annotations

import numpy as np
import pandas as pd

def add_group_ids(df: pd.DataFrame, lolo_by_field_only: bool = True) -> pd.DataFrame:
    df = df.copy()

    # ensure ffy_id exists
    if "ffy_id" not in df.columns:
        df["ffy_id"] = df["farm"].astype(str) + "_" + df["field"].astype(str) + "_" + df["year"].astype(str)

    # LOLO group:
    # - by field only => hold out all years of that field together
    # - else by ffy_id (field-year)
    if lolo_by_field_only:
        df["lolo_group"] = df["farm"].astype(str) + "_" + df["field"].astype(str)
    else:
        df["lolo_group"] = df["ffy_id"].astype(str)

    return df

def basic_clean(df: pd.DataFrame, y_col: str = "yield", n_col: str = "n_rate") -> pd.DataFrame:
    df = df.copy()
    # Keep NA seed allowed; keep NA soil allowed
    df = df[df[y_col].notna() & df[n_col].notna()].copy()
    return df

def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Choose covariates used by ML models.
    We EXCLUDE ids and target, but INCLUDE soil/topo/weather and seed.
    """
    drop = {"yield", "ffy_id", "farm", "field", "year", "lolo_group"}
    cand = [c for c in df.columns if c not in drop]

    # Keep n_rate and seed, and all numeric topo/soil/weather
    # (this is intentionally permissive; you can prune later)
    keep = []
    for c in cand:
        # n_app_stage is currently constant "NA" in available weather outputs.
        if c in ("n_rate", "s_rate", "prcp_t", "gdd_t", "edd_t"):
            keep.append(c)
        elif c.startswith("topo_"):
            keep.append(c)
        elif c in ("sandtotal_r","silttotal_r","claytotal_r","awc_r","om_r","dbovendry_r","lon","lat"):
            keep.append(c)

    # make sure n_rate is present
    if "n_rate" not in keep:
        keep.append("n_rate")

    out = df[keep].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    keep2 = [c for c in out.columns if not out[c].isna().all()]
    return out[keep2].copy(), keep2

def make_interactions_for_diagnostics(dfX: pd.DataFrame) -> pd.DataFrame:
    """
    Step3 wants stability of mechanisms like N*clay or N*TWI.
    Create explicit interaction columns for inspection and SHAP ranking.
    """
    X = dfX.copy()
    if "n_rate" in X.columns and "claytotal_r" in X.columns:
        X["n_x_clay"] = X["n_rate"] * X["claytotal_r"]
    if "n_rate" in X.columns and "topo_twi_mean" in X.columns:
        X["n_x_twi"] = X["n_rate"] * X["topo_twi_mean"]
    return X
