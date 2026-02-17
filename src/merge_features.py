from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .utils import get_logger, ensure_dir


@dataclass(frozen=True)
class MergePaths:
    """
    Centralized paths for merge outputs.
    """
    root_dir: Path

    def topo_dir(self) -> Path:
        return self.root_dir / "Data" / "Export" / "parquet" / "enriched_topo"

    def weather_dir(self) -> Path:
        return self.root_dir / "Data" / "Export" / "parquet" / "enriched_weather"

    def out_dir(self) -> Path:
        return self.root_dir / "Data" / "Export" / "parquet" / "model_ready"


def read_topo_features(paths: MergePaths, ffy_id: str) -> Optional[pd.DataFrame]:
    """
    Read per-plot topo features for one ffy_id.
    Expected file name from your topo batch:
      {ffy_id}_topo_enriched.parquet

    Returns:
      DataFrame with keys (ffy_id, obs_id) + topo columns
      or None if file missing.
    """
    logger = get_logger()
    topo_path = paths.topo_dir() / f"{ffy_id}_topo_enriched.parquet"
    if not topo_path.exists():
        logger.warning(f"Missing topo file -> {topo_path}")
        return None

    df = pd.read_parquet(topo_path)

    needed_keys = ["ffy_id", "obs_id"]
    for k in needed_keys:
        if k not in df.columns:
            raise KeyError(f"Topo file missing key column '{k}': {topo_path}")

    topo_cols = [c for c in ["elev_mean", "slope_mean", "aspect_mean", "tpi_mean"] if c in df.columns]
    keep_cols = needed_keys + topo_cols

    if len(topo_cols) == 0:
        logger.warning(f"Topo file has no expected topo columns. Columns={list(df.columns)}")

    return df[keep_cols].copy()


def read_weather_features(paths: MergePaths, ffy_id: str) -> Optional[pd.DataFrame]:
    """
    Read 1-row weather features for one ffy_id.
    Expected file name from your weather batch:
      {ffy_id}_weather_features.parquet

    Returns:
      1-row DataFrame keyed by ffy_id
      or None if file missing.
    """
    logger = get_logger()
    w_path = paths.weather_dir() / f"{ffy_id}_weather_features.parquet"
    if not w_path.exists():
        logger.warning(f"Missing weather file -> {w_path}")
        return None

    df = pd.read_parquet(w_path)

    if "ffy_id" not in df.columns:
        raise KeyError(f"Weather file missing 'ffy_id' column: {w_path}")

    if len(df) != 1:
        logger.warning(f"Weather file expected 1 row but got {len(df)} rows: {w_path}")

    return df.copy()


def merge_one_field(
    exp_df: pd.DataFrame,
    topo_df: Optional[pd.DataFrame],
    weather_df: Optional[pd.DataFrame],
    ffy_id: str,
) -> pd.DataFrame:
    """
    Merge base exp_df (plot-level) with:
      - topo_df: plot-level, keyed by (ffy_id, obs_id)
      - weather_df: field-year level, keyed by ffy_id (broadcast)

    Output: model-ready plot-level DataFrame.
    """
    logger = get_logger()

    if "ffy_id" not in exp_df.columns:
        raise KeyError("exp_df must include column 'ffy_id'")
    if "obs_id" not in exp_df.columns:
        raise KeyError("exp_df must include column 'obs_id'")

    out = exp_df.copy()

    # ---- topo join (plot-level) ----
    if topo_df is not None:
        before = len(out)
        out = out.merge(topo_df, on=["ffy_id", "obs_id"], how="left", validate="one_to_one")
        after = len(out)
        if after != before:
            raise RuntimeError(f"Topo merge changed row count: before={before}, after={after}")
        logger.info(f"Merged topo: added cols={ [c for c in topo_df.columns if c not in ['ffy_id','obs_id']] }")
    else:
        logger.info("Topo not merged (missing file)")

    # ---- weather join (field-year level, broadcast) ----
    if weather_df is not None:
        before = len(out)

        # Keep weather columns (exclude lon/lat if you want, but I keep them for now)
        w_keep = weather_df.copy()

        out = out.merge(w_keep, on="ffy_id", how="left", validate="many_to_one")
        after = len(out)
        if after != before:
            raise RuntimeError(f"Weather merge changed row count: before={before}, after={after}")
        logger.info(f"Merged weather: added cols={ [c for c in weather_df.columns if c != 'ffy_id'] }")
    else:
        logger.info("Weather not merged (missing file)")

    # Optional: enforce ffy_id consistency
    if (out["ffy_id"].astype(str) != str(ffy_id)).any():
        logger.warning("Merged output contains unexpected ffy_id values (check inputs).")

    return out


def save_model_ready(paths: MergePaths, ffy_id: str, df: pd.DataFrame) -> Path:
    """
    Save one field-year model-ready parquet.
    """
    logger = get_logger()
    out_dir = paths.out_dir()
    ensure_dir(out_dir)

    out_path = out_dir / f"{ffy_id}_model_ready.parquet"
    df.to_parquet(str(out_path), index=False, engine="pyarrow")
    logger.info(f"Saved model-ready: {out_path}")
    return out_path
