# src/etl/build_model_ready.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.io import ProjectPaths
from src.utils import ensure_dir, get_logger


def _repeat_1row_to_n(df_1row: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df_1row) != 1:
        raise ValueError(f"Expected 1 row table, got {len(df_1row)} rows.")
    return pd.concat([df_1row] * n, ignore_index=True)


def _attach_manifest(
    df: pd.DataFrame,
    manifest: pd.DataFrame | None,
    logger,
    ffy_id: str,
    name: str,
    *,
    status_col: str | None = None,
    missing_status: str = "MISSING",
) -> pd.DataFrame:
    """
    Phase 1 policy:
      - TOPO / SSURGO / WEATHER are 1-row-per-ffy manifests/summaries.
      - Repeat them to match plot-level exp rows.
      - If missing, do not block pipeline.
    """
    if manifest is None or manifest.empty:
        logger.warning("[%s] %s table missing/empty; continuing without it", ffy_id, name)
        if status_col and status_col not in df.columns:
            df[status_col] = missing_status
        return df

    # 1-row manifest: repeat to plot rows
    if len(manifest) == 1:
        rep = _repeat_1row_to_n(manifest, len(df))
        cols = [c for c in rep.columns if c not in df.columns]
        out = pd.concat([df, rep[cols]], axis=1)
        if status_col and status_col not in out.columns:
            out[status_col] = "OK"
        return out

    # Allow already plot-level features by rowcount match (optional)
    if len(manifest) == len(df):
        cols = [c for c in manifest.columns if c not in df.columns]
        out = pd.concat([df, manifest[cols].reset_index(drop=True)], axis=1)
        if status_col and status_col not in out.columns:
            out[status_col] = "OK"
        return out

    logger.warning(
        "[%s] %s table has %d rows (expected 1 or %d). Skipping attach in Phase 1.",
        ffy_id, name, len(manifest), len(df)
    )
    if status_col and status_col not in df.columns:
        df[status_col] = "SKIPPED_BAD_SHAPE"
    return df


def _build_one(*, ffy_id: str, paths: ProjectPaths, overwrite: bool, logger) -> None:
    exp_table_dir = paths.parquet_exp_table_dir
    topo_dir = paths.parquet_enriched_topo_dir
    ssurgo_dir = paths.parquet_enriched_ssurgo_dir
    weather_dir = paths.parquet_enriched_weather_dir
    out_dir = paths.parquet_model_ready_dir

    exp_path = exp_table_dir / f"{ffy_id}_exp_table.parquet"
    if not exp_path.exists():
        raise FileNotFoundError(f"[{ffy_id}] Missing exp_table parquet: {exp_path}")

    topo_path = topo_dir / f"{ffy_id}_topo_table.parquet"
    ssurgo_path = ssurgo_dir / f"{ffy_id}_ssurgo_table.parquet"
    weather_path = weather_dir / f"{ffy_id}_weather_table.parquet"
    out_path = out_dir / f"{ffy_id}_model_ready.parquet"

    if out_path.exists() and not overwrite:
        logger.info("[SKIP] model_ready exists: %s", out_path)
        return

    exp = pd.read_parquet(exp_path)

    # Required columns in exp_table
    required = ["yield", "n_rate"]
    missing = [c for c in required if c not in exp.columns]
    if missing:
        raise ValueError(f"[{ffy_id}] exp_table missing required columns: {missing}")

    if "obs_id" not in exp.columns:
        logger.warning("[%s] exp_table has no obs_id column (ok for Phase1; Phase2 will prefer it).", ffy_id)

    topo = pd.read_parquet(topo_path) if topo_path.exists() else pd.DataFrame()
    ssurgo = pd.read_parquet(ssurgo_path) if ssurgo_path.exists() else pd.DataFrame()
    weather = pd.read_parquet(weather_path) if weather_path.exists() else pd.DataFrame()

    df = exp.copy()
    df["ffy_id"] = ffy_id

    df = _attach_manifest(df, topo, logger, ffy_id, name="TOPO",
                          status_col="topo_status", missing_status="MISSING_TOPO")
    df = _attach_manifest(df, ssurgo, logger, ffy_id, name="SSURGO",
                          status_col="ssurgo_status", missing_status="MISSING_SSURGO")
    df = _attach_manifest(df, weather, logger, ffy_id, name="WEATHER",
                          status_col="weather_status", missing_status="MISSING_WEATHER")

    df.to_parquet(out_path, index=False)
    logger.info("[OK] model_ready saved: %s", out_path)


def run(*, overwrite: bool = False) -> None:
    """
    Build Phase 1 model_ready:
      model_ready = exp_table + repeated 1-row manifests (topo/ssurgo/weather)

    This should NOT do spatial overlays or zonal stats.
    """
    logger = get_logger()
    paths = ProjectPaths(Path("."))

    for d in [
        paths.parquet_exp_table_dir,
        paths.parquet_enriched_topo_dir,
        paths.parquet_enriched_ssurgo_dir,
        paths.parquet_enriched_weather_dir,
    ]:
        if not d.exists():
            raise FileNotFoundError(f"Missing required directory: {d}")

    ensure_dir(paths.parquet_model_ready_dir)

    exp_files = sorted(paths.parquet_exp_table_dir.glob("*_exp_table.parquet"))
    if not exp_files:
        raise FileNotFoundError(f"No exp_table parquet found in: {paths.parquet_exp_table_dir}")

    ffy_ids = [p.name.replace("_exp_table.parquet", "") for p in exp_files]
    logger.info("Found %d exp tables.", len(ffy_ids))

    ok = 0
    fail = 0
    for ffy_id in ffy_ids:
        try:
            _build_one(ffy_id=ffy_id, paths=paths, overwrite=overwrite, logger=logger)
            ok += 1
        except Exception as e:
            fail += 1
            logger.exception("[FAIL] build_model_ready ffy_id=%s | %s", ffy_id, e)
            continue

    logger.info("build_model_ready finished. ok=%d fail=%d", ok, fail)


def run_one_field(ffy_id: str, *, overwrite: bool = False) -> None:
    logger = get_logger()
    paths = ProjectPaths(Path("."))
    ensure_dir(paths.parquet_model_ready_dir)
    _build_one(ffy_id=ffy_id, paths=paths, overwrite=overwrite, logger=logger)
