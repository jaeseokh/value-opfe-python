# src/etl/fetch_ssurgo.py
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.io import ProjectPaths, list_ffy_ids
from src.utils import ensure_dir, get_logger


def _project_root() -> Path:
    # .../src/etl/fetch_ssurgo.py -> project root is 2 levels up
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SsurgoPhase1Config:
    # IMPORTANT: R script lives under scripts/r/
    r_script: Path = _project_root() / "scripts" / "r" / "ssurgo_fetch_clip_one_field.R"
    out_layer: str = "ssurgo"
    overwrite: bool = False
    vars: tuple[str, ...] = (
        "sandtotal_r",
        "silttotal_r",
        "claytotal_r",
        "awc_r",
        "om_r",
        "dbovendry_r",
    )


def _out_gpkg_dir(paths: ProjectPaths) -> Path:
    return paths.gpkg_root / "enriched_ssurgo"


def _out_parquet_dir(paths: ProjectPaths) -> Path:
    return paths.parquet_enriched_ssurgo_dir


def _write_manifest_from_gpkg(ffy_id: str, gpkg_path: Path, out_parq: Path, layer: str) -> None:
    """
    Create a 1-row manifest parquet from the SSURGO gpkg:
      - counts
      - file path
      - layer name
    (No plot-level overlay here; Phase 2 will do that.)
    """
    logger = get_logger()
    gdf = gpd.read_file(gpkg_path, layer=layer)

    mukey_col = "mukey"
    if mukey_col in gdf.columns:
        n_unique = int(pd.Series(gdf[mukey_col].astype(str)).nunique())
    else:
        n_unique = 0

    row = {
        "ffy_id": ffy_id,
        "ssurgo_status": "OK",
        "n_polygons": int(len(gdf)),
        "n_unique_mukey": n_unique,
        "gpkg_path": str(gpkg_path),
        "layer": layer,
    }
    pd.DataFrame([row]).to_parquet(out_parq, index=False)
    logger.info("[OK] SSURGO manifest built: %s", out_parq)


def run_one_field(
    ffy_id: str,
    *,
    cfg: SsurgoPhase1Config | None = None,
    overwrite: bool | None = None,
    vars_csv: str | None = None,
) -> None:
    """
    Run one field.

    - If cfg is provided, it's used as base config.
    - overwrite (if not None) overrides cfg.overwrite
    - vars_csv (if provided) overrides cfg.vars (CSV list)
    """
    logger = get_logger()
    base = cfg or SsurgoPhase1Config()

    # override config from convenience args
    ow = base.overwrite if overwrite is None else bool(overwrite)

    if vars_csv and vars_csv.strip():
        v = [x.strip() for x in vars_csv.split(",")]
        v = [x for x in v if x]
        base = SsurgoPhase1Config(
            r_script=base.r_script,
            out_layer=base.out_layer,
            overwrite=ow,
            vars=tuple(v),
        )
    else:
        base = SsurgoPhase1Config(
            r_script=base.r_script,
            out_layer=base.out_layer,
            overwrite=ow,
            vars=base.vars,
        )

    if not base.r_script.exists():
        raise FileNotFoundError(f"Missing R script: {base.r_script}")

    paths = ProjectPaths(Path("."))
    bdry_gpkg = paths.gpkg_bdry_dir / f"{ffy_id}_bdry.gpkg"
    if not bdry_gpkg.exists():
        raise FileNotFoundError(f"Boundary GPKG not found: {bdry_gpkg}")

    out_gpkg_dir = _out_gpkg_dir(paths)
    out_parq_dir = _out_parquet_dir(paths)
    ensure_dir(out_gpkg_dir)
    ensure_dir(out_parq_dir)

    out_gpkg = out_gpkg_dir / f"{ffy_id}_ssurgo.gpkg"
    out_parq = out_parq_dir / f"{ffy_id}_ssurgo_table.parquet"

    # idempotent
    if out_gpkg.exists() and (not base.overwrite):
        logger.info("[SKIP] SSURGO gpkg exists: %s", out_gpkg)
        if not out_parq.exists():
            _write_manifest_from_gpkg(ffy_id, out_gpkg, out_parq, base.out_layer)
        return

    logger.info("[RUN] SSURGO fetch+clip ffy_id=%s", ffy_id)

    vars_csv2 = ",".join(base.vars)
    cmd = ["Rscript", str(base.r_script), ffy_id, str(bdry_gpkg), str(out_gpkg), vars_csv2]
    subprocess.run(cmd, check=True)

    logger.info("[OK] SSURGO gpkg saved: %s", out_gpkg)
    _write_manifest_from_gpkg(ffy_id, out_gpkg, out_parq, base.out_layer)
    logger.info("[OK] SSURGO parquet manifest saved: %s", out_parq)


def run_all_fields(*, overwrite: bool = False, vars_csv: str | None = None) -> None:
    logger = get_logger()
    paths = ProjectPaths(Path("."))

    ensure_dir(_out_gpkg_dir(paths))
    ensure_dir(_out_parquet_dir(paths))

    ffy_ids = list_ffy_ids(paths)
    logger.info("SSURGO: found %d ffy_ids", len(ffy_ids))

    ok = 0
    fail = 0
    for ffy_id in ffy_ids:
        try:
            run_one_field(ffy_id, overwrite=overwrite, vars_csv=vars_csv)
            ok += 1
        except Exception as e:
            logger.exception("[FAIL] SSURGO for %s: %s", ffy_id, e)
            fail += 1
            continue

    logger.info("SSURGO finished. ok=%d fail=%d", ok, fail)
