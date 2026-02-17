# src/io.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd

from .utils import get_logger

FFY_PATTERN = re.compile(r"^\d+_\d+_\d{4}$")


# -----------------------
# Helpers
# -----------------------
def is_valid_ffy_id(ffy_id: str) -> bool:
    return bool(FFY_PATTERN.match(ffy_id))


def parse_ffy_from_filename(path: Path) -> Optional[str]:
    """
    Extract ffy_id from file names like:
      8_1_2023_exp.gpkg
      8_1_2023_bdry.gpkg
      8_1_2023_exp_table.parquet
    """
    parts = path.name.split("_")
    if len(parts) < 4:
        return None
    candidate = "_".join(parts[:3])
    return candidate if is_valid_ffy_id(candidate) else None


def pick_existing_dir(candidates: list[Path], what: str) -> Path:
    for d in candidates:
        if d.exists():
            return d
    raise FileNotFoundError(f"{what} not found. Tried:\n" + "\n".join(str(d) for d in candidates))



def _crs_matches_expected(gdf: gpd.GeoDataFrame, expected: str) -> bool:
    """
    Return True if gdf.crs matches expected CRS.
    Tries EPSG integer comparison first, then falls back to CRS string.
    """
    if gdf.crs is None:
        return False

    # expected like "EPSG:4326"
    expected_epsg = None
    try:
        if expected.upper().startswith("EPSG:"):
            expected_epsg = int(expected.split(":")[1])
    except Exception:
        expected_epsg = None

    gdf_epsg = None
    try:
        gdf_epsg = gdf.crs.to_epsg()
    except Exception:
        gdf_epsg = None

    if expected_epsg is not None and gdf_epsg is not None:
        return gdf_epsg == expected_epsg

    # fallback: compare normalized strings
    try:
        return gdf.crs.to_string().upper() == expected.upper()
    except Exception:
        return False


# -----------------------
# Project paths
# -----------------------
@dataclass(frozen=True)
class ProjectPaths:
    """
    Centralized path registry. This is the "contract" for where the pipeline reads/writes.

    We intentionally support both:
      - data/export/...   (preferred, lowercase)
      - Data/Export/...   (legacy)

    The pipeline will use whichever exists; if neither exists, it will default to lowercase.
    """
    root: Path

    # ---- GPKG roots (input geometry)
    @property
    def gpkg_root(self) -> Path:
        # Prefer existing; otherwise default to lowercase convention
        candidates = [
            self.root / "data" / "export" / "gpkg",
            self.root / "Data" / "Export" / "gpkg",
        ]
        for d in candidates:
            if d.exists():
                return d
        # default (even if not exists yet)
        return candidates[0]

    @property
    def gpkg_exp_dir(self) -> Path:
        return self.gpkg_root / "exp"

    @property
    def gpkg_bdry_dir(self) -> Path:
        return self.gpkg_root / "bdry"

    # ---- Parquet roots (feature tables)
    @property
    def parquet_root(self) -> Path:
        candidates = [
            self.root / "data" / "export" / "parquet",
            self.root / "Data" / "Export" / "parquet",
        ]
        for d in candidates:
            if d.exists():
                return d
        return candidates[0]

    @property
    def parquet_exp_table_dir(self) -> Path:
        return self.parquet_root / "exp_table"

    @property
    def parquet_enriched_weather_dir(self) -> Path:
        return self.parquet_root / "enriched_weather"

    @property
    def parquet_enriched_topo_dir(self) -> Path:
        return self.parquet_root / "enriched_topo"

    @property
    def parquet_enriched_ssurgo_dir(self) -> Path:
        return self.parquet_root / "enriched_ssurgo"

    @property
    def parquet_model_ready_dir(self) -> Path:
        return self.parquet_root / "model_ready"

    @property
    def parquet_phase2_features_dir(self) -> Path:
        return self.parquet_root / "phase2_features"

    @property
    def phase3_out_dir(self) -> Path:
        return self.root / "data" / "export" / "phase3"

    # ---- Raster roots (topo rasters)
    @property
    def raster_root(self) -> Path:
        candidates = [
            self.root / "data" / "export" / "rasters",
            self.root / "Data" / "Export" / "rasters",
        ]
        for d in candidates:
            if d.exists():
                return d
        return candidates[0]

    @property
    def rasters_enriched_topo_dir(self) -> Path:
        return self.raster_root / "enriched_topo"

# -----------------------
# Listing / reading
# -----------------------
def list_ffy_ids(paths: ProjectPaths) -> list[str]:
    """
    List ffy_ids that exist in exp gpkg directory.
    """
    logger = get_logger()
    exp_dir = paths.gpkg_exp_dir
    exp_files = sorted(exp_dir.glob("*_exp.gpkg"))

    ffy_ids: list[str] = []
    for f in exp_files:
        ffy = parse_ffy_from_filename(f)
        if ffy:
            ffy_ids.append(ffy)

    ffy_ids = sorted(set(ffy_ids))
    logger.info(f"Found {len(ffy_ids)} ffy_ids in {exp_dir}")
    return ffy_ids


def build_file_paths(paths: ProjectPaths, ffy_id: str) -> Tuple[Path, Path, Path]:
    """
    Return (exp_gpkg, bdry_gpkg, exp_table_parquet) for one field-year.
    """
    if not is_valid_ffy_id(ffy_id):
        raise ValueError(f"Invalid ffy_id format: {ffy_id}")

    exp_gpkg = paths.gpkg_exp_dir / f"{ffy_id}_exp.gpkg"
    bdry_gpkg = paths.gpkg_bdry_dir / f"{ffy_id}_bdry.gpkg"
    exp_table = paths.parquet_exp_table_dir / f"{ffy_id}_exp_table.parquet"
    return exp_gpkg, bdry_gpkg, exp_table


def read_one_field(
    paths: ProjectPaths,
    ffy_id: str,
    exp_layer: str = "exp",
    bdry_layer: str = "bdry",
    expected_crs: str = "EPSG:4326",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """
    Read one ffy_id:
      - experimental polygons from GPKG layer 'exp'
      - boundary polygon from GPKG layer 'bdry'
      - tabular data from Parquet exp_table (yield/n_rate/s_rate)
    """
    logger = get_logger()
    exp_gpkg, bdry_gpkg, exp_table = build_file_paths(paths, ffy_id)

    # existence checks
    for p in (exp_gpkg, bdry_gpkg, exp_table):
        if not p.exists():
            raise FileNotFoundError(f"Missing file for {ffy_id}: {p}")

    logger.info(f"Reading exp polygons: {exp_gpkg.name} (layer={exp_layer})")
    exp_gdf = gpd.read_file(exp_gpkg, layer=exp_layer)

    logger.info(f"Reading boundary:     {bdry_gpkg.name} (layer={bdry_layer})")
    bdry_gdf = gpd.read_file(bdry_gpkg, layer=bdry_layer)

    logger.info(f"Reading exp table:    {exp_table.name}")
    exp_df = pd.read_parquet(exp_table)

    # CRS checks
    if exp_gdf.crs is None:
        logger.warning("exp_gdf CRS missing; setting to expected CRS.")
        exp_gdf = exp_gdf.set_crs(expected_crs)

    if bdry_gdf.crs is None:
        logger.warning("bdry_gdf CRS missing; setting to expected CRS.")
        bdry_gdf = bdry_gdf.set_crs(expected_crs)

    # normalize CRS 
    if not _crs_matches_expected(exp_gdf, expected_crs):
        logger.info("Reprojecting exp_gdf CRS %s -> %s", exp_gdf.crs, expected_crs)
        exp_gdf = exp_gdf.to_crs(expected_crs)

    if not _crs_matches_expected(bdry_gdf, expected_crs):
        logger.info("Reprojecting bdry_gdf CRS %s -> %s", bdry_gdf.crs, expected_crs)
        bdry_gdf = bdry_gdf.to_crs(expected_crs)

    # required schema
    required = {"yield", "n_rate"}
    missing = required - set(exp_df.columns)
    if missing:
        raise ValueError(f"[{ffy_id}] exp_table missing required columns: {sorted(missing)}")

    # seed optional
    if "s_rate" not in exp_df.columns:
        logger.warning(f"[{ffy_id}] exp_table has no s_rate column (seed optional).")

    # sanity checks
    if len(bdry_gdf) < 1:
        raise ValueError(f"[{ffy_id}] boundary has no rows")
    if len(exp_gdf) < 1:
        raise ValueError(f"[{ffy_id}] exp polygons has no rows")

    # IMPORTANT: row alignment (common in your pipeline)
    if len(exp_gdf) != len(exp_df):
        logger.warning(
            f"[{ffy_id}] Row mismatch: exp_gdf has {len(exp_gdf):,} polygons "
            f"but exp_df has {len(exp_df):,} rows. "
            "If you have an ID key (e.g., obs_id), we should merge by key instead of row order."
        )

    return exp_gdf, bdry_gdf, exp_df
