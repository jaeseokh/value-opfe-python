from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd

from .utils import get_logger


FFY_PATTERN = re.compile(r"^\d+_\d+_\d{4}$")


@dataclass(frozen=True)
class ProjectPaths:
    """
    Standard project paths so scripts don't hardcode directories.
    """
    root_dir: Path

    @property
    def gpkg_exp_dir(self) -> Path:
        return self.root_dir / "Data" / "Export" / "gpkg" / "exp"

    @property
    def gpkg_bdry_dir(self) -> Path:
        return self.root_dir / "Data" / "Export" / "gpkg" / "bdry"

    @property
    def parquet_exp_table_dir(self) -> Path:
        return self.root_dir / "Data" / "Export" / "parquet" / "exp"


def is_valid_ffy_id(ffy_id: str) -> bool:
    return bool(FFY_PATTERN.match(ffy_id))


def parse_ffy_from_filename(path: Path) -> Optional[str]:
    """
    Extract ffy_id from file names like:
      1_1_2023_exp.gpkg
      1_1_2023_bdry.gpkg
      1_1_2023_exp_table.parquet
    """
    name = path.name
    # Try the common pattern: "<ffy_id>_something.ext"
    parts = name.split("_")
    if len(parts) < 4:
        return None
    candidate = "_".join(parts[:3])
    return candidate if is_valid_ffy_id(candidate) else None


def list_ffy_ids(paths: ProjectPaths) -> list[str]:
    """
    List ffy_ids that exist in exp gpkg directory.
    """
    logger = get_logger()
    if not paths.gpkg_exp_dir.exists():
        raise FileNotFoundError(f"Missing directory: {paths.gpkg_exp_dir}")

    exp_files = sorted(paths.gpkg_exp_dir.glob("*_exp.gpkg"))
    ffy_ids = []
    for f in exp_files:
        ffy = parse_ffy_from_filename(f)
        if ffy:
            ffy_ids.append(ffy)

    # unique and sorted
    ffy_ids = sorted(set(ffy_ids))
    logger.info(f"Found {len(ffy_ids)} ffy_ids in {paths.gpkg_exp_dir}")
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
      - tabular data from Parquet (no geometry)

    Returns: (exp_gdf, bdry_gdf, exp_df)
    """
    logger = get_logger()

    exp_gpkg, bdry_gpkg, exp_table = build_file_paths(paths, ffy_id)

    # Existence checks (finance managers love this)
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
        logger.warning("exp_gdf CRS is missing. Setting to expected CRS.")
        exp_gdf = exp_gdf.set_crs(expected_crs)

    if bdry_gdf.crs is None:
        logger.warning("bdry_gdf CRS is missing. Setting to expected CRS.")
        bdry_gdf = bdry_gdf.set_crs(expected_crs)

    # Normalize CRS to expected
    if str(exp_gdf.crs).upper() != expected_crs.upper():
        logger.info(f"Reprojecting exp_gdf CRS {exp_gdf.crs} -> {expected_crs}")
        exp_gdf = exp_gdf.to_crs(expected_crs)

    if str(bdry_gdf.crs).upper() != expected_crs.upper():
        logger.info(f"Reprojecting bdry_gdf CRS {bdry_gdf.crs} -> {expected_crs}")
        bdry_gdf = bdry_gdf.to_crs(expected_crs)

    # Minimal schema validation
    required_cols = {"yield", "n_rate", "s_rate"}
    missing = required_cols - set(exp_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in exp_table for {ffy_id}: {sorted(missing)}")

    # Lightweight sanity checks
    if bdry_gdf.shape[0] < 1:
        raise ValueError(f"Boundary has no rows for {ffy_id}")
    if exp_gdf.shape[0] < 1:
        raise ValueError(f"Experimental polygons have no rows for {ffy_id}")

    return exp_gdf, bdry_gdf, exp_df
