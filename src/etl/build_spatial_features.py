# src/etl/phase2_build_features.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask

from src.io import ProjectPaths, list_ffy_ids, read_one_field
from src.utils import ensure_dir, get_logger


@dataclass(frozen=True)
class Phase2Config:
    overwrite: bool = False

    # keys / ids
    obs_id_col: str = "obs_id"

    # ssurgo gpkg is in Phase1
    ssurgo_layer: str = "ssurgo"

    # soil vars expected in SSURGO gpkg
    # (these will be output with the SAME names in phase2_features)
    ssurgo_vars: tuple[str, ...] = (
        "sandtotal_r",
        "silttotal_r",
        "claytotal_r",
        "awc_r",
        "om_r",
        "dbovendry_r",
    )

    # topo vars expected in Phase1 rasters
    topo_vars: tuple[str, ...] = ("dem", "slope", "aspect", "twi")


def _parse_ffy_id(ffy_id: str) -> dict[str, int]:
    # ffy_id like "10_1_2022"
    farm, field, year = ffy_id.split("_")
    return {"farm": int(farm), "field": int(field), "year": int(year)}


def _make_valid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Avoid GEOS crashes on overlay/intersection by ensuring valid geometries.
    Uses .make_valid() if available; otherwise uses buffer(0).
    """
    gdf = gdf.copy()
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def _phase2_out_dir(paths: ProjectPaths) -> Path:
    return paths.parquet_phase2_features_dir


def _phase2_out_path(paths: ProjectPaths, ffy_id: str) -> Path:
    return _phase2_out_dir(paths) / f"{ffy_id}_phase2_features.parquet"


def _get_topo_raster_dir(paths: ProjectPaths, ffy_id: str) -> Path:
    # Phase1 wrote rasters to: data/export/rasters/enriched_topo/<ffy_id>/
    return paths.rasters_enriched_topo_dir / ffy_id


def _get_ssurgo_gpkg_path(paths: ProjectPaths, ffy_id: str) -> Path:
    # Phase1 wrote SSURGO gpkg to: data/export/gpkg/enriched_ssurgo/<ffy_id>_ssurgo.gpkg
    return paths.gpkg_root / "enriched_ssurgo" / f"{ffy_id}_ssurgo.gpkg"


def _read_weather_table(paths: ProjectPaths, ffy_id: str) -> pd.DataFrame:
    # Phase1 weather parquet: data/export/parquet/enriched_weather/<ffy_id>_weather_table.parquet
    p = paths.parquet_enriched_weather_dir / f"{ffy_id}_weather_table.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Weather parquet not found: {p}")
    df = pd.read_parquet(p)
    if df.shape[0] != 1:
        df = df.head(1)
    return df


def _polygon_mean_from_raster(raster_path: Path, gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Compute mean raster value within each polygon using rasterio.mask.mask.
    Returns float array of length len(gdf), NaN if polygon has no valid pixels.
    """
    out = np.full((len(gdf),), np.nan, dtype="float64")

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        for i, geom in enumerate(gdf.geometry):
            try:
                data, _ = mask(
                    src,
                    [geom],
                    crop=True,
                    all_touched=True,
                    filled=True,
                    nodata=nodata,
                )
                arr = data[0].astype("float64")
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                out[i] = float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
            except Exception:
                out[i] = np.nan

    return out


def _build_topo_features(
    paths: ProjectPaths,
    ffy_id: str,
    exp_gdf: gpd.GeoDataFrame,
    cfg: Phase2Config,
) -> pd.DataFrame:
    """
    For each obs polygon: mean(dem/slope/aspect/twi) from Phase1 rasters.
    """
    logger = get_logger()

    raster_dir = _get_topo_raster_dir(paths, ffy_id)
    if not raster_dir.exists():
        raise FileNotFoundError(f"Topo raster dir not found: {raster_dir}")

    # Phase1 rasters are EPSG:5070
    exp_5070 = exp_gdf.to_crs("EPSG:5070")

    rows: dict[str, Any] = {cfg.obs_id_col: exp_5070[cfg.obs_id_col].to_numpy()}

    for v in cfg.topo_vars:
        rp = raster_dir / f"{v}.tif"
        if not rp.exists():
            raise FileNotFoundError(f"Missing topo raster: {rp}")

        logger.info("Phase2 TOPO: polygon means for %s (%s)", v, rp)
        rows[f"topo_{v}_mean"] = _polygon_mean_from_raster(rp, exp_5070)

    return pd.DataFrame(rows)


def _build_ssurgo_area_weighted(
    paths: ProjectPaths,
    ffy_id: str,
    exp_gdf: gpd.GeoDataFrame,
    cfg: Phase2Config,
) -> pd.DataFrame:
    """
    Area-weighted SSURGO properties per obs polygon.

    Output columns:
      - obs_id
      - sandtotal_r, silttotal_r, claytotal_r, awc_r, om_r, dbovendry_r
        (area-weighted mean over SSURGO polygons intersecting each obs polygon)

    If there is no overlap, soil columns will be NA (permitted).
    """
    logger = get_logger()

    gpkg = _get_ssurgo_gpkg_path(paths, ffy_id)
    if not gpkg.exists():
        raise FileNotFoundError(f"SSURGO gpkg not found: {gpkg}")

    ss = gpd.read_file(gpkg, layer=cfg.ssurgo_layer)

    missing = [c for c in cfg.ssurgo_vars if c not in ss.columns]
    if missing:
        raise ValueError(
            f"SSURGO gpkg missing vars: {missing}. Available: {list(ss.columns)}"
        )

    ss = ss[["geometry", *cfg.ssurgo_vars]].copy()

    # metric CRS for areas
    exp_5070 = exp_gdf.to_crs("EPSG:5070")
    ss_5070 = ss.to_crs("EPSG:5070")

    # validity guardrails
    exp_5070 = _make_valid(exp_5070)
    ss_5070 = _make_valid(ss_5070)

    logger.info("Phase2 SSURGO: intersect obs polygons with SSURGO polygons (area-weighted)")
    inter = gpd.overlay(
        exp_5070[[cfg.obs_id_col, "geometry"]],
        ss_5070,
        how="intersection",
        keep_geom_type=False,
    )

    # start with one row per obs_id (so we can left-merge results)
    out = (
        pd.DataFrame({cfg.obs_id_col: exp_5070[cfg.obs_id_col].to_numpy()})
        .drop_duplicates()
        .copy()
    )

    if inter.empty:
        for v in cfg.ssurgo_vars:
            out[v] = np.nan
        return out

    inter["__area"] = inter.geometry.area.astype("float64")

    for v in cfg.ssurgo_vars:
        x = pd.to_numeric(inter[v], errors="coerce")
        w = inter["__area"]

        # weighted mean per obs_id: sum(w*x)/sum(w)
        num = (w * x).groupby(inter[cfg.obs_id_col]).sum(min_count=1)
        den = w.groupby(inter[cfg.obs_id_col]).sum(min_count=1)
        aw = (num / den).rename(v)

        out = out.merge(aw.reset_index(), on=cfg.obs_id_col, how="left")

    logger.info("Phase2 SSURGO: done (area-weighted) | inter rows=%d", len(inter))
    return out


def build_one_field_features(ffy_id: str, cfg: Phase2Config | None = None) -> Path:
    logger = get_logger()
    cfg = cfg or Phase2Config()

    paths = ProjectPaths(Path("."))

    out_dir = _phase2_out_dir(paths)
    ensure_dir(out_dir)
    out_path = _phase2_out_path(paths, ffy_id)

    if out_path.exists() and (not cfg.overwrite):
        logger.info("[SKIP] Phase2 features exist: %s", out_path)
        return out_path

    logger.info("[RUN] Phase2 build features ffy_id=%s", ffy_id)

    # Phase1 base obs polygons + exp table
    exp_gdf, _bdry_gdf, exp_df = read_one_field(paths, ffy_id)

    if cfg.obs_id_col not in exp_df.columns:
        raise KeyError(f"Expected {cfg.obs_id_col} in exp table columns: {list(exp_df.columns)}")
    if cfg.obs_id_col not in exp_gdf.columns:
        raise KeyError(f"Expected {cfg.obs_id_col} in exp gpkg columns: {list(exp_gdf.columns)}")

    base = exp_df.copy()

    topo_df = _build_topo_features(paths, ffy_id, exp_gdf, cfg)
    ssurgo_df = _build_ssurgo_area_weighted(paths, ffy_id, exp_gdf, cfg)
    weather_df = _read_weather_table(paths, ffy_id)

    # Merge at obs level
    feat = (
        base
        .merge(topo_df, on=cfg.obs_id_col, how="left")
        .merge(ssurgo_df, on=cfg.obs_id_col, how="left")
    )

    # Broadcast weather scalars to all obs rows
    for c in weather_df.columns:
        if c == "ffy_id":
            continue
        feat[c] = weather_df.iloc[0][c]

    # attach IDs
    ids = _parse_ffy_id(ffy_id)
    feat["ffy_id"] = ffy_id
    feat["farm"] = ids["farm"]
    feat["field"] = ids["field"]
    feat["year"] = ids["year"]

    feat.to_parquet(out_path, index=False)
    logger.info("[OK] Phase2 features saved: %s | shape=%s", out_path, feat.shape)
    return out_path


def build_all_fields(*, overwrite: bool = False) -> Path:
    logger = get_logger()
    paths = ProjectPaths(Path("."))

    out_dir = _phase2_out_dir(paths)
    ensure_dir(out_dir)
    out_all = out_dir / "all_fields_features.parquet"

    ffy_ids = list_ffy_ids(paths)
    logger.info("Phase2: building features for %d ffy_ids", len(ffy_ids))

    parts = []
    ok, fail = 0, 0

    for ffy_id in ffy_ids:
        try:
            p = build_one_field_features(ffy_id, cfg=Phase2Config(overwrite=overwrite))
            parts.append(pd.read_parquet(p))
            ok += 1
        except Exception as e:
            fail += 1
            logger.exception("[FAIL] Phase2 ffy_id=%s | %s", ffy_id, e)

    if not parts:
        raise RuntimeError("Phase2 produced 0 successful fields")

    df = pd.concat(parts, axis=0, ignore_index=True)
    df.to_parquet(out_all, index=False)
    logger.info("[OK] Phase2 combined saved: %s | shape=%s | ok=%d fail=%d", out_all, df.shape, ok, fail)
    return out_all
