from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr

import py3dep
import rioxarray  # noqa: F401  needed for .rio
from scipy.ndimage import uniform_filter

from src.io import ProjectPaths, list_ffy_ids, read_one_field
from src.utils import ensure_dir, get_logger


@dataclass(frozen=True)
class TopoPhase1Config:
    dem_resolution_m: int = 10
    expand_bbox_deg: float = 0.002  # pad bbox to avoid edge artifacts
    overwrite: bool = False
    twi_window: int = 9  # proxy TWI neighborhood size


def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def _bbox_from_gdf(gdf_4326: gpd.GeoDataFrame, pad_deg: float) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = gdf_4326.total_bounds
    return (float(xmin - pad_deg), float(ymin - pad_deg), float(xmax + pad_deg), float(ymax + pad_deg))


def fetch_dem_for_boundary(bdry_gdf: gpd.GeoDataFrame, cfg: TopoPhase1Config) -> xr.DataArray:
    """
    Fetch 3DEP DEM covering the boundary bbox (EPSG:4326).
    Returns a DEM DataArray with a CRS set.
    """
    logger = get_logger()

    if bdry_gdf.crs is None:
        raise ValueError("Boundary CRS missing")

    bdry_4326 = bdry_gdf.to_crs("EPSG:4326")
    bbox = _bbox_from_gdf(bdry_4326, cfg.expand_bbox_deg)
    logger.info("Fetching 3DEP DEM bbox=%s res=%sm", bbox, cfg.dem_resolution_m)

    dem = None

    # robust to py3dep signature differences
    try:
        dem = py3dep.get_dem(bbox, resolution=cfg.dem_resolution_m)
    except TypeError as e:
        logger.warning("py3dep.get_dem(bbox, ...) failed: %s", e)

    if dem is None:
        geom = bdry_4326.geometry.unary_union
        try:
            dem = py3dep.get_dem(geometry=geom, resolution=cfg.dem_resolution_m)
        except TypeError as e:
            logger.warning("py3dep.get_dem(geometry=...) failed: %s", e)
            dem = py3dep.get_dem(geom=geom, resolution=cfg.dem_resolution_m)

    dem = dem.rename("dem")
    if not dem.rio.crs:
        dem = dem.rio.write_crs("EPSG:4326")
    return dem


def compute_slope_aspect_twi(
    dem_4326: xr.DataArray,
    bdry_gdf: gpd.GeoDataFrame,
    cfg: TopoPhase1Config,
) -> dict[str, xr.DataArray]:
    """
    Compute slope/aspect in UTM meters, then reproject to EPSG:5070.
    TWI is a fast/stable proxy (Phase 2 can replace with true flow-acc if desired).
    """
    logger = get_logger()

    bdry_4326 = bdry_gdf.to_crs("EPSG:4326")
    c = bdry_4326.geometry.unary_union.centroid
    utm = f"EPSG:{_utm_epsg_from_lonlat(float(c.x), float(c.y))}"
    logger.info("Reproject DEM to %s for slope/aspect", utm)

    dem_utm = dem_4326.rio.reproject(utm)

    # pixel size meters
    transform = dem_utm.rio.transform()
    x_res = abs(transform.a)
    y_res = abs(transform.e)

    z = dem_utm.values.astype("float64")

    # fill nodata safely
    if not np.isfinite(z).all():
        finite = np.isfinite(z)
        m = float(np.nanmean(z)) if finite.any() else 0.0
        z = np.where(finite, z, m)

    dz_dy, dz_dx = np.gradient(z, y_res, x_res)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = (np.degrees(aspect_rad) + 360) % 360

    slope = xr.DataArray(
        slope_deg, coords=dem_utm.coords, dims=dem_utm.dims, name="slope"
    ).rio.write_crs(utm)

    aspect = xr.DataArray(
        aspect_deg, coords=dem_utm.coords, dims=dem_utm.dims, name="aspect"
    ).rio.write_crs(utm)

    # TWI proxy
    w = int(cfg.twi_window)
    mean_win = uniform_filter(z, size=w, mode="nearest")
    relief = z - mean_win
    twi_proxy = np.log1p(np.maximum(relief, 0.0)) / (1.0 + np.maximum(slope_deg, 0.0))

    twi = xr.DataArray(
        twi_proxy, coords=dem_utm.coords, dims=dem_utm.dims, name="twi"
    ).rio.write_crs(utm)

    # unify CRS for storage
    target = "EPSG:5070"
    dem_5070 = dem_utm.rio.reproject(target)
    slope_5070 = slope.rio.reproject(target)
    aspect_5070 = aspect.rio.reproject(target)
    twi_5070 = twi.rio.reproject(target)

    return {"dem": dem_5070, "slope": slope_5070, "aspect": aspect_5070, "twi": twi_5070}


def clip_to_boundary(r: xr.DataArray, bdry_gdf: gpd.GeoDataFrame) -> xr.DataArray:
    bdry = bdry_gdf.to_crs(r.rio.crs)
    geom = [bdry.geometry.unary_union]
    return r.rio.clip(geom, bdry.crs, drop=True, all_touched=True)


def _write_topo_manifest(
    ffy_id: str, dem_tif: Path, slope_tif: Path, aspect_tif: Path, twi_tif: Path, out_parq: Path
) -> None:
    df = pd.DataFrame([{
        "ffy_id": ffy_id,
        "topo_status": "OK",
        "dem_tif": str(dem_tif),
        "slope_tif": str(slope_tif),
        "aspect_tif": str(aspect_tif),
        "twi_tif": str(twi_tif),
    }])
    df.to_parquet(out_parq, index=False)


def run_one_field(ffy_id: str, cfg: TopoPhase1Config | None = None) -> None:
    logger = get_logger()
    cfg = cfg or TopoPhase1Config()

    paths = ProjectPaths(Path("."))

    raster_dir = paths.rasters_enriched_topo_dir / ffy_id
    ensure_dir(raster_dir)

    out_parq_dir = paths.parquet_enriched_topo_dir
    ensure_dir(out_parq_dir)
    out_parq = out_parq_dir / f"{ffy_id}_topo_table.parquet"

    dem_tif = raster_dir / "dem.tif"
    slope_tif = raster_dir / "slope.tif"
    aspect_tif = raster_dir / "aspect.tif"
    twi_tif = raster_dir / "twi.tif"

    if (
        dem_tif.exists() and slope_tif.exists() and aspect_tif.exists() and twi_tif.exists()
        and (not cfg.overwrite)
    ):
        logger.info("[SKIP] TOPO rasters exist: %s", raster_dir)
        if not out_parq.exists():
            _write_topo_manifest(ffy_id, dem_tif, slope_tif, aspect_tif, twi_tif, out_parq)
        return

    exp_gdf, bdry_gdf, _exp_df = read_one_field(paths, ffy_id)
    logger.info("TOPO Phase1: ffy_id=%s | bdry rows=%d exp rows=%d", ffy_id, len(bdry_gdf), len(exp_gdf))

    dem = fetch_dem_for_boundary(bdry_gdf, cfg)
    layers = compute_slope_aspect_twi(dem, bdry_gdf, cfg)

    dem_c = clip_to_boundary(layers["dem"], bdry_gdf)
    slope_c = clip_to_boundary(layers["slope"], bdry_gdf)
    aspect_c = clip_to_boundary(layers["aspect"], bdry_gdf)
    twi_c = clip_to_boundary(layers["twi"], bdry_gdf)

    dem_c.rio.to_raster(dem_tif)
    slope_c.rio.to_raster(slope_tif)
    aspect_c.rio.to_raster(aspect_tif)
    twi_c.rio.to_raster(twi_tif)

    logger.info("[OK] TOPO rasters saved: %s", raster_dir)

    _write_topo_manifest(ffy_id, dem_tif, slope_tif, aspect_tif, twi_tif, out_parq)
    logger.info("[OK] TOPO parquet manifest saved: %s", out_parq)


def run_all_fields(*, overwrite: bool = False) -> None:
    logger = get_logger()
    paths = ProjectPaths(Path("."))

    cfg = TopoPhase1Config(overwrite=overwrite)

    ffy_ids = list_ffy_ids(paths)
    logger.info("TOPO Phase1: found %d ffy_ids", len(ffy_ids))

    ok = 0
    fail = 0
    for ffy_id in ffy_ids:
        try:
            run_one_field(ffy_id, cfg=cfg)
            ok += 1
        except Exception as e:
            fail += 1
            logger.exception("[FAIL] TOPO Phase1 ffy_id=%s | %s", ffy_id, e)

    logger.info("TOPO Phase1 finished. ok=%d fail=%d", ok, fail)
