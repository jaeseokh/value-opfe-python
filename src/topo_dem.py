from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import geopandas as gpd
import xarray as xr
from scipy.ndimage import uniform_filter

from rasterstats import zonal_stats

import py3dep
import rioxarray  # noqa: F401  (needed for .rio accessor)

from .utils import get_logger


@dataclass(frozen=True)
class TopoConfig:
    """
    Configuration for topography extraction.
    """
    dem_resolution_m: int = 10  # 3DEP supports various resolutions; 10m is usually fine
    expand_bbox_deg: float = 0.002  # expand bbox slightly to avoid edge effects
    nodata: float = np.nan


def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """
    Compute a reasonable UTM EPSG code from lon/lat.
    UTM is good for slope/aspect because units are meters.
    """
    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        return 32600 + zone  # WGS84 / UTM northern hemisphere
    return 32700 + zone      # WGS84 / UTM southern hemisphere


def _bbox_from_gdf(gdf_4326: gpd.GeoDataFrame, pad_deg: float) -> Tuple[float, float, float, float]:
    """
    Return (xmin, ymin, xmax, ymax) with padding in degrees.
    Assumes CRS is EPSG:4326.
    """
    xmin, ymin, xmax, ymax = gdf_4326.total_bounds
    return (
        float(xmin - pad_deg),
        float(ymin - pad_deg),
        float(xmax + pad_deg),
        float(ymax + pad_deg),
    )


def fetch_dem_for_boundary(
    boundary_gdf: gpd.GeoDataFrame,
    cfg: TopoConfig = TopoConfig(),
) -> xr.DataArray:
    """
    Fetch 3DEP DEM as an xarray DataArray for the boundary bbox.

    Robust across py3dep versions:
    - Some versions accept get_dem(bbox, resolution=...)
    - Some accept get_dem(geometry=..., resolution=...)
    """
    logger = get_logger()

    if boundary_gdf.crs is None:
        raise ValueError("boundary_gdf CRS is missing")

    boundary_4326 = boundary_gdf.to_crs("EPSG:4326")
    bbox = _bbox_from_gdf(boundary_4326, cfg.expand_bbox_deg)
    logger.info(f"Fetching 3DEP DEM for bbox={bbox} at res={cfg.dem_resolution_m}m")

    dem = None

    # Try bbox-style first (positional bbox avoids keyword mismatch)
    try:
        dem = py3dep.get_dem(bbox, resolution=cfg.dem_resolution_m)
    except TypeError as e:
        logger.warning(f"py3dep.get_dem(bbox, ...) failed: {e}")

    # Fallback: geometry-style (some versions use 'geometry' or 'geom')
    if dem is None:
        geom = boundary_4326.geometry.unary_union
        try:
            dem = py3dep.get_dem(geometry=geom, resolution=cfg.dem_resolution_m)
        except TypeError as e:
            logger.warning(f"py3dep.get_dem(geometry=...) failed: {e}")
            try:
                dem = py3dep.get_dem(geom=geom, resolution=cfg.dem_resolution_m)
            except TypeError as e2:
                raise TypeError(
                    "py3dep.get_dem API mismatch. Tried bbox positional, geometry=, geom=. "
                    f"Last error: {e2}"
                ) from e2

    dem = dem.rename("elev")

    if not dem.rio.crs:
        dem = dem.rio.write_crs("EPSG:4326")

    return dem


def compute_terrain_layers(
    dem_4326: xr.DataArray,
    boundary_gdf: gpd.GeoDataFrame,
) -> Dict[str, xr.DataArray]:
    """
    Compute terrain layers (elev, slope, aspect, tpi).

    Strategy:
    - Reproject DEM to a UTM CRS (meters) for slope/aspect computations
    - Compute:
        - slope (degrees) using gradient in projected meters
        - aspect (degrees) from gradients
        - TPI as elevation minus local mean elevation (uniform_filter)
    """
    logger = get_logger()

    boundary_4326 = boundary_gdf.to_crs("EPSG:4326")
    centroid = boundary_4326.geometry.unary_union.centroid
    lon, lat = float(centroid.x), float(centroid.y)

    utm_epsg = _utm_epsg_from_lonlat(lon, lat)
    utm_crs = f"EPSG:{utm_epsg}"
    logger.info(f"Reprojecting DEM to {utm_crs} for terrain metrics")

    dem = dem_4326.rio.reproject(utm_crs)

    # Elevation layer
    elev = dem

    # Pixel size in meters from affine transform
    transform = dem.rio.transform()
    x_res = abs(transform.a)
    y_res = abs(transform.e)

    z = dem.values.astype("float64")

    # gradient: dz/dx, dz/dy
    dz_dy, dz_dx = np.gradient(z, y_res, x_res)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    # aspect: 0-360
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = (np.degrees(aspect_rad) + 360) % 360

    slope = xr.DataArray(
        slope_deg,
        coords=dem.coords,
        dims=dem.dims,
        name="slope",
        attrs={"units": "degrees"},
    ).rio.write_crs(utm_crs)

    aspect = xr.DataArray(
        aspect_deg,
        coords=dem.coords,
        dims=dem.dims,
        name="aspect",
        attrs={"units": "degrees"},
    ).rio.write_crs(utm_crs)

    # TPI: cell - mean(neighborhood)
    window = 9  # tune later if needed
    mean_win = uniform_filter(z, size=window, mode="nearest")
    tpi_val = z - mean_win

    tpi = xr.DataArray(
        tpi_val,
        coords=dem.coords,
        dims=dem.dims,
        name="tpi",
        attrs={"units": "m"},
    ).rio.write_crs(utm_crs)

    layers = {"elev": elev, "slope": slope, "aspect": aspect, "tpi": tpi}
    logger.info("Terrain layers computed: elev, slope, aspect, tpi")
    return layers


def area_weighted_extract(
    polygons_gdf: gpd.GeoDataFrame,
    raster: xr.DataArray,
    column_name: str,
) -> np.ndarray:
    """
    Polygon mean extraction using rasterstats.zonal_stats.

    Note:
    - Computes mean over raster cells intersecting polygon (stable, widely used)
    - Not exact area-weighted mean; can be upgraded later if needed
    """
    logger = get_logger()

    if polygons_gdf.crs is None:
        raise ValueError("polygons_gdf CRS is missing")
    if not raster.rio.crs:
        raise ValueError("Raster CRS is missing")

    polys = polygons_gdf.to_crs(raster.rio.crs)

    arr = raster.values.astype("float32")
    transform = raster.rio.transform()

    logger.info(f"Zonal mean extracting: {column_name} for {len(polys):,} polygons")

    zs = zonal_stats(
        vectors=polys.geometry,
        raster=arr,
        affine=transform,
        stats=["mean"],
        nodata=None,
        all_touched=True,
    )

    out = np.array([d.get("mean", np.nan) if d else np.nan for d in zs], dtype="float64")
    return out


def enrich_with_topography(
    exp_gdf: gpd.GeoDataFrame,
    bdry_gdf: gpd.GeoDataFrame,
    cfg: TopoConfig = TopoConfig(),
) -> gpd.GeoDataFrame:
    """
    Add topography columns to the experimental polygons:
      elev_mean, slope_mean, aspect_mean, tpi_mean
    """
    logger = get_logger()

    dem = fetch_dem_for_boundary(bdry_gdf, cfg=cfg)
    layers = compute_terrain_layers(dem, bdry_gdf)

    elev_mean = area_weighted_extract(exp_gdf, layers["elev"], "elev_mean")
    slope_mean = area_weighted_extract(exp_gdf, layers["slope"], "slope_mean")
    aspect_mean = area_weighted_extract(exp_gdf, layers["aspect"], "aspect_mean")
    tpi_mean = area_weighted_extract(exp_gdf, layers["tpi"], "tpi_mean")

    out = exp_gdf.copy()
    out["elev_mean"] = elev_mean
    out["slope_mean"] = slope_mean
    out["aspect_mean"] = aspect_mean
    out["tpi_mean"] = tpi_mean

    logger.info("Topography enrichment complete")
    return out
