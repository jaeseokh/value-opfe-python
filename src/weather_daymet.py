from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
import geopandas as gpd
import pydaymet

from .utils import get_logger


@dataclass(frozen=True)
class WeatherConfig:
    # Growing degree day definition (match with R defaults)
    gdd_base_c: float = 10.0
    gdd_cap_c: float = 30.0
    edd_thresh_c: float = 30.0

    # Feature options
    apr_to_sep_only: bool = True
    heavy_rain_mm: float = 10.0
    dry_day_mm: float = 1e-6

    # Climatology windows (years, inclusive)
    clim_5y: int = 5
    clim_30y: int = 30

    # Daymet region (na = North America)
    region: str = "na"


def _parse_year_from_ffy_id(ffy_id: str) -> int:
    # ffy_id like "10_1_2022"
    return int(ffy_id.split("_")[-1])


def _centroid_lonlat(boundary_gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    if boundary_gdf.crs is None:
        raise ValueError("boundary_gdf CRS is missing")

    b4326 = boundary_gdf.to_crs("EPSG:4326")
    c = b4326.geometry.unary_union.centroid
    return float(c.x), float(c.y)


def _ensure_datetime(x: Optional[str | pd.Timestamp]) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    ts = pd.to_datetime(x)
    if pd.isna(ts):
        return None
    return ts


def _add_derived_daily_vars(df: pd.DataFrame, cfg: WeatherConfig) -> pd.DataFrame:
    """
    Expect columns: prcp (mm/day), tmin (C), tmax (C)
    Add: gdd, edd
    """
    out = df.copy()
    tavg = (out["tmin"] + out["tmax"]) / 2.0

    # GDD: max(min(tavg, cap) - base, 0)
    out["gdd"] = (tavg.clip(upper=cfg.gdd_cap_c) - cfg.gdd_base_c).clip(lower=0)

    # EDD: max(tmax - thresh, 0)
    out["edd"] = (out["tmax"] - cfg.edd_thresh_c).clip(lower=0)

    return out


def _filter_apr_sep(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Apr(4) ... Sep(9)."""
    return df[df["date"].dt.month.isin([4, 5, 6, 7, 8, 9])].copy()


def fetch_daymet_daily_point(
    lon: float,
    lat: float,
    start_date: str,
    end_date: str,
    cfg: WeatherConfig,
) -> pd.DataFrame:
    """
    Download daily Daymet for a single coordinate (lon, lat).
    Robust to pydaymet return formats across versions.

    Returns a DataFrame with columns: date, prcp, tmin, tmax
    """
    logger = get_logger()
    logger.info(
        f"Downloading Daymet daily at (lon={lon:.6f}, lat={lat:.6f}) dates=({start_date}, {end_date})"
    )

    df = pydaymet.get_bycoords(
        coords=(lon, lat),
        dates=(start_date, end_date),
        variables=["prcp", "tmin", "tmax"],
        region=cfg.region,
        time_scale="daily",
    )

    # ---- Normalize the "date" column robustly ----
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df["date"] = df.index
        df = df.reset_index(drop=True)
    else:
        df = df.reset_index()

        if "date" not in df.columns:
            if "time" in df.columns:
                df = df.rename(columns={"time": "date"})
            elif "index" in df.columns:
                df = df.rename(columns={"index": "date"})
            else:
                # last resort: find a datetime-like column
                for c in df.columns:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().mean() > 0.9:
                        df = df.rename(columns={c: "date"})
                        break
                else:
                    raise KeyError(
                        "Could not find a date column from pydaymet output. "
                        f"Columns={list(df.columns)}"
                    )

    df["date"] = pd.to_datetime(df["date"])

    # ---- Normalize variable names (handle units like 'prcp (mm/day)') ----
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map: Dict[str, str] = {}
    for c in df.columns:
        if c == "date":
            rename_map[c] = "date"
        elif c.startswith("prcp"):
            rename_map[c] = "prcp"
        elif c.startswith("tmin"):
            rename_map[c] = "tmin"
        elif c.startswith("tmax"):
            rename_map[c] = "tmax"

    df = df.rename(columns=rename_map)

    required = ["date", "prcp", "tmin", "tmax"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns {missing}. Columns={list(df.columns)}")

    df = df[required].copy()

    return df


def _dry_spell_max(prcp: pd.Series, dry_day_mm: float) -> int:
    """Longest run of consecutive dry days."""
    is_dry = (prcp <= dry_day_mm).astype(int).tolist()
    max_run = 0
    cur = 0
    for v in is_dry:
        if v == 1:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return int(max_run)


def _window_slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Inclusive start, inclusive end."""
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def compute_weather_features(
    daily: pd.DataFrame,
    s_time: Optional[pd.Timestamp],
    n_time: Optional[pd.Timestamp],
    y_time: Optional[pd.Timestamp],
    cfg: WeatherConfig,
) -> Dict[str, float | str]:
    """
    Compute a readable subset of  R features.

    Outputs:
      - seasonal totals (Apr–Sep if cfg.apr_to_sep_only)
      - post-N windows (7/15/30) precipitation, gdd, edd, heavy rain days
      - N-to-yield totals and dry spell metrics
      - stage windows S1..S4 (simple version)
    """
    out: Dict[str, float | str] = {}

    daily2 = _add_derived_daily_vars(daily, cfg)
    if cfg.apr_to_sep_only:
        daily2 = _filter_apr_sep(daily2)

    # Seasonal totals (Apr–Sep)
    out["prcp_t"] = float(daily2["prcp"].sum(skipna=True))
    out["gdd_t"] = float(daily2["gdd"].sum(skipna=True))
    out["edd_t"] = float(daily2["edd"].sum(skipna=True))

    # If we lack event dates, return seasonal totals only
    if s_time is None or n_time is None or y_time is None:
        out["n_app_stage"] = "NA"
        return out

    # Post-N windows: 7/15/30 days after N
    for k in (7, 15, 30):
        end_k = n_time + pd.Timedelta(days=k)
        seg = _window_slice(daily2, n_time + pd.Timedelta(days=1), end_k)
        out[f"precip_postN_d{k}"] = float(seg["prcp"].sum(skipna=True))
        out[f"gdd_postN_d{k}"] = float(seg["gdd"].sum(skipna=True))
        out[f"edd_postN_d{k}"] = float(seg["edd"].sum(skipna=True))
        out[f"heavy_rain_days_postN_d{k}"] = float((seg["prcp"] >= cfg.heavy_rain_mm).sum())

    # N to yield window
    seg_ny = _window_slice(daily2, n_time + pd.Timedelta(days=1), y_time)
    out["precip_N_to_yield"] = float(seg_ny["prcp"].sum(skipna=True))
    out["gdd_N_to_yield"] = float(seg_ny["gdd"].sum(skipna=True))
    out["edd_N_to_yield"] = float(seg_ny["edd"].sum(skipna=True))
    out["dry_days_N_to_yield"] = float((seg_ny["prcp"] <= cfg.dry_day_mm).sum())
    out["max_dry_spell_N_to_yield"] = float(_dry_spell_max(seg_ny["prcp"], cfg.dry_day_mm))
    out["heavy_rain_days_N_to_yield"] = float((seg_ny["prcp"] >= cfg.heavy_rain_mm).sum())

    # Stage windows (simple version similar to R logic)
    s1_end = s_time + pd.Timedelta(days=14)
    s2_end = n_time - pd.Timedelta(days=1)
    s3_end = n_time + pd.Timedelta(days=14)
    s4_end = y_time

    stages = {
        "S1": _window_slice(daily2, s_time, s1_end),
        "S2": _window_slice(daily2, s1_end + pd.Timedelta(days=1), s2_end)
        if s2_end >= s1_end else daily2.iloc[0:0],
        "S3": _window_slice(daily2, n_time, s3_end),
        "S4": _window_slice(daily2, s3_end + pd.Timedelta(days=1), s4_end)
        if s4_end >= s3_end else daily2.iloc[0:0],
    }

    for st, seg in stages.items():
        out[f"precip_{st}"] = float(seg["prcp"].sum(skipna=True))
        out[f"gdd_{st}"] = float(seg["gdd"].sum(skipna=True))
        out[f"edd_{st}"] = float(seg["edd"].sum(skipna=True))
        out[f"dry_days_{st}"] = float((seg["prcp"] <= cfg.dry_day_mm).sum())
        out[f"heavy_rain_days_{st}"] = float((seg["prcp"] >= cfg.heavy_rain_mm).sum())
        out[f"max_dry_spell_{st}"] = float(_dry_spell_max(seg["prcp"], cfg.dry_day_mm)) if len(seg) else 0.0

    out["days_to_N_app"] = float((n_time - s_time).days)
    out["precip_15_post_N"] = float(stages["S3"]["prcp"].sum(skipna=True))

    # Label similar to R note
    out["n_app_stage"] = "S1" if n_time <= s1_end else "S3_or_S4"

    return out


def compute_climatology_means(
    lon: float,
    lat: float,
    year: int,
    cfg: WeatherConfig,
) -> Dict[str, float]:
    """
    Compute 5y / 30y mean of daily prcp/gdd/edd over Apr–Sep (if cfg.apr_to_sep_only).

    Note: This is an expensive call (downloads multi-year daily series twice).
    We will cache later.
    """
    logger = get_logger()

    def _avg_n(n_years: int) -> Dict[str, float]:
        start_year = year - (n_years - 1)
        start_date = f"{start_year}-01-01"
        end_date = f"{year}-12-31"

        daily = fetch_daymet_daily_point(lon, lat, start_date, end_date, cfg)
        daily = _add_derived_daily_vars(daily, cfg)
        if cfg.apr_to_sep_only:
            daily = _filter_apr_sep(daily)

        return {
            f"prcp_{n_years}": float(daily["prcp"].mean(skipna=True)),
            f"gdd_{n_years}": float(daily["gdd"].mean(skipna=True)),
            f"edd_{n_years}": float(daily["edd"].mean(skipna=True)),
        }

    logger.info("Computing climatology means (5y and 30y)")
    out: Dict[str, float] = {}
    out.update(_avg_n(cfg.clim_5y))
    out.update(_avg_n(cfg.clim_30y))
    return out


def build_weather_features_for_ffy(
    ffy_id: str,
    boundary_gdf: gpd.GeoDataFrame,
    date_row: Dict[str, str],
    cfg: WeatherConfig = WeatherConfig(),
) -> pd.DataFrame:
    """
    Returns a 1-row DataFrame with weather features for this ffy_id.
    """
    logger = get_logger()

    year = _parse_year_from_ffy_id(ffy_id)
    lon, lat = _centroid_lonlat(boundary_gdf)

    daily = fetch_daymet_daily_point(
        lon=lon,
        lat=lat,
        start_date=f"{year}-01-01",
        end_date=f"{year}-12-31",
        cfg=cfg,
    )

    s_time = _ensure_datetime(date_row.get("s_time"))
    n_time = _ensure_datetime(date_row.get("n_time"))
    y_time = _ensure_datetime(date_row.get("yield_time"))

    feats = compute_weather_features(daily, s_time, n_time, y_time, cfg)
    clim = compute_climatology_means(lon, lat, year, cfg)

    row: Dict[str, float | str] = {"ffy_id": ffy_id, "lon": lon, "lat": lat}
    row.update(feats)
    row.update(clim)

    logger.info(f"Weather features built for {ffy_id}")
    return pd.DataFrame([row])
