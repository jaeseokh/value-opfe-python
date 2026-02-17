from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.io import ProjectPaths, list_ffy_ids, read_one_field
from src.utils import get_logger, ensure_dir
from src.weather_daymet import WeatherConfig, build_weather_features_for_ffy


def load_date_manifest(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"ffy_id", "s_time", "n_time", "yield_time"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"date_manifest.csv missing columns: {sorted(missing)}")
    return df


def main() -> None:
    logger = get_logger()

    project_root = Path(".").resolve()
    paths = ProjectPaths(root_dir=project_root)

    out_dir = project_root / "Data" / "Export" / "parquet" / "enriched_weather"
    ensure_dir(out_dir)

    manifest_path = project_root / "Data" / "date_manifest.csv"
    manifest_df = load_date_manifest(manifest_path)

    ffy_ids = list_ffy_ids(paths)
    if not ffy_ids:
        raise RuntimeError("No ffy_ids found.")

    ffy_id = ffy_ids[0]
    logger.info(f"Running weather enrichment for ffy_id={ffy_id}")

    # read boundary (needed for centroid lon/lat)
    _, bdry_gdf, _ = read_one_field(paths, ffy_id)

    row = manifest_df.loc[manifest_df["ffy_id"] == ffy_id]
    if row.empty:
        raise RuntimeError(f"No date manifest entry for ffy_id={ffy_id}")

    date_row = row.iloc[0].to_dict()

    cfg = WeatherConfig(apr_to_sep_only=True)
    weather_1row = build_weather_features_for_ffy(ffy_id, bdry_gdf, date_row, cfg=cfg)

    out_path = out_dir / f"{ffy_id}_weather_features.parquet"
    weather_1row.to_parquet(out_path, index=False, engine="fastparquet")

    logger.info(f"Saved: {out_path}")
    logger.info("---- Weather feature preview ----")
    logger.info(weather_1row.T.head(30).to_string())


if __name__ == "__main__":
    main()
