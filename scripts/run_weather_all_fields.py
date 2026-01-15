from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.io import ProjectPaths, list_ffy_ids, read_one_field
from src.utils import get_logger, ensure_dir
from src.weather_daymet import WeatherConfig, build_weather_features_for_ffy


# ------------------
# helpers
# ------------------
def load_date_manifest(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # expected columns: ffy_id, s_time, n_time, yield_time
    return df


# ------------------
# main batch logic
# ------------------
logger = get_logger()

project_root = Path(".").resolve()
paths = ProjectPaths(root_dir=project_root)

out_dir = project_root / "Data" / "Export" / "parquet" / "enriched_weather"
ensure_dir(out_dir)

FORCE = True  # overwrite existing outputs

manifest_path = project_root / "Data" / "date_manifest.csv"
if not manifest_path.exists():
    raise FileNotFoundError(
        f"Missing {manifest_path}. Create it with columns: ffy_id,s_time,n_time,yield_time"
    )

manifest_df = load_date_manifest(manifest_path)
manifest_df["ffy_id"] = manifest_df["ffy_id"].astype(str)

manifest_map = {
    str(row["ffy_id"]): {
        "s_time": row.get("s_time"),
        "n_time": row.get("n_time"),
        "yield_time": row.get("yield_time"),
    }
    for _, row in manifest_df.iterrows()
}

ffy_ids = list_ffy_ids(paths)
logger.info(f"Found {len(ffy_ids)} ffy_ids in export folder")

n_ok = 0
n_skip = 0
n_fail = 0

for i, ffy_id in enumerate(ffy_ids, start=1):
    out_path = out_dir / f"{ffy_id}_weather_features.parquet"

    if out_path.exists() and not FORCE:
        logger.info(f"[{i}/{len(ffy_ids)}] already exists -> skipping {ffy_id}")
        n_skip += 1
        continue

    if ffy_id not in manifest_map:
        logger.warning(f"[{i}/{len(ffy_ids)}] No manifest row for {ffy_id} -> skipping")
        n_skip += 1
        continue

    try:
        _, bdry_gdf, _ = read_one_field(paths, ffy_id)
        date_row = manifest_map[ffy_id]

        cfg = WeatherConfig(apr_to_sep_only=True)
        weather_1row = build_weather_features_for_ffy(
            ffy_id, bdry_gdf, date_row, cfg=cfg
        )

        weather_1row.to_parquet(str(out_path), index=False, engine="fastparquet")
        logger.info(f"[{i}/{len(ffy_ids)}] OK -> {ffy_id}")
        n_ok += 1

    except Exception as e:
        logger.exception(f"[{i}/{len(ffy_ids)}] FAIL -> {ffy_id}: {e}")
        n_fail += 1

logger.info("---- Batch summary ----")
logger.info(f"OK:   {n_ok}")
logger.info(f"SKIP: {n_skip}")
logger.info(f"FAIL: {n_fail}")
