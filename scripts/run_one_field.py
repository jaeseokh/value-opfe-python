from __future__ import annotations

from pathlib import Path

from src.io import ProjectPaths, list_ffy_ids, read_one_field
from src.utils import get_logger


def main() -> None:
    logger = get_logger()

    project_root = Path(".").resolve()
    paths = ProjectPaths(root_dir=project_root)

    ffy_ids = list_ffy_ids(paths)
    if not ffy_ids:
        raise RuntimeError("No ffy_ids found. Check Data/Export/gpkg/exp contents.")

    # Pick the first one for a smoke test
    ffy_id = ffy_ids[0]
    logger.info(f"Smoke test ffy_id = {ffy_id}")

    exp_gdf, bdry_gdf, exp_df = read_one_field(paths, ffy_id)

    logger.info("---- Summary ----")
    logger.info(f"exp_gdf rows: {len(exp_gdf):,} | CRS: {exp_gdf.crs}")
    logger.info(f"bdry_gdf rows: {len(bdry_gdf):,} | CRS: {bdry_gdf.crs}")
    logger.info(f"exp_df rows:  {len(exp_df):,}")
    logger.info(f"exp_df columns (first 20): {list(exp_df.columns)[:20]}")

    logger.info("Yield / N summary:")
    logger.info(exp_df[["yield", "n_rate", "s_rate"]].describe().to_string())


if __name__ == "__main__":
    main()
