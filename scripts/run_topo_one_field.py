from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.io import ProjectPaths, list_ffy_ids, read_one_field
from src.topo_dem import TopoConfig, enrich_with_topography
from src.utils import get_logger, ensure_dir


def main() -> None:
    logger = get_logger()

    project_root = Path(".").resolve()
    paths = ProjectPaths(root_dir=project_root)

    out_dir = project_root / "Data" / "Export" / "parquet" / "enriched_topo"
    ensure_dir(out_dir)

    ffy_ids = list_ffy_ids(paths)
    if not ffy_ids:
        raise RuntimeError("No ffy_ids found.")

    ffy_id = ffy_ids[0]
    logger.info(f"Running topo enrichment for ffy_id={ffy_id}")

    exp_gdf, bdry_gdf, exp_df = read_one_field(paths, ffy_id)

    # Add topo columns to polygons
    cfg = TopoConfig(dem_resolution_m=10)
    exp_topo_gdf = enrich_with_topography(exp_gdf, bdry_gdf, cfg=cfg)

    # Join topo columns back to exp_df by row order
    topo_cols = ["elev_mean", "slope_mean", "aspect_mean", "tpi_mean"]
    topo_df = exp_topo_gdf[topo_cols].copy()

    enriched_df = pd.concat(
        [exp_df.reset_index(drop=True), topo_df.reset_index(drop=True)],
        axis=1,
    )

    out_path = out_dir / f"{ffy_id}_topo_enriched.parquet"

    enriched_df.to_parquet(str(out_path), index=False, engine="fastparquet")

    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
