from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.io import ProjectPaths, list_ffy_ids, read_one_field
from src.utils import get_logger, ensure_dir


def main() -> None:
    logger = get_logger()

    project_root = Path(".").resolve()
    paths = ProjectPaths(root_dir=project_root)

    out_dir = project_root / "outputs" / "figures"
    ensure_dir(out_dir)

    ffy_ids = list_ffy_ids(paths)
    if not ffy_ids:
        raise RuntimeError("No ffy_ids found. Check Data/Export/gpkg/exp contents.")

    # Use the same logic as smoke test: first ffy_id
    ffy_id = ffy_ids[0]
    logger.info(f"Plotting ffy_id = {ffy_id}")

    exp_gdf, bdry_gdf, exp_df = read_one_field(paths, ffy_id)

    # ----------------------------
    # Plot 1: Map colored by yield
    # ----------------------------
    fig, ax = plt.subplots(figsize=(9, 9))

    # Boundary outline first
    bdry_gdf.boundary.plot(ax=ax, linewidth=2)

    # Experimental polygons shaded by yield
    exp_gdf.plot(column="yield", ax=ax, legend=True)

    ax.set_title(f"Experimental polygons (Yield) — {ffy_id}")
    ax.set_axis_off()
    plt.tight_layout()

    map_path = out_dir / f"{ffy_id}_yield_map.png"
    plt.savefig(map_path, dpi=200)
    logger.info(f"Saved map: {map_path}")
    plt.show()

    # ----------------------------
    # Plot 2: Yield vs N-rate scatter
    # ----------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(exp_df["n_rate"], exp_df["yield"], alpha=0.6)

    ax.set_title(f"Yield vs N-rate — {ffy_id}")
    ax.set_xlabel("N rate")
    ax.set_ylabel("Yield")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    plt.tight_layout()

    scat_path = out_dir / f"{ffy_id}_yield_vs_nrate.png"
    plt.savefig(scat_path, dpi=200)
    logger.info(f"Saved scatter: {scat_path}")
    plt.show()


if __name__ == "__main__":
    main()
