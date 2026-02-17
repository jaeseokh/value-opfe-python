#!/usr/bin/env python3

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray as rxr


def plot_ssurgo(ffy_id: str):
    gpkg = Path(f"data/export/gpkg/enriched_ssurgo/{ffy_id}_ssurgo.gpkg")
    if not gpkg.exists():
        raise FileNotFoundError(gpkg)

    gdf = gpd.read_file(gpkg, layer="ssurgo")

    print("\nSSURGO COLUMNS:")
    print(list(gdf.columns))

    # detect numeric columns (exclude geometry)
    numeric_cols = [
        c for c in gdf.columns
        if c != "geometry" and gdf[c].dtype.kind in "fi"
    ]

    print("\nSSURGO NUMERIC VARIABLES:")
    print(numeric_cols)

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 6))
        gdf.plot(column=col, ax=ax, legend=True)
        ax.set_title(f"{ffy_id} — SSURGO: {col}")
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()


def plot_topo(ffy_id: str):
    base = Path(f"data/export/rasters/enriched_topo/{ffy_id}")
    rasters = ["dem", "slope", "aspect", "twi"]

    for name in rasters:
        tif = base / f"{name}.tif"
        if not tif.exists():
            print("Missing:", tif)
            continue

        da = rxr.open_rasterio(tif).squeeze()

        print(f"\nTOPO VARIABLE: {name}")
        print("  CRS:", da.rio.crs)
        print("  Min/Max:", float(da.min()), float(da.max()))

        fig, ax = plt.subplots(figsize=(6, 6))
        da.plot(ax=ax)
        ax.set_title(f"{ffy_id} — TOPO: {name}")
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/plot_phase1_maps.py <ffy_id>")
        sys.exit(1)

    ffy_id = sys.argv[1]

    print("\n===== PHASE 1 SSURGO MAPS =====")
    plot_ssurgo(ffy_id)

    print("\n===== PHASE 1 TOPO MAPS =====")
    plot_topo(ffy_id)
