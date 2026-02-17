from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def parquet_count(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern)))


def parquet_preview_any(path: Path, pattern: str, n=3) -> None:
    files = sorted(path.glob(pattern))
    if not files:
        print(f"[WARN] No parquet files: {path}/{pattern}")
        return
    p = files[0]
    df = pd.read_parquet(p)
    print(f"\n[PARQUET PREVIEW] {p}")
    print("shape:", df.shape)
    print("cols:", df.columns.tolist()[:60])
    print(df.head(n))


def main() -> None:
    root = Path(".").resolve()
    pq_root = root / "data" / "export" / "parquet"

    model_ready_dir = pq_root / "model_ready"
    weather_dir = pq_root / "enriched_weather"
    ssurgo_dir = pq_root / "enriched_ssurgo"
    topo_dir = pq_root / "enriched_topo"

    print("\n=== FILE CHECKS ===")
    print("model_ready:", parquet_count(model_ready_dir, "*_model_ready.parquet"))
    print("weather:", parquet_count(weather_dir, "*_weather_table.parquet"))
    print("ssurgo:", parquet_count(ssurgo_dir, "*_ssurgo_table.parquet"))
    print("topo:", parquet_count(topo_dir, "*_topo_table.parquet"))

    parquet_preview_any(model_ready_dir, "*_model_ready.parquet")
    parquet_preview_any(weather_dir, "*_weather_table.parquet")
    parquet_preview_any(ssurgo_dir, "*_ssurgo_table.parquet")
    parquet_preview_any(topo_dir, "*_topo_table.parquet")

    print("\n=== DB CHECKS ===")
    db = os.getenv("PGDATABASE", "ofpe_share")
    sh("pg_isready")
    sh(f'psql -d {db} -c "\\dt"')
    sh(f'psql -d {db} -c "\\dv"')

    print("\n=== ROW COUNTS (core tables) ===")
    sh(
        f"""psql -d {db} -c "
        SELECT 'ofpe_plot_obs' AS table, COUNT(*) FROM ofpe_plot_obs
        UNION ALL SELECT 'weather_season', COUNT(*) FROM weather_season
        UNION ALL SELECT 'soil_ssurgo_field', COUNT(*) FROM soil_ssurgo_field
        UNION ALL SELECT 'topo_field', COUNT(*) FROM topo_field;
        " """
    )

    print("\n=== VIEW OUTPUT CHECKS ===")
    sh(f'psql -d {db} -c "SELECT COUNT(*) AS n FROM vw_model_features;"')
    sh(f'psql -d {db} -c "SELECT * FROM vw_feature_coverage ORDER BY year LIMIT 30;"')
    sh(f'psql -d {db} -c "SELECT * FROM vw_regime_support;"')

    print("\n[OK] Phase1 smoke test complete.")


if __name__ == "__main__":
    main()
