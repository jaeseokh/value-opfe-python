# scripts/fit_eonr_gam.py
from __future__ import annotations

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from pygam import LinearGAM, s

# -------------------------------------------------------------------
# project imports
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corn-price", type=float, default=5.0)
    ap.add_argument("--n-price", type=float, default=0.8)
    ap.add_argument("--n-grid", type=int, default=100)
    ap.add_argument("--out-name", default="eonr_gam.parquet")
    args = ap.parse_args()

    logger = get_logger()

    data_path = PROJECT_ROOT / "data" / "export" / "parquet" / "model_ready" / "analysis_df.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing analysis_df: {data_path}")

    logger.info(f"Loading analysis_df: {data_path}")
    df = pd.read_parquet(data_path)

    out_dir = PROJECT_ROOT / "data" / "export" / "parquet" / "eonr"
    ensure_dir(out_dir)

    results = []

    for ffy_id, g in df.groupby("ffy_id"):
        g = g.sort_values("n_rate").copy()

        if g["n_rate"].nunique() < 5:
            logger.warning(f"[{ffy_id}] too few N levels — skip")
            continue

        X = g[["n_rate"]].values
        y = g["yield"].values

        try:
            gam = LinearGAM(s(0)).fit(X, y)
        except Exception as e:
            logger.warning(f"[{ffy_id}] GAM failed: {e}")
            continue

        n_min, n_max = g["n_rate"].min(), g["n_rate"].max()
        n_grid = np.linspace(n_min, n_max, args.n_grid)

        y_hat = gam.predict(n_grid)

        profit = (
            args.corn_price * y_hat
            - args.n_price * n_grid
        )

        idx_star = np.argmax(profit)

        for i, N in enumerate(n_grid):
            results.append(
                {
                    "ffy_id": ffy_id,
                    "model": "GAM_in_field",
                    "n_rate": N,
                    "yield_hat": y_hat[i],
                    "profit": profit[i],
                    "corn_price": args.corn_price,
                    "n_price": args.n_price,
                    "is_eonr": i == idx_star,
                }
            )

    out = pd.DataFrame(results)

    out_path = out_dir / args.out_name
    out.to_parquet(out_path, index=False)

    logger.info(f"Saved EONR GAM results: {out_path}")
    logger.info(f"Fields processed: {out['ffy_id'].nunique()}")

if __name__ == "__main__":
    main()
