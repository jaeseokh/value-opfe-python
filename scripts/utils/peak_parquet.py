# scripts/peek_parquet.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Parquet file path")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--missing", action="store_true")
    ap.add_argument("--cols", type=int, default=40)
    ap.add_argument("--rows", type=int, default=5)
    args = ap.parse_args()

    p = Path(args.path)
    df = pd.read_parquet(p)

    print(f"\nFILE: {p}")
    print(f"SHAPE: {df.shape}")
    print("\nCOLUMNS:")
    cols = df.columns.tolist()
    print(cols[:args.cols], ("..." if len(cols) > args.cols else ""))
    print("\nDTYPES (top):")
    print(df.dtypes.head(30))

    print("\nHEAD:")
    print(df.head(args.rows).to_string(index=False))

    if args.missing:
        print("\nMISSINGNESS (top 30):")
        miss = df.isna().mean().sort_values(ascending=False).head(30)
        print(miss.to_string())

    if args.describe:
        num = df.select_dtypes(include="number")
        if num.shape[1] == 0:
            print("\nDESCRIBE (numeric): [none] (no numeric columns)")
        else:
            print("\nDESCRIBE (numeric):")
            print(num.describe().transpose().head(30).to_string())

if __name__ == "__main__":
    main()
