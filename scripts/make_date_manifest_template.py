from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.io import ProjectPaths, list_ffy_ids

def parse_year(ffy_id: str) -> int:
    return int(ffy_id.split("_")[-1])

def main() -> None:
    project_root = Path(".").resolve()
    paths = ProjectPaths(root_dir=project_root)

    ffy_ids = list_ffy_ids(paths)
    rows = []
    for ffy_id in ffy_ids:
        y = parse_year(ffy_id)
        rows.append(
            {
                "ffy_id": ffy_id,
                "s_time": f"{y}-04-20",
                "n_time": f"{y}-05-15",
                "yield_time": f"{y}-10-10",
            }
        )

    out_path = project_root / "Data" / "date_manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values("ffy_id")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
