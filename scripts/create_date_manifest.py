from pathlib import Path
import pandas as pd

# Project root (run from repo root)
project_root = Path(".").resolve()

out_path = project_root / "Data" / "date_manifest.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

# ---- EDIT THESE MANUALLY ----
rows = [
    {
        "ffy_id": "10_1_2022",
        "s_time": "2022-04-20",
        "n_time": "2022-05-15",
        "yield_time": "2022-10-10",
    },
    # add more rows here
]

df = pd.DataFrame(rows)

df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")
print(df)
