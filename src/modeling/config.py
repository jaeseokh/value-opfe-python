from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

@dataclass(frozen=True)
class Phase3Config:
    data_path: Path = Path("data/export/parquet/phase2_features/all_fields_features.parquet")
    out_dir: Path = Path("data/export/phase3")

    # Profit params (set these to your thesis values)
    price_yield_per_bu: float = 5.00
    price_n_per_lb: float = 0.60

    # N grid for EONR optimization (lbs/acre)
    n_min: float = 0.0
    n_max: float = 250.0
    n_step: float = 5.0

    # CV grouping: LOLO at "field-location" level
    group_col: str = "ffy_id"  # can also use "field_location" if you define it

    # core columns
    y_col: str = "yield"
    n_col: str = "n_rate"
    seed_col: str = "s_rate"

    # If you want strict LOLO by "field id" (ignoring year), set this True
    # Default is field-year LOLO; set PHASE3_LOLO_BY_FIELD_ONLY=1 for field-only LOLO.
    lolo_by_field_only: bool = _env_bool("PHASE3_LOLO_BY_FIELD_ONLY", False)
