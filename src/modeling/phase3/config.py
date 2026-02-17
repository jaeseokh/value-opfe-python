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
    # input / output
    data_path: Path = Path(
        os.getenv(
            "PHASE3_DATA_PATH",
            "data/export/parquet/phase2_features/all_fields_features.parquet",
        )
    )
    out_dir: Path = Path(os.getenv("PHASE3_OUT_DIR", "data/export/phase3"))

    # profit params
    price_yield_per_bu: float = 5.00
    price_n_per_lb: float = 0.60

    # N grid (lbs/acre)
    n_min: float = 0.0
    n_max: float = 250.0
    n_step: float = 5.0

    # core columns
    y_col: str = "yield"
    n_col: str = "n_rate"
    seed_col: str = "s_rate"

    # LOLO settings
    # Default: hold out by full field-year (ffy_id), not pooled field across years.
    # Override with PHASE3_LOLO_BY_FIELD_ONLY=1 if you want farm_field grouping.
    lolo_by_field_only: bool = _env_bool("PHASE3_LOLO_BY_FIELD_ONLY", False)
    seed: int = 123
