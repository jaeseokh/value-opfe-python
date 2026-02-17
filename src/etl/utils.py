from __future__ import annotations

# ETL modules currently import from ".utils"
# but the canonical logger lives in src/utils.py.
# This shim keeps your ETL imports stable.

from src.utils import get_logger  # re-export
