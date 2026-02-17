from __future__ import annotations
from pathlib import Path
import logging
import logging.config
import yaml

def setup_logging(logging_yaml: str = "conf/logging.yaml") -> None:
    with open(logging_yaml, "r") as f:
        config = yaml.safe_load(f)
    # ensure log dir exists even before handler opens file
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(config)

def ensure_output_dirs() -> None:
    for p in ["outputs/figures", "outputs/tables", "outputs/logs", "outputs/models"]:
        Path(p).mkdir(parents=True, exist_ok=True)
