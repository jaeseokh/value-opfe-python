from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = "ofpe") -> logging.Logger:
    """
    Create a simple console logger.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers in notebooks / repeated runs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_dir(path: Path) -> None:
    """
    Create directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)
