from __future__ import annotations

import logging
import subprocess

from src.settings import load_project_config, get_database_url

logger = logging.getLogger(__name__)


def run() -> None:
    cfg = load_project_config()

    # Prefer DATABASE_URL (or whatever env is configured)
    db_url = get_database_url(cfg)

    # If URL is the default SQLAlchemy-style local url, psql won't like it.
    # In that fallback case, we use "-d cfg.db_name".
    use_dbname = db_url.startswith("postgresql+psycopg2://localhost/")

    for f in cfg.sql_files:
        logger.info("Applying SQL file: %s", f)

        if use_dbname:
            cmd = ["psql", "-d", cfg.db_name, "-f", f]
        else:
            # psql accepts postgres://... or postgresql://...
            cleaned = db_url.replace("postgresql+psycopg2://", "postgresql://")
            cmd = ["psql", cleaned, "-f", f]

        subprocess.run(cmd, check=True)

    logger.info("SQL build complete.")
