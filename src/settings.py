from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import yaml

@dataclass
class ProjectConfig:
    db_name: str
    database_url_env: str
    outputs: Path
    figures: Path
    tables: Path
    logs: Path
    models: Path
    sql_files: list[str]

def load_project_config(project_yaml: str = "conf/project.yaml") -> ProjectConfig:
    with open(project_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    sql_files = cfg["sql"]["files"]
    proj = cfg["project"]

    return ProjectConfig(
        db_name=proj["db_name"],
        database_url_env=proj["database_url_env"],
        outputs=Path(paths["outputs"]),
        figures=Path(paths["figures"]),
        tables=Path(paths["tables"]),
        logs=Path(paths["logs"]),
        models=Path(paths["models"]),
        sql_files=sql_files,
    )

def get_database_url(cfg: ProjectConfig) -> str:
    url = os.getenv(cfg.database_url_env, "")
    if not url:
        # fallback: local db by name if env not set
        # (works with psql default local socket)
        return f"postgresql+psycopg2://localhost/{cfg.db_name}"
    return url
