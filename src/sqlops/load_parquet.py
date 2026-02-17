from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlalchemy as sa

from src.utils import get_logger


@dataclass(frozen=True)
class DBConfig:
    host: str = os.getenv("PGHOST", "localhost")
    port: int = int(os.getenv("PGPORT", "5432"))
    dbname: str = os.getenv("PGDATABASE", "ofpe_share")
    user: str = os.getenv("PGUSER", os.getenv("USER", "jaeseokhwang"))
    password: Optional[str] = os.getenv("PGPASSWORD", None)

    def sqlalchemy_url(self) -> str:
        if self.password:
            return (
                f"postgresql+psycopg2://{self.user}:{self.password}"
                f"@{self.host}:{self.port}/{self.dbname}"
            )
        return f"postgresql+psycopg2://{self.user}@{self.host}:{self.port}/{self.dbname}"


def _read_all_parquet(parquet_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(parquet_dir.glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def _read_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table file type: {path}")


def _resolve_phase3_simple_dir(root: Path) -> Path:
    """
    Resolve script-style Phase3 output directory.
    Priority:
      1) PHASE3_OUT_DIR (if set)
      2) data/export/phase3
    """
    env_dir = os.getenv("PHASE3_OUT_DIR")
    if env_dir:
        return Path(env_dir)
    return root / "data" / "export" / "phase3"


def _drop_all_views(engine: sa.Engine, schema: str = "public") -> None:
    """
    Drop all views so we can replace tables without 'dependent objects' errors.
    We recreate views in sql_build immediately after.
    """
    sql = sa.text(
        """
        DO $$
        DECLARE r record;
        BEGIN
          FOR r IN (
            SELECT table_schema, table_name
            FROM information_schema.views
            WHERE table_schema = :schema
          ) LOOP
            EXECUTE format('DROP VIEW IF EXISTS %I.%I CASCADE', r.table_schema, r.table_name);
          END LOOP;
        END $$;
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, {"schema": schema})


def _ensure_cols(df: pd.DataFrame, required: list[str], name: str) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}. Have: {list(df.columns)}")
    return df


def _add_field_id_from_ffy_id(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """
    If field_id is missing but ffy_id exists, create field_id = "<farm>_<field>"
    from ffy_id like "10_1_2022".
    """
    if df.empty:
        return df
    if "field_id" in df.columns:
        return df
    if "ffy_id" not in df.columns:
        raise ValueError(f"[{name}] Need field_id or ffy_id to construct field_id. Have: {list(df.columns)}")

    parts = df["ffy_id"].astype(str).str.split("_", expand=True)
    if parts.shape[1] < 2:
        raise ValueError(f"[{name}] ffy_id format unexpected (need at least 2 parts). Example: 10_1_2022")
    df = df.copy()
    df["field_id"] = parts[0].astype(str) + "_" + parts[1].astype(str)
    return df


def _load_replace(df: pd.DataFrame, engine: sa.Engine, table_name: str) -> None:
    logger = get_logger()
    if df.empty:
        logger.warning("[SKIP] %s: no rows", table_name)
        return
    logger.info("[LOAD] %s: %d rows", table_name, len(df))
    df.to_sql(
        table_name,
        engine,
        schema="public",
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=50_000,
    )


def run() -> None:
    """
    Phase1 DB load (reset-style):
      1) Drop views (CASCADE)
      2) Replace tables from parquet
      3) Run sql_build stage to recreate views
    """
    logger = get_logger()
    root = Path(".").resolve()
    pq_root = root / "data" / "export" / "parquet"

    model_ready_dir = pq_root / "model_ready"
    weather_dir = pq_root / "enriched_weather"
    ssurgo_dir = pq_root / "enriched_ssurgo"
    topo_dir = pq_root / "enriched_topo"
    phase3_dir = pq_root / "phase3_diagnostics"
    phase3_simple_dir = _resolve_phase3_simple_dir(root)

    engine = sa.create_engine(DBConfig().sqlalchemy_url())

    logger.info("[STEP] Dropping views (safe reset)")
    _drop_all_views(engine)

    # -------------------------
    # 1) ofpe_plot_obs (obs-level)
    # -------------------------
    ofpe = _read_all_parquet(model_ready_dir, "*_model_ready.parquet")
    ofpe = _ensure_cols(ofpe, ["year", "n_rate", "yield"], "ofpe_plot_obs")

    # field_id from ffy_id or (farm, field)
    if "ffy_id" in ofpe.columns:
        parts = ofpe["ffy_id"].astype(str).str.split("_", expand=True)
        ofpe["field_id"] = parts[0].astype(str) + "_" + parts[1].astype(str)
    elif "farm" in ofpe.columns and "field" in ofpe.columns:
        ofpe["field_id"] = ofpe["farm"].astype(str) + "_" + ofpe["field"].astype(str)
    else:
        raise ValueError("[ofpe_plot_obs] Need ffy_id or (farm, field) to build field_id")

    if "plot_id" not in ofpe.columns:
        if "obs_id" in ofpe.columns:
            ofpe["plot_id"] = ofpe["obs_id"].astype(str)
        else:
            ofpe["plot_id"] = pd.RangeIndex(len(ofpe)).astype(str)

    ofpe_out = ofpe[["field_id", "year", "plot_id", "n_rate", "yield"]].copy()
    _load_replace(ofpe_out, engine, "ofpe_plot_obs")

    # -------------------------
    # 2) weather_season (field-year aggregated)
    # -------------------------
    w = _read_all_parquet(weather_dir, "*_weather_table.parquet")
    if not w.empty:
        # make field_id, year from ffy_id
        if "ffy_id" in w.columns and ("field_id" not in w.columns or "year" not in w.columns):
            parts = w["ffy_id"].astype(str).str.split("_", expand=True)
            if parts.shape[1] < 3:
                raise ValueError("[weather_season] ffy_id format unexpected (need 3 parts like 10_1_2022)")
            w["field_id"] = parts[0].astype(str) + "_" + parts[1].astype(str)
            w["year"] = parts[2].astype(int)

        w = _ensure_cols(w, ["field_id", "year"], "weather_season")

        # rename python totals into SQL names
        if "prcp_t" in w.columns and "prcp_gs" not in w.columns:
            w = w.rename(columns={"prcp_t": "prcp_gs"})
        if "gdd_t" in w.columns and "gdd_gs" not in w.columns:
            w = w.rename(columns={"gdd_t": "gdd_gs"})
        if "edd_t" in w.columns and "edd_gs" not in w.columns:
            w = w.rename(columns={"edd_t": "edd_gs"})

        # optional columns if not computed
        if "heat_days_ge30c_gs" not in w.columns:
            w["heat_days_ge30c_gs"] = pd.NA
        if "tmax_gs_mean" not in w.columns:
            w["tmax_gs_mean"] = pd.NA
        if "tmin_gs_mean" not in w.columns:
            w["tmin_gs_mean"] = pd.NA

        keep = [
            "field_id",
            "year",
            "prcp_gs",
            "gdd_gs",
            "edd_gs",
            "tmax_gs_mean",
            "tmin_gs_mean",
            "heat_days_ge30c_gs",
        ]
        w_out = w[[c for c in keep if c in w.columns]].copy()
        _load_replace(w_out, engine, "weather_season")
    else:
        logger.warning("[SKIP] weather_season: no parquet rows")

    # -------------------------
    # 3) soil_ssurgo_field (field-level)
    # Your Phase1 SSURGO parquet is often just (ffy_id, ssurgo_status)
    # so we construct field_id and collapse to one row per field.
    # -------------------------
    s = _read_all_parquet(ssurgo_dir, "*_ssurgo_table.parquet")
    if not s.empty:
        s = _add_field_id_from_ffy_id(s, name="soil_ssurgo_field")

        # Collapse to field-level (choose OK if any OK; else first)
        if "ssurgo_status" in s.columns:
            s2 = (
                s.assign(_ok=(s["ssurgo_status"].astype(str).str.upper() == "OK").astype(int))
                .sort_values(["field_id", "_ok"], ascending=[True, False])
                .drop_duplicates("field_id")
                .drop(columns=["_ok"])
            )
            if "n_mukey" not in s2.columns and "n_unique_mukey" in s2.columns:
                s2["n_mukey"] = pd.to_numeric(s2["n_unique_mukey"], errors="coerce")
            keep = [
                "field_id",
                "dominant_mukey",
                "dom_share",
                "n_mukey",
                "hhi",
                "entropy",
                "ssurgo_status",
            ]
            for c in keep:
                if c not in s2.columns:
                    s2[c] = pd.NA
            s_out = s2[keep].copy()
        else:
            s2 = s[["field_id"]].drop_duplicates().copy()
            s2["dominant_mukey"] = pd.NA
            s2["dom_share"] = pd.NA
            s2["n_mukey"] = pd.NA
            s2["hhi"] = pd.NA
            s2["entropy"] = pd.NA
            s2["ssurgo_status"] = pd.NA
            s_out = s2

        _load_replace(s_out, engine, "soil_ssurgo_field")
    else:
        logger.warning("[SKIP] soil_ssurgo_field: no parquet rows")

    # -------------------------
    # 4) topo_field (field-level)
    # Depending on your writer, topo parquet may have ffy_id or field_id.
    # We'll support both and collapse to one row per field.
    # -------------------------
    t = _read_all_parquet(topo_dir, "*_topo_table.parquet")
    if not t.empty:
        t = _add_field_id_from_ffy_id(t, name="topo_field") if "field_id" not in t.columns else t

        keep = ["field_id", "elev", "slope", "aspect", "tpi", "elev_mean", "slope_mean", "aspect_mean", "tpi_mean"]
        for c in keep:
            if c not in t.columns:
                t[c] = pd.NA
        t2 = t[keep].copy()
        for c in keep:
            if c == "field_id":
                continue
            t2[c] = pd.to_numeric(t2[c], errors="coerce")
        t_out = t2.groupby("field_id", as_index=False)[keep[1:]].mean(numeric_only=True)
        for c in keep[1:]:
            if c not in t_out.columns:
                t_out[c] = pd.NA
        t_out = t_out[keep]

        _load_replace(t_out, engine, "topo_field")
    else:
        logger.warning("[SKIP] topo_field: no parquet rows")

    # -------------------------
    # 5) Phase3 diagnostics (optional, table-per-output)
    # -------------------------
    phase3_files = {
        "step1_lolo_rmse_by_location.parquet": "phase3_lolo_rmse_by_location",
        "step1_lolo_rmse_summary.parquet": "phase3_lolo_rmse_summary",
        "step2_eonr_by_field.parquet": "phase3_eonr_by_field",
        "step2_eonr_summary.parquet": "phase3_eonr_summary",
        "step3_mechanism_detail.parquet": "phase3_mechanism_detail",
        "step3_mechanism_summary.parquet": "phase3_mechanism_summary",
        "step4_assumptions_table.parquet": "phase3_assumptions_table",
    }
    if phase3_dir.exists():
        for fname, table_name in phase3_files.items():
            p = phase3_dir / fname
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            _load_replace(df, engine, table_name)
    else:
        logger.info("[SKIP] phase3_diagnostics: directory not found (%s)", phase3_dir)

    # -------------------------
    # 6) Phase3 outputs from scripts/3_00..3_04 (optional)
    # -------------------------
    phase3_simple_files = {
        "step1_predictions_obs.parquet": "phase3_step1_predictions_obs",
        "step2_eonr_gap.csv": "phase3_step2_eonr_gap",
        "step3_focus_feature_ranks.csv": "phase3_step3_focus_feature_ranks",
        "table_summary_step2.csv": "phase3_step4_summary_step2",
        "table_summary_step3_ranks.csv": "phase3_step4_summary_step3_ranks",
        "paper_tables/table_step4_assumptions.csv": "phase3_assumptions_table",
    }
    if phase3_simple_dir.exists():
        for fname, table_name in phase3_simple_files.items():
            p = phase3_simple_dir / fname
            if not p.exists():
                continue
            df = _read_tabular(p)
            _load_replace(df, engine, table_name)
    else:
        logger.info("[SKIP] phase3 simple outputs: directory not found (%s)", phase3_simple_dir)

    logger.info("[OK] DB tables loaded. Now run: python scripts/run_pipeline.py --stage sql_build")
