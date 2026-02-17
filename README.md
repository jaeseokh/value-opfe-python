# value_multifield_python

Public code repository for OFPE Phase 1-3 pipeline (feature engineering, transfer diagnostics, EONR evaluation).

## Data policy
- Raw/private data are **not** included.
- Large intermediate files (parquet/gpkg/rasters) are excluded via `.gitignore`.
- Public outputs for reporting are included in:
  - `outputs/figures/`
  - `outputs/tables/`

## Pipeline layout
- `src/etl/`: Phase 1-2 ETL modules
- `src/modeling/phase3/`: Phase 3 modeling/diagnostic modules
- `src/sqlops/`: parquet-to-Postgres loader and SQL apply helper
- `scripts/`: runnable entrypoints (`1_xx`, `2_xx`, `3_xx`)
- `sql/`: schema and analytics views (`00`-`04`)
- `conf/`: pipeline and project configuration

## Core outputs
- Phase 2 features:
  - `data/export/parquet/phase2_features/all_fields_features.parquet` (local, ignored)
- Phase 3 run directory (default):
  - `data/export/phase3/` (local, ignored)
- Shareable artifacts:
  - `outputs/figures/*.png`
  - `outputs/tables/*.csv`

## Typical run order
1. Phase 1 ETL:
   - `python scripts/1_01_run_weather.py`
   - `python scripts/1_02_run_topo.py`
   - `python scripts/1_03_run_ssurgo.py`
   - `python scripts/1_04_run_build_model_ready.py`
2. Phase 2 features:
   - `python scripts/2_01_run_build_spatial_features.py`
3. Phase 3 diagnostics:
   - `python scripts/3_00_prepare.py`
   - `python scripts/3_01_step1_train_constrained.py`
   - `python scripts/3_02_step2_eonr_gap.py`
   - `python scripts/3_03_step3_shap_stability.py`
   - `python scripts/3_04_reports.py`
4. SQL/Postgres:
   - `python scripts/3_06_run_phase3_load_postgres.py`
   - `bash scripts/3_05_build_phase3_sql_views.sh ofpe_share`
