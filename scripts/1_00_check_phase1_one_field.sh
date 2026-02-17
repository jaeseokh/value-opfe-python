#!/usr/bin/env bash
set -euo pipefail

trap 'echo "❌ Failed at line $LINENO: $BASH_COMMAND" >&2' ERR

ffy="${1:-10_1_2022}"
mode="${2:---overwrite}"   # default: overwrite; allow: --no-overwrite

echo "Using ffy_id=${ffy}"
echo "PWD=$(pwd)"

# --- Guard: ensure we are at project root ---
if [[ ! -d "src" || ! -d "scripts" || ! -d ".venv" ]]; then
  echo "❌ Run this from the project root (where src/, scripts/, .venv/ exist)." >&2
  exit 2
fi

# Always activate venv in SHELL (not R)
source .venv/bin/activate
echo "Python: $(which python)"
python --version

OW_FLAG="--overwrite"
if [[ "$mode" == "--no-overwrite" ]]; then
  OW_FLAG=""
fi

echo "== (1) Weather =="
python scripts/1_01_run_weather.py --ffy-id "$ffy" $OW_FLAG

echo "== (2) Topo rasters + topo manifest parquet =="
python scripts/1_02_run_topo.py --ffy-id "$ffy" $OW_FLAG

echo "== (3) SSURGO gpkg + props + manifest parquet =="
python scripts/1_03_run_ssurgo.py --ffy-id "$ffy" $OW_FLAG

echo "== (4) Build model_ready =="
python scripts/1_04_run_build_model_ready.py --ffy-id "$ffy" $OW_FLAG

echo "== (5) Quick inspect: model_ready columns/head =="
python scripts/utils/peak_parquet.py "data/export/parquet/model_ready/${ffy}_model_ready.parquet"

echo "== (6) Visual check maps (ssurgo + topo) =="
python scripts/utils/plot_phase1_maps.py "$ffy"

echo "DONE ✅"
