#!/usr/bin/env bash
set -euo pipefail

DB_NAME="${1:-ofpe_share}"

echo "[INFO] Building Phase3 SQL views in database: ${DB_NAME}"
psql -d "${DB_NAME}" -f sql/04_phase3_views.sql
echo "[OK] Phase3 SQL views build complete."

