set -euo pipefail
DB_NAME="${1:-ofpe_share}"

echo "[INFO] Building SQL objects in database: ${DB_NAME}"

psql -d "${DB_NAME}" -f sql/00_schema.sql
psql -d "${DB_NAME}" -f sql/01_staging_views.sql
psql -d "${DB_NAME}" -f sql/02_feature_views.sql
psql -d "${DB_NAME}" -f sql/03_audit_views.sql

echo "[OK] SQL build complete."
