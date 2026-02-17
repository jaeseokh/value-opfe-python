-- =====================================================
-- 00_schema.sql
-- Minimal Phase1 schema
-- =====================================================

CREATE TABLE IF NOT EXISTS ofpe_plot_obs (
  field_id TEXT,
  year INT,
  plot_id TEXT,
  n_rate DOUBLE PRECISION,
  yield DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS weather_fieldyear (
  field_id TEXT,
  year INT,
  prcp_gs DOUBLE PRECISION,
  gdd_gs DOUBLE PRECISION,
  edd_gs DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS soil_ssurgo_plot (
  field_id        TEXT,
  year            INT,
  plot_id         TEXT,
  dominant_mukey  TEXT,
  dom_share       DOUBLE PRECISION,
  n_mukey         INT,
  om              DOUBLE PRECISION,
  clay            DOUBLE PRECISION,
  sand            DOUBLE PRECISION,
  awc             DOUBLE PRECISION,
  drainage_class  DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS soil_ssurgo_field (
  field_id        TEXT,
  dominant_mukey  TEXT,
  dom_share       DOUBLE PRECISION,
  n_mukey         INT,
  hhi             DOUBLE PRECISION,
  entropy         DOUBLE PRECISION,
  ssurgo_status   TEXT
);

CREATE TABLE IF NOT EXISTS topo_field (
  field_id TEXT,
  elev DOUBLE PRECISION,
  slope DOUBLE PRECISION,
  aspect DOUBLE PRECISION,
  tpi DOUBLE PRECISION,
  elev_mean DOUBLE PRECISION,
  slope_mean DOUBLE PRECISION,
  aspect_mean DOUBLE PRECISION,
  tpi_mean DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS weather_season (
  field_id TEXT,
  year INT,
  prcp_gs NUMERIC,
  gdd_gs NUMERIC,
  edd_gs NUMERIC,
  tmax_gs_mean NUMERIC,
  tmin_gs_mean NUMERIC,
  heat_days_ge30c_gs INT
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS idx_ofpe_field_year ON ofpe_plot_obs(field_id, year);
CREATE INDEX IF NOT EXISTS idx_weather_season_field_year ON weather_season(field_id, year);
CREATE INDEX IF NOT EXISTS idx_weather_field_year ON weather_fieldyear(field_id, year);
