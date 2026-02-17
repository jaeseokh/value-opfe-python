-- =====================================================
-- 01_staging_views.sql (Phase1 aligned: field-year tables)
-- =====================================================

-- Drop staging views first (safe reruns)
DROP VIEW IF EXISTS stg_ofpe_plot_obs CASCADE;
DROP VIEW IF EXISTS stg_weather_gs   CASCADE;
DROP VIEW IF EXISTS stg_soil         CASCADE;
DROP VIEW IF EXISTS stg_topo         CASCADE;

-- 1) OFPE plot observations (standardized pass-through)
CREATE VIEW stg_ofpe_plot_obs AS
SELECT
    field_id,
    year,
    plot_id,
    n_rate,
    yield,
    (yield IS NOT NULL AND n_rate IS NOT NULL AND n_rate >= 0) AS pass_qc_basic
FROM ofpe_plot_obs;

-- 2) Weather (Phase1 already field-year aggregated)
-- Expect table: weather_season(field_id, year, prcp_gs, gdd_gs, edd_gs, tmax_gs_mean, tmin_gs_mean, heat_days_ge30c_gs)
CREATE VIEW stg_weather_gs AS
SELECT
    field_id,
    year,
    prcp_gs,
    gdd_gs,
    edd_gs,
    tmax_gs_mean,
    tmin_gs_mean,
    heat_days_ge30c_gs
FROM weather_season;

-- 3) SSURGO (Phase1 metrics: mukey dominance + diversity)
-- Expect columns like: dominant_mukey, dom_share, n_mukey, hhi, entropy, ssurgo_status
CREATE VIEW stg_soil AS
SELECT
    field_id,
    dominant_mukey,
    dom_share,
    n_mukey,
    hhi,
    entropy,
    ssurgo_status
FROM soil_ssurgo_field;

-- 4) Topography (Phase1 metrics: *_mean)
CREATE VIEW stg_topo AS
SELECT
    field_id,
    elev_mean,
    slope_mean,
    aspect_mean,
    tpi_mean
FROM topo_field;
