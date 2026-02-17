-- =====================================================
-- 02_feature_views.sql
-- Feature mart + regime labels (Phase1 aligned)
-- =====================================================

DROP VIEW IF EXISTS vw_prcp_gs_cutoffs CASCADE;
DROP VIEW IF EXISTS vw_weather_regime  CASCADE;
DROP VIEW IF EXISTS vw_model_features  CASCADE;

-- A) precip cutoffs across field-years
CREATE VIEW vw_prcp_gs_cutoffs AS
SELECT
    percentile_cont(0.333) WITHIN GROUP (ORDER BY prcp_gs) AS q33_prcp_gs,
    percentile_cont(0.666) WITHIN GROUP (ORDER BY prcp_gs) AS q66_prcp_gs
FROM stg_weather_gs
WHERE prcp_gs IS NOT NULL;

-- B) add regime label
CREATE VIEW vw_weather_regime AS
SELECT
    w.*,
    CASE
        WHEN w.prcp_gs <  c.q33_prcp_gs THEN 'dry'
        WHEN w.prcp_gs >= c.q66_prcp_gs THEN 'wet'
        ELSE 'normal'
    END AS regime_prcp
FROM stg_weather_gs w
CROSS JOIN vw_prcp_gs_cutoffs c;

-- C) final modeling dataset
CREATE VIEW vw_model_features AS
SELECT
    o.field_id,
    o.year,
    o.plot_id,
    o.n_rate,
    o.yield,

    wr.prcp_gs,
    wr.gdd_gs,
    wr.edd_gs,
    wr.tmax_gs_mean,
    wr.tmin_gs_mean,
    wr.heat_days_ge30c_gs,
    wr.regime_prcp,

    s.dominant_mukey,
    s.dom_share,
    s.n_mukey,
    s.hhi,
    s.entropy,
    s.ssurgo_status,

    t.elev_mean,
    t.slope_mean,
    t.aspect_mean,
    t.tpi_mean,

    -- QC: require core outcome/treatment + weather
    (o.pass_qc_basic AND wr.prcp_gs IS NOT NULL) AS pass_qc
FROM stg_ofpe_plot_obs o
LEFT JOIN vw_weather_regime wr
    ON o.field_id = wr.field_id
   AND o.year     = wr.year
LEFT JOIN stg_soil s
    ON o.field_id = s.field_id
LEFT JOIN stg_topo t
    ON o.field_id = t.field_id;
