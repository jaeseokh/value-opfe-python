-- =====================================================
-- 03_audit_views.sql
-- Audits: coverage + missing joins + regime support
-- =====================================================

DROP VIEW IF EXISTS vw_feature_coverage CASCADE;
DROP VIEW IF EXISTS vw_missing_weather  CASCADE;
DROP VIEW IF EXISTS vw_missing_soil     CASCADE;
DROP VIEW IF EXISTS vw_missing_topo     CASCADE;
DROP VIEW IF EXISTS vw_regime_support   CASCADE;

CREATE VIEW vw_feature_coverage AS
SELECT
    year,
    COUNT(*) AS n_rows,
    SUM((prcp_gs IS NOT NULL)::INT) AS n_has_weather,
    SUM((dominant_mukey IS NOT NULL)::INT) AS n_has_soil,
    SUM((elev_mean IS NOT NULL)::INT) AS n_has_topo,
    SUM((pass_qc IS TRUE)::INT) AS n_pass_qc
FROM vw_model_features
GROUP BY year
ORDER BY year;

CREATE VIEW vw_missing_weather AS
SELECT DISTINCT
    o.field_id, o.year
FROM stg_ofpe_plot_obs o
LEFT JOIN stg_weather_gs w
    ON o.field_id = w.field_id AND o.year = w.year
WHERE w.field_id IS NULL;

CREATE VIEW vw_missing_soil AS
SELECT DISTINCT
    o.field_id
FROM stg_ofpe_plot_obs o
LEFT JOIN stg_soil s
    ON o.field_id = s.field_id
WHERE s.field_id IS NULL;

CREATE VIEW vw_missing_topo AS
SELECT DISTINCT
    o.field_id
FROM stg_ofpe_plot_obs o
LEFT JOIN stg_topo t
    ON o.field_id = t.field_id
WHERE t.field_id IS NULL;

CREATE VIEW vw_regime_support AS
SELECT
    regime_prcp,
    COUNT(DISTINCT (field_id || '-' || year::TEXT)) AS n_field_years
FROM vw_weather_regime
GROUP BY regime_prcp
ORDER BY regime_prcp;
