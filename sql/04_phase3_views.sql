-- =====================================================
-- 04_phase3_views.sql
-- Optional views for Phase3 diagnostics
-- Supports either:
--   A) legacy phase3_diagnostics parquet tables (phase3_*)
--   B) scripts/3_00..3_04 outputs (phase3_step*)
-- =====================================================

DROP VIEW IF EXISTS vw_phase3_model_scorecard CASCADE;
DROP VIEW IF EXISTS vw_phase3_eonr_risk CASCADE;
DROP VIEW IF EXISTS vw_phase3_assumptions CASCADE;

DO $$
BEGIN
  IF to_regclass('public.phase3_step1_predictions_obs') IS NOT NULL
     AND to_regclass('public.phase3_step2_eonr_gap') IS NOT NULL THEN
    EXECUTE $v$
      CREATE VIEW vw_phase3_model_scorecard AS
      WITH step1 AS (
          SELECT 'rf'::text AS model,
                 sqrt(avg(power(y_true - yhat_rf, 2)))::double precision AS rmse_test
          FROM phase3_step1_predictions_obs
          UNION ALL
          SELECT 'xgb_uncon'::text AS model,
                 sqrt(avg(power(y_true - yhat_xgb_uncon, 2)))::double precision AS rmse_test
          FROM phase3_step1_predictions_obs
          UNION ALL
          SELECT 'xgb_monotone'::text AS model,
                 sqrt(avg(power(y_true - yhat_xgb_monotone, 2)))::double precision AS rmse_test
          FROM phase3_step1_predictions_obs
      ),
      step2 AS (
          SELECT
              'xgb_monotone'::text AS model,
              avg(abs_delta_eonr)::double precision AS mean_abs_delta_eonr,
              percentile_cont(0.5) WITHIN GROUP (ORDER BY abs_delta_eonr)::double precision AS median_abs_delta_eonr,
              avg("loss_$per_acre_proxy")::double precision AS mean_profit_loss
          FROM phase3_step2_eonr_gap
      )
      SELECT
          s1.model,
          s1.rmse_test,
          s2.mean_abs_delta_eonr,
          s2.median_abs_delta_eonr,
          s2.mean_profit_loss
      FROM step1 s1
      LEFT JOIN step2 s2
        ON s1.model = s2.model
      ORDER BY s1.model
    $v$;

  ELSIF to_regclass('public.phase3_lolo_rmse_summary') IS NOT NULL
        AND to_regclass('public.phase3_eonr_summary') IS NOT NULL THEN
    EXECUTE $v$
      CREATE VIEW vw_phase3_model_scorecard AS
      SELECT
          COALESCE(s1.model, s2.model) AS model,
          s1.rmse_test_mean::double precision AS rmse_test,
          s2.delta_n_mean::double precision AS mean_abs_delta_eonr,
          s2.delta_n_median::double precision AS median_abs_delta_eonr,
          s2.profit_loss_mean::double precision AS mean_profit_loss
      FROM phase3_lolo_rmse_summary s1
      FULL OUTER JOIN phase3_eonr_summary s2
          ON s1.model = s2.model
    $v$;

  ELSE
    EXECUTE $v$
      CREATE VIEW vw_phase3_model_scorecard AS
      SELECT
          NULL::text AS model,
          NULL::double precision AS rmse_test,
          NULL::double precision AS mean_abs_delta_eonr,
          NULL::double precision AS median_abs_delta_eonr,
          NULL::double precision AS mean_profit_loss
      WHERE FALSE
    $v$;
  END IF;
END
$$;

DO $$
BEGIN
  IF to_regclass('public.phase3_step2_eonr_gap') IS NOT NULL THEN
    EXECUTE $v$
      CREATE VIEW vw_phase3_eonr_risk AS
      SELECT
          lolo_group AS location_id,
          'xgb_monotone'::text AS model,
          true_eonr,
          pred_eonr_xgb_monotone AS pred_eonr,
          delta_eonr,
          abs_delta_eonr,
          "loss_$per_acre_proxy" AS profit_loss,
          (abs_delta_eonr >= 35)::INT AS flag_delta_n_ge35,
          (abs_delta_eonr >= 40)::INT AS flag_delta_n_ge40,
          ("loss_$per_acre_proxy" > 0)::INT AS flag_profit_loss_positive
      FROM phase3_step2_eonr_gap
    $v$;

  ELSIF to_regclass('public.phase3_eonr_by_field') IS NOT NULL THEN
    EXECUTE $v$
      CREATE VIEW vw_phase3_eonr_risk AS
      SELECT
          location_id,
          model,
          eonr_true AS true_eonr,
          eonr_pred AS pred_eonr,
          delta_n AS delta_eonr,
          delta_n AS abs_delta_eonr,
          profit_loss,
          (delta_n >= 35)::INT AS flag_delta_n_ge35,
          (delta_n >= 40)::INT AS flag_delta_n_ge40,
          (profit_loss > 0)::INT AS flag_profit_loss_positive
      FROM phase3_eonr_by_field
    $v$;

  ELSE
    EXECUTE $v$
      CREATE VIEW vw_phase3_eonr_risk AS
      SELECT
          NULL::text AS location_id,
          NULL::text AS model,
          NULL::double precision AS true_eonr,
          NULL::double precision AS pred_eonr,
          NULL::double precision AS delta_eonr,
          NULL::double precision AS abs_delta_eonr,
          NULL::double precision AS profit_loss,
          NULL::int AS flag_delta_n_ge35,
          NULL::int AS flag_delta_n_ge40,
          NULL::int AS flag_profit_loss_positive
      WHERE FALSE
    $v$;
  END IF;
END
$$;

DO $$
BEGIN
  IF to_regclass('public.phase3_assumptions_table') IS NOT NULL THEN
    EXECUTE $v$
      CREATE VIEW vw_phase3_assumptions AS
      SELECT
          assumption,
          required_condition,
          empirical_evidence,
          verdict
      FROM phase3_assumptions_table
    $v$;

  ELSIF to_regclass('public.phase3_step4_summary_step2') IS NOT NULL THEN
    EXECUTE $v$
      CREATE VIEW vw_phase3_assumptions AS
      SELECT
          'Transferability'::text AS assumption,
          'Mean abs_delta_eonr <= 35 lbs/acre'::text AS required_condition,
          ('mean_abs_delta_eonr=' || round(mean_abs_delta_eonr::numeric, 2)::text)::text AS empirical_evidence,
          CASE WHEN mean_abs_delta_eonr <= 35 THEN 'OK' ELSE 'Violated' END::text AS verdict
      FROM phase3_step4_summary_step2
    $v$;

  ELSE
    EXECUTE $v$
      CREATE VIEW vw_phase3_assumptions AS
      SELECT
          NULL::text AS assumption,
          NULL::text AS required_condition,
          NULL::text AS empirical_evidence,
          NULL::text AS verdict
      WHERE FALSE
    $v$;
  END IF;
END
$$;
