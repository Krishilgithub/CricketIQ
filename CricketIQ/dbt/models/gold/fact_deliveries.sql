-- ======================================================================
-- models/gold/fact_deliveries.sql
-- ======================================================================

{{ config(materialized='table') }}

SELECT
    d.match_id,
    d.innings_number,
    d.over_number,
    d.ball_number,
    d.innings_ball_sequence,
    d.batting_team,
    d.batter,
    d.bowler,
    d.non_striker,
    d.runs_batter,
    d.runs_extras,
    d.runs_total,
    d.extras_wides,
    d.extras_noballs,
    d.is_legal_ball,
    -- Join to wickets to flag if this ball resulted in a wicket
    CASE WHEN w.player_out IS NOT NULL THEN 1 ELSE 0 END AS is_wicket,
    w.player_out AS wicket_player_out,
    w.dismissal_kind
FROM {{ ref('slv_deliveries') }} d
LEFT JOIN {{ ref('slv_wickets') }} w
  ON  d.match_id = w.match_id
  AND d.innings_number = w.innings_number
  AND d.over_number = w.over_number
  AND d.ball_number = w.ball_number
