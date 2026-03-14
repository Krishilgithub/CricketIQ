-- ======================================================================
-- models/gold/fact_matches.sql
-- ----------------------------------------------------------------------
-- Fact table for matches. Downstream models should read from gold.
-- ======================================================================

{{ config(materialized='table') }}

SELECT
    match_id,
    match_date,
    season,
    event_name,
    venue,
    city,
    toss_winner,
    toss_decision,
    winner,
    result_type,
    result_margin,
    method,
    -- Target label for pre-match predictions (did team 1 win?)
    -- We need a canonical "team_1" and "team_2". We define team_1 as toss_winner for consistency.
    toss_winner AS team_1,
    CASE WHEN toss_winner = winner THEN 1 ELSE 0 END AS team_1_win
FROM {{ ref('slv_matches') }}
