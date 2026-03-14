-- ======================================================================
-- models/gold/mart_batting_stats.sql
-- ----------------------------------------------------------------------
-- Rolling batting stats per player per match.
-- ======================================================================

{{ config(materialized='table') }}

WITH match_batting AS (
    SELECT
        d.match_id,
        m.match_date,
        d.batting_team AS team,
        d.batter,
        SUM(d.runs_batter) AS match_runs,
        SUM(CASE WHEN d.is_legal_ball = 1 THEN 1 ELSE 0 END) AS balls_faced,
        MAX(CASE WHEN w.player_out IS NOT NULL THEN 1 ELSE 0 END) AS was_dismissed
    FROM {{ ref('fact_deliveries') }} d
    JOIN {{ ref('fact_matches') }} m ON d.match_id = m.match_id
    LEFT JOIN {{ ref('fact_wickets') }} w ON d.match_id = w.match_id 
       AND d.innings_number = w.innings_number 
       AND d.over_number = w.over_number 
       AND d.ball_number = w.ball_number 
       AND d.batter = w.player_out
    GROUP BY d.match_id, m.match_date, d.batting_team, d.batter
)

SELECT
    match_id,
    match_date,
    team,
    batter,
    match_runs,
    balls_faced,
    was_dismissed,
    SUM(match_runs) OVER (
        PARTITION BY batter 
        ORDER BY match_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS career_runs_before_match,
    SUM(was_dismissed) OVER (
        PARTITION BY batter 
        ORDER BY match_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS career_dismissals_before_match
FROM match_batting
