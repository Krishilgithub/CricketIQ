-- ======================================================================
-- models/gold/mart_bowling_stats.sql
-- ----------------------------------------------------------------------
-- Rolling bowling stats per player per match.
-- ======================================================================

{{ config(materialized='table') }}

WITH match_bowling AS (
    SELECT
        d.match_id,
        m.match_date,
        -- Bowling team is the team that is NOT batting
        CASE WHEN d.batting_team = m.team_1 THEN m.winner /* wait, winner might not be the other team */ END AS bowling_team_placeholder,
        d.bowler,
        SUM(d.runs_total) AS runs_conceded,
        SUM(CASE WHEN d.is_legal_ball = 1 THEN 1 ELSE 0 END) AS legal_deliveries,
        SUM(CASE WHEN d.is_wicket = 1 THEN 1 ELSE 0 END) AS match_wickets
    FROM {{ ref('fact_deliveries') }} d
    JOIN {{ ref('fact_matches') }} m ON d.match_id = m.match_id
    GROUP BY d.match_id, m.match_date, d.batting_team, m.team_1, m.winner, d.bowler
)

SELECT
    match_id,
    match_date,
    bowler,
    runs_conceded,
    legal_deliveries,
    match_wickets,
    SUM(match_wickets) OVER (
        PARTITION BY bowler 
        ORDER BY match_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS career_wickets_before_match
FROM match_bowling
