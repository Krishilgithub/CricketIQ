-- ======================================================================
-- models/gold/mart_team_form.sql
-- ----------------------------------------------------------------------
-- Calculates rolling team form (last 5 matches win rate). 
-- This uses DuckDB window functions over matches.
-- ======================================================================

{{ config(materialized='table') }}

WITH team_matches AS (
    -- Expand matches into two rows per match (one for each playing team)
    SELECT 
        m.match_id,
        m.match_date,
        t.team,
        CASE WHEN m.winner = t.team THEN 1 WHEN m.result_type IN ('tie', 'no result') THEN 0.5 ELSE 0 END AS is_win
    FROM {{ ref('fact_matches') }} m
    JOIN {{ ref('slv_match_teams') }} t ON m.match_id = t.match_id
),

rolling_stats AS (
    SELECT
        match_id,
        team,
        match_date,
        is_win,
        -- Calculate the win rate over the preceding 5 matches (excluding the current match)
        AVG(is_win) OVER (
            PARTITION BY team 
            ORDER BY match_date 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS form_last_5_win_rate,
        SUM(1) OVER (
            PARTITION BY team 
            ORDER BY match_date 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS matches_played_last_5
    FROM team_matches
)

SELECT * FROM rolling_stats
