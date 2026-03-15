-- ======================================================================
-- models/silver/slv_matches.sql
-- ----------------------------------------------------------------------
-- Cleans the matches table.
-- Filters for:
--   - gender = 'male'
--   - match_type = 'T20'
--   - team_type = 'international'
-- Applies canonical mappings for venue and team names.
-- ======================================================================

WITH raw_matches AS (
    SELECT *
    FROM {{ source('bronze', 'matches') }}
    WHERE gender = 'male'
      AND match_type = 'T20'
      AND team_type = 'international'
),

teams_map AS (
    SELECT * FROM {{ ref('team_aliases') }}
),

venue_map AS (
    SELECT * FROM {{ ref('venue_aliases') }}
)

SELECT
    m.match_id,
    CAST(m.match_date AS DATE) AS match_date,
    m.season,
    m.event_name,
    m.event_match_number,
    COALESCE(v.canonical, m.venue) AS venue,
    m.city,
    COALESCE(t1.canonical, m.toss_winner) AS toss_winner,
    m.toss_decision,
    COALESCE(t2.canonical, m.winner) AS winner,
    m.result_type,
    m.result_margin,
    m.method
FROM raw_matches m
LEFT JOIN venue_map v  ON m.venue = v.alias
LEFT JOIN teams_map t1 ON m.toss_winner = t1.alias
LEFT JOIN teams_map t2 ON m.winner = t2.alias
