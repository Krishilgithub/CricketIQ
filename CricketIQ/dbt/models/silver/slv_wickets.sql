-- ======================================================================
-- models/silver/slv_wickets.sql
-- ----------------------------------------------------------------------
-- Cleans the wickets table. Maps batting_team to canonical names.
-- Only includes wickets for valid matches.
-- ======================================================================

WITH raw_wickets AS (
    SELECT *
    FROM {{ source('bronze', 'wickets') }}
),

valid_matches AS (
    SELECT match_id
    FROM {{ ref('slv_matches') }}
),

teams_map AS (
    SELECT * FROM {{ ref('team_aliases') }}
)

SELECT
    w.match_id,
    w.innings_number,
    CAST(w.over AS INTEGER) AS over_number,
    CAST(w.ball_in_over AS INTEGER) AS ball_number,
    COALESCE(t.canonical, w.batting_team) AS batting_team,
    w.player_out,
    w.kind AS dismissal_kind,
    w.fielders
FROM raw_wickets w
INNER JOIN valid_matches v ON w.match_id = v.match_id
LEFT JOIN teams_map t ON w.batting_team = t.alias
