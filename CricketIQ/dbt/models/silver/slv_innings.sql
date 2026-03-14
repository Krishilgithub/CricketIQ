-- ======================================================================
-- models/silver/slv_innings.sql
-- ----------------------------------------------------------------------
-- Cleans the innings aggregates.
-- ======================================================================

WITH raw_innings AS (
    SELECT *
    FROM {{ source('bronze', 'innings') }}
),

valid_matches AS (
    SELECT match_id
    FROM {{ ref('slv_matches') }}
),

teams_map AS (
    SELECT * FROM {{ ref('team_aliases') }}
)

SELECT
    i.match_id,
    i.innings_number,
    COALESCE(t.canonical, i.team) AS batting_team,
    CAST(i.total_runs AS INTEGER) AS total_runs,
    CAST(i.total_wickets AS INTEGER) AS total_wickets,
    CAST(i.total_balls AS INTEGER) AS total_balls,
    CAST(i.extras_byes AS INTEGER) AS extras_byes,
    CAST(i.extras_legbyes AS INTEGER) AS extras_legbyes,
    CAST(i.extras_noballs AS INTEGER) AS extras_noballs,
    CAST(i.extras_wides AS INTEGER) AS extras_wides,
    CAST(i.extras_penalty AS INTEGER) AS extras_penalty
FROM raw_innings i
INNER JOIN valid_matches v ON i.match_id = v.match_id
LEFT JOIN teams_map t ON i.team = t.alias
