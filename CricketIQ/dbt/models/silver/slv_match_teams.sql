-- ======================================================================
-- models/silver/slv_match_teams.sql
-- ----------------------------------------------------------------------
-- Applies canonical mappings to the team names playing in a match.
-- Only includes matches that exist in slv_matches.
-- ======================================================================

WITH raw_teams AS (
    SELECT *
    FROM {{ source('bronze', 'match_teams') }}
),

valid_matches AS (
    SELECT match_id
    FROM {{ ref('slv_matches') }}
),

teams_map AS (
    SELECT * FROM {{ ref('team_aliases') }}
)

SELECT
    r.match_id,
    COALESCE(t.canonical, r.team) AS team
FROM raw_teams r
INNER JOIN valid_matches v ON r.match_id = v.match_id
LEFT JOIN teams_map t ON r.team = t.alias
