-- ======================================================================
-- models/silver/slv_deliveries.sql
-- ----------------------------------------------------------------------
-- Cleans the deliveries table.
-- Filters for valid matches only. Applies canonical mapping to batting_team.
-- ======================================================================

WITH raw_deliveries AS (
    SELECT *
    FROM {{ source('bronze', 'deliveries') }}
),

valid_matches AS (
    SELECT match_id
    FROM {{ ref('slv_matches') }}
),

teams_map AS (
    SELECT * FROM {{ ref('team_aliases') }}
)

SELECT
    d.match_id,
    d.innings_number,
    CAST(d.over AS INTEGER) AS over_number,
    CAST(d.ball_in_over AS INTEGER) AS ball_number,
    -- Add a sequential ball_id for rolling window logic
    ROW_NUMBER() OVER (
        PARTITION BY d.match_id, d.innings_number 
        ORDER BY d.over, d.ball_in_over
    ) AS innings_ball_sequence,
    COALESCE(t.canonical, d.batting_team) AS batting_team,
    d.batter,
    d.bowler,
    d.non_striker,
    CAST(d.runs_batter AS INTEGER) AS runs_batter,
    CAST(d.runs_extras AS INTEGER) AS runs_extras,
    CAST(d.runs_total  AS INTEGER) AS runs_total,
    CAST(d.extras_wides   AS INTEGER) AS extras_wides,
    CAST(d.extras_noballs AS INTEGER) AS extras_noballs,
    CAST(d.extras_byes    AS INTEGER) AS extras_byes,
    CAST(d.extras_legbyes AS INTEGER) AS extras_legbyes,
    CAST(d.extras_penalty AS INTEGER) AS extras_penalty,
    CASE WHEN d.extras_wides > 0 OR d.extras_noballs > 0 THEN 0 ELSE 1 END AS is_legal_ball
FROM raw_deliveries d
INNER JOIN valid_matches v ON d.match_id = v.match_id
LEFT JOIN teams_map t ON d.batting_team = t.alias
