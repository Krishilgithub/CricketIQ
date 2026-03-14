"""
Data Warehouse Schema Definitions — Star Schema on DuckDB.

Creates the Medallion Architecture layers:
  - Bronze: Raw ingested data (as-is from Cricsheet)
  - Silver: Cleaned, normalized, type-cast
  - Gold: Aggregated fact/dimension tables for analytics

Usage:
    python src/warehouse/schema.py  # Creates/resets the warehouse
"""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DUCKDB_PATH


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection to the warehouse."""
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DUCKDB_PATH))


def create_schemas(conn: duckdb.DuckDBPyConnection):
    """Create the three medallion architecture schemas."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS bronze;")
    conn.execute("CREATE SCHEMA IF NOT EXISTS silver;")
    conn.execute("CREATE SCHEMA IF NOT EXISTS gold;")
    print("✅ Created schemas: bronze, silver, gold")


def create_bronze_tables(conn: duckdb.DuckDBPyConnection):
    """Bronze layer: Raw data tables, minimal transformation."""

    conn.execute("DROP TABLE IF EXISTS bronze.raw_deliveries;")
    conn.execute("""
        CREATE TABLE bronze.raw_deliveries (
            match_id        VARCHAR,
            innings         INTEGER,
            batting_team    VARCHAR,
            bowling_team    VARCHAR,
            over            INTEGER,
            ball            INTEGER,
            ball_id         VARCHAR,
            phase           VARCHAR,
            batter          VARCHAR,
            bowler          VARCHAR,
            non_striker     VARCHAR,
            batter_runs     INTEGER,
            extras_runs     INTEGER,
            total_runs      INTEGER,
            is_wide         BOOLEAN,
            wide_runs       INTEGER,
            is_noball       BOOLEAN,
            noball_runs     INTEGER,
            is_bye          BOOLEAN,
            bye_runs        INTEGER,
            is_legbye       BOOLEAN,
            legbye_runs     INTEGER,
            is_penalty      BOOLEAN,
            penalty_runs    INTEGER,
            is_wicket       BOOLEAN,
            wicket_kind     VARCHAR,
            player_out      VARCHAR,
            fielders        VARCHAR,
            is_boundary_four BOOLEAN,
            is_boundary_six  BOOLEAN,
            is_dot_ball     BOOLEAN,
            match_date      VARCHAR,
            venue           VARCHAR,
            event_name      VARCHAR,
            source          VARCHAR,
            _loaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.execute("DROP TABLE IF EXISTS bronze.raw_matches;")
    conn.execute("""
        CREATE TABLE bronze.raw_matches (
            match_id            VARCHAR PRIMARY KEY,
            data_version        VARCHAR,
            created_date        VARCHAR,
            match_type          VARCHAR,
            match_type_number   INTEGER,
            gender              VARCHAR,
            season              VARCHAR,
            team1               VARCHAR,
            team2               VARCHAR,
            match_date          VARCHAR,
            venue               VARCHAR,
            city                VARCHAR,
            toss_winner         VARCHAR,
            toss_decision       VARCHAR,
            winner              VARCHAR,
            result_type         VARCHAR,
            result_margin       INTEGER,
            player_of_match     VARCHAR,
            event_name          VARCHAR,
            event_match_number  INTEGER,
            event_group         VARCHAR,
            event_stage         VARCHAR,
            overs_per_side      INTEGER,
            balls_per_over      INTEGER,
            team1_players       VARCHAR,
            team2_players       VARCHAR,
            source              VARCHAR,
            _loaded_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.execute("DROP TABLE IF EXISTS bronze.raw_player_registry;")
    conn.execute("""
        CREATE TABLE bronze.raw_player_registry (
            player_name     VARCHAR,
            cricsheet_id    VARCHAR,
            match_id        VARCHAR,
            source          VARCHAR,
            _loaded_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    print("✅ Created bronze layer tables")


def create_silver_tables(conn: duckdb.DuckDBPyConnection):
    """Silver layer: Cleaned, typed, and normalized tables."""

    conn.execute("DROP TABLE IF EXISTS silver.deliveries;")
    conn.execute("""
        CREATE TABLE silver.deliveries (
            match_id        VARCHAR NOT NULL,
            innings         INTEGER NOT NULL,
            batting_team    VARCHAR NOT NULL,
            bowling_team    VARCHAR NOT NULL,
            over            INTEGER NOT NULL,
            ball            INTEGER NOT NULL,
            phase           VARCHAR NOT NULL,
            batter          VARCHAR NOT NULL,
            bowler          VARCHAR NOT NULL,
            non_striker     VARCHAR,
            batter_runs     INTEGER NOT NULL DEFAULT 0,
            extras_runs     INTEGER NOT NULL DEFAULT 0,
            total_runs      INTEGER NOT NULL DEFAULT 0,
            is_wide         BOOLEAN DEFAULT FALSE,
            is_noball       BOOLEAN DEFAULT FALSE,
            is_bye          BOOLEAN DEFAULT FALSE,
            is_legbye       BOOLEAN DEFAULT FALSE,
            is_wicket       BOOLEAN DEFAULT FALSE,
            wicket_kind     VARCHAR,
            player_out      VARCHAR,
            is_boundary_four BOOLEAN DEFAULT FALSE,
            is_boundary_six  BOOLEAN DEFAULT FALSE,
            is_dot_ball     BOOLEAN DEFAULT FALSE,
            is_legal_ball   BOOLEAN,        -- computed: NOT wide AND NOT noball
            cumulative_runs INTEGER,        -- running total in innings
            cumulative_wickets INTEGER,     -- running wickets in innings
            run_rate        DOUBLE,         -- current run rate
            source          VARCHAR NOT NULL,
            match_date      DATE
        );
    """)

    conn.execute("DROP TABLE IF EXISTS silver.matches;")
    conn.execute("""
        CREATE TABLE silver.matches (
            match_id        VARCHAR PRIMARY KEY,
            match_type      VARCHAR,
            season          VARCHAR,
            team1           VARCHAR NOT NULL,
            team2           VARCHAR NOT NULL,
            match_date      DATE,
            venue           VARCHAR,
            city            VARCHAR,
            toss_winner     VARCHAR,
            toss_decision   VARCHAR,
            winner          VARCHAR,
            result_type     VARCHAR,
            result_margin   INTEGER,
            player_of_match VARCHAR,
            event_name      VARCHAR,
            event_stage     VARCHAR,
            overs_per_side  INTEGER DEFAULT 20,
            source          VARCHAR NOT NULL,
            -- Derived fields
            toss_win_match_win BOOLEAN,     -- Did toss winner also win match?
            bat_first_team  VARCHAR,        -- Team that batted first
            chase_team      VARCHAR,        -- Team that chased
            is_world_cup    BOOLEAN         -- Is this a T20 World Cup match?
        );
    """)

    conn.execute("DROP TABLE IF EXISTS silver.players;")
    conn.execute("""
        CREATE TABLE silver.players (
            cricsheet_id    VARCHAR,
            player_name     VARCHAR NOT NULL,
            -- Aggregated unique identifier
            UNIQUE(cricsheet_id, player_name)
        );
    """)

    print("✅ Created silver layer tables")


def create_gold_tables(conn: duckdb.DuckDBPyConnection):
    """Gold layer: Aggregated fact and dimension tables (Star Schema)."""

    # ── DIMENSION TABLES ──

    conn.execute("DROP TABLE IF EXISTS gold.dim_player;")
    conn.execute("""
        CREATE TABLE gold.dim_player (
            player_key      INTEGER PRIMARY KEY,
            cricsheet_id    VARCHAR,
            player_name     VARCHAR NOT NULL,
            teams_played    VARCHAR,        -- Comma-separated teams
            matches_played  INTEGER,
            batting_style   VARCHAR,        -- (to be enriched)
            bowling_style   VARCHAR,        -- (to be enriched)
            role            VARCHAR         -- (to be enriched: batter, bowler, allrounder)
        );
    """)

    conn.execute("DROP TABLE IF EXISTS gold.dim_team;")
    conn.execute("""
        CREATE TABLE gold.dim_team (
            team_key    INTEGER PRIMARY KEY,
            team_name   VARCHAR NOT NULL UNIQUE,
            is_icc_full_member BOOLEAN DEFAULT FALSE
        );
    """)

    conn.execute("DROP TABLE IF EXISTS gold.dim_venue;")
    conn.execute("""
        CREATE TABLE gold.dim_venue (
            venue_key   INTEGER PRIMARY KEY,
            venue_name  VARCHAR NOT NULL,
            city        VARCHAR,
            country     VARCHAR,    -- (to be enriched)
            avg_first_innings_score DOUBLE,
            avg_second_innings_score DOUBLE,
            matches_hosted INTEGER
        );
    """)

    conn.execute("DROP TABLE IF EXISTS gold.dim_date;")
    conn.execute("""
        CREATE TABLE gold.dim_date (
            date_key    INTEGER PRIMARY KEY,
            full_date   DATE NOT NULL UNIQUE,
            year        INTEGER,
            month       INTEGER,
            day         INTEGER,
            day_of_week VARCHAR,
            quarter     INTEGER,
            is_weekend  BOOLEAN
        );
    """)

    conn.execute("DROP TABLE IF EXISTS gold.dim_tournament;")
    conn.execute("""
        CREATE TABLE gold.dim_tournament (
            tournament_key  INTEGER PRIMARY KEY,
            event_name      VARCHAR NOT NULL,
            season          VARCHAR,
            is_world_cup    BOOLEAN DEFAULT FALSE,
            is_league       BOOLEAN DEFAULT FALSE
        );
    """)

    # ── FACT TABLES ──

    conn.execute("DROP TABLE IF EXISTS gold.fact_match_results;")
    conn.execute("""
        CREATE TABLE gold.fact_match_results (
            match_id            VARCHAR PRIMARY KEY,
            date_key            INTEGER REFERENCES gold.dim_date(date_key),
            venue_key           INTEGER REFERENCES gold.dim_venue(venue_key),
            tournament_key      INTEGER REFERENCES gold.dim_tournament(tournament_key),
            team1_key           INTEGER REFERENCES gold.dim_team(team_key),
            team2_key           INTEGER REFERENCES gold.dim_team(team_key),
            winner_key          INTEGER REFERENCES gold.dim_team(team_key),
            toss_winner_key     INTEGER REFERENCES gold.dim_team(team_key),
            toss_decision       VARCHAR,
            result_type         VARCHAR,
            result_margin       INTEGER,
            player_of_match_key INTEGER REFERENCES gold.dim_player(player_key),
            team1_score         INTEGER,
            team1_wickets       INTEGER,
            team1_overs         DOUBLE,
            team2_score         INTEGER,
            team2_wickets       INTEGER,
            team2_overs         DOUBLE,
            source              VARCHAR
        );
    """)

    conn.execute("DROP TABLE IF EXISTS gold.fact_batting_innings;")
    conn.execute("""
        CREATE TABLE gold.fact_batting_innings (
            match_id        VARCHAR,
            innings         INTEGER,
            player_key      INTEGER REFERENCES gold.dim_player(player_key),
            team_key        INTEGER REFERENCES gold.dim_team(team_key),
            venue_key       INTEGER REFERENCES gold.dim_venue(venue_key),
            date_key        INTEGER REFERENCES gold.dim_date(date_key),
            runs_scored     INTEGER,
            balls_faced     INTEGER,
            fours           INTEGER,
            sixes           INTEGER,
            dot_balls_faced INTEGER,
            strike_rate     DOUBLE,
            is_out          BOOLEAN,
            dismissal_type  VARCHAR,
            batting_position INTEGER,
            pp_runs         INTEGER,    -- Powerplay runs
            middle_runs     INTEGER,    -- Middle overs runs
            death_runs      INTEGER,    -- Death overs runs
            source          VARCHAR
        );
    """)

    conn.execute("DROP TABLE IF EXISTS gold.fact_bowling_innings;")
    conn.execute("""
        CREATE TABLE gold.fact_bowling_innings (
            match_id        VARCHAR,
            innings         INTEGER,
            player_key      INTEGER REFERENCES gold.dim_player(player_key),
            team_key        INTEGER REFERENCES gold.dim_team(team_key),
            venue_key       INTEGER REFERENCES gold.dim_venue(venue_key),
            date_key        INTEGER REFERENCES gold.dim_date(date_key),
            overs_bowled    DOUBLE,
            balls_bowled    INTEGER,
            runs_conceded   INTEGER,
            wickets_taken   INTEGER,
            economy_rate    DOUBLE,
            dot_balls       INTEGER,
            wides           INTEGER,
            noballs         INTEGER,
            fours_conceded  INTEGER,
            sixes_conceded  INTEGER,
            pp_wickets      INTEGER,
            middle_wickets  INTEGER,
            death_wickets   INTEGER,
            source          VARCHAR
        );
    """)

    conn.execute("DROP TABLE IF EXISTS gold.fact_innings_summary;")
    conn.execute("""
        CREATE TABLE gold.fact_innings_summary (
            match_id        VARCHAR,
            innings         INTEGER,
            team_key        INTEGER REFERENCES gold.dim_team(team_key),
            venue_key       INTEGER REFERENCES gold.dim_venue(venue_key),
            total_runs      INTEGER,
            total_wickets   INTEGER,
            total_overs     DOUBLE,
            run_rate        DOUBLE,
            pp_runs         INTEGER,
            pp_wickets      INTEGER,
            pp_run_rate     DOUBLE,
            middle_runs     INTEGER,
            middle_wickets  INTEGER,
            middle_run_rate DOUBLE,
            death_runs      INTEGER,
            death_wickets   INTEGER,
            death_run_rate  DOUBLE,
            boundary_runs   INTEGER,
            fours           INTEGER,
            sixes           INTEGER,
            dot_balls       INTEGER,
            extras          INTEGER,
            source          VARCHAR,
            PRIMARY KEY(match_id, innings)
        );
    """)

    print("✅ Created gold layer tables (Star Schema)")


def initialize_warehouse():
    """Create the full warehouse with all layers."""
    print("=" * 60)
    print("🏗️  INITIALIZING DATA WAREHOUSE")
    print("=" * 60)
    print(f"   Database: {DUCKDB_PATH}")

    conn = get_connection()
    create_schemas(conn)
    create_bronze_tables(conn)
    create_silver_tables(conn)
    create_gold_tables(conn)
    conn.close()

    print("\n✅ Warehouse initialization complete!")
    print(f"   Database file: {DUCKDB_PATH}")
    print(f"   Database size: {DUCKDB_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    initialize_warehouse()
