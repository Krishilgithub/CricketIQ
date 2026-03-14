"""
ETL Pipeline: Bronze → Silver → Gold transformation.

Orchestrates the full data flow from raw parsed Cricsheet data
through the medallion architecture layers.

Usage:
    python src/etl/run_pipeline.py                        # Full pipeline
    python src/etl/run_pipeline.py --stage bronze         # Only bronze
    python src/etl/run_pipeline.py --stage silver         # Only silver
    python src/etl/run_pipeline.py --stage gold           # Only gold
"""

import argparse
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    DUCKDB_PATH, RAW_T20I_DIR, RAW_IPL_DIR, RAW_BBL_DIR, RAW_CPL_DIR,
    RAW_REGISTER_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR,
)
from src.ingestion.parse_cricsheet import CricsheetParser, load_people_register
from src.warehouse.schema import initialize_warehouse, get_connection


# ══════════════════════════════════════════════════════════════
# BRONZE LAYER: Ingest raw data as-is
# ══════════════════════════════════════════════════════════════

def load_bronze(conn: duckdb.DuckDBPyConnection):
    """Parse raw JSON files and load into bronze tables."""
    print("\n" + "=" * 60)
    print("🥉 BRONZE LAYER: Raw Data Ingestion")
    print("=" * 60)

    parser = CricsheetParser()

    # Identify available sources
    source_dirs = {}
    for label, directory in [
        ("t20i", RAW_T20I_DIR),
        ("ipl", RAW_IPL_DIR),
        ("bbl", RAW_BBL_DIR),
        ("cpl", RAW_CPL_DIR),
    ]:
        if directory.exists() and any(directory.glob("*.json")):
            source_dirs[label] = directory

    if not source_dirs:
        print("❌ No raw data found. Run: python src/ingestion/download_data.py")
        return

    deliveries_df, matches_df, registry_df = parser.parse_all_sources(source_dirs)

    # Load into DuckDB bronze tables
    if not deliveries_df.is_empty():
        pandas_del = deliveries_df.to_pandas()
        conn.execute("DELETE FROM bronze.raw_deliveries;")
        conn.execute("INSERT INTO bronze.raw_deliveries SELECT *, CURRENT_TIMESTAMP FROM pandas_del")
        print(f"  ✅ Loaded {len(pandas_del)} deliveries → bronze.raw_deliveries")

        # Also save as parquet
        BRONZE_DIR.mkdir(parents=True, exist_ok=True)
        deliveries_df.write_parquet(BRONZE_DIR / "raw_deliveries.parquet")

    if not matches_df.is_empty():
        pandas_mat = matches_df.to_pandas()
        conn.execute("DELETE FROM bronze.raw_matches;")
        conn.execute("INSERT INTO bronze.raw_matches SELECT *, CURRENT_TIMESTAMP FROM pandas_mat")
        print(f"  ✅ Loaded {len(pandas_mat)} matches → bronze.raw_matches")

        matches_df.write_parquet(BRONZE_DIR / "raw_matches.parquet")

    if not registry_df.is_empty():
        pandas_reg = registry_df.to_pandas()
        conn.execute("DELETE FROM bronze.raw_player_registry;")
        conn.execute("INSERT INTO bronze.raw_player_registry SELECT *, CURRENT_TIMESTAMP FROM pandas_reg")
        print(f"  ✅ Loaded {len(pandas_reg)} registry entries → bronze.raw_player_registry")

        registry_df.write_parquet(BRONZE_DIR / "raw_player_registry.parquet")


# ══════════════════════════════════════════════════════════════
# SILVER LAYER: Clean, normalize, enrich
# ══════════════════════════════════════════════════════════════

def load_silver(conn: duckdb.DuckDBPyConnection):
    """Transform bronze → silver with cleaning and enrichment."""
    print("\n" + "=" * 60)
    print("🥈 SILVER LAYER: Cleaning & Normalization")
    print("=" * 60)

    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    # ── Clean Deliveries ──
    conn.execute("DELETE FROM silver.deliveries;")
    conn.execute("""
        INSERT INTO silver.deliveries
        SELECT
            match_id,
            innings,
            TRIM(batting_team) as batting_team,
            TRIM(bowling_team) as bowling_team,
            over,
            ball,
            phase,
            TRIM(batter) as batter,
            TRIM(bowler) as bowler,
            TRIM(non_striker) as non_striker,
            COALESCE(batter_runs, 0) as batter_runs,
            COALESCE(extras_runs, 0) as extras_runs,
            COALESCE(total_runs, 0) as total_runs,
            COALESCE(is_wide, FALSE) as is_wide,
            COALESCE(is_noball, FALSE) as is_noball,
            COALESCE(is_bye, FALSE) as is_bye,
            COALESCE(is_legbye, FALSE) as is_legbye,
            COALESCE(is_wicket, FALSE) as is_wicket,
            wicket_kind,
            player_out,
            COALESCE(is_boundary_four, FALSE) as is_boundary_four,
            COALESCE(is_boundary_six, FALSE) as is_boundary_six,
            COALESCE(is_dot_ball, FALSE) as is_dot_ball,
            -- Computed: legal ball
            (NOT COALESCE(is_wide, FALSE) AND NOT COALESCE(is_noball, FALSE)) as is_legal_ball,
            -- Cumulative runs (window function)
            SUM(COALESCE(total_runs, 0)) OVER (
                PARTITION BY match_id, innings
                ORDER BY over, ball
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as cumulative_runs,
            -- Cumulative wickets
            SUM(CASE WHEN COALESCE(is_wicket, FALSE) THEN 1 ELSE 0 END) OVER (
                PARTITION BY match_id, innings
                ORDER BY over, ball
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as cumulative_wickets,
            -- Current run rate
            CASE
                WHEN over > 0 THEN
                    CAST(SUM(COALESCE(total_runs, 0)) OVER (
                        PARTITION BY match_id, innings
                        ORDER BY over, ball
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS DOUBLE) / (over + 1)
                ELSE 0
            END as run_rate,
            source,
            TRY_CAST(match_date AS DATE) as match_date
        FROM bronze.raw_deliveries
        WHERE match_id IS NOT NULL
          AND batter IS NOT NULL
          AND bowler IS NOT NULL;
    """)

    silver_del_count = conn.execute("SELECT COUNT(*) FROM silver.deliveries").fetchone()[0]
    print(f"  ✅ Cleaned {silver_del_count} deliveries → silver.deliveries")

    # Export to parquet
    silver_del = conn.execute("SELECT * FROM silver.deliveries").pl()
    silver_del.write_parquet(SILVER_DIR / "deliveries.parquet")

    # ── Clean Matches ──
    conn.execute("DELETE FROM silver.matches;")
    conn.execute("""
        INSERT INTO silver.matches
        SELECT
            match_id,
            match_type,
            season,
            TRIM(team1) as team1,
            TRIM(team2) as team2,
            TRY_CAST(match_date AS DATE) as match_date,
            TRIM(venue) as venue,
            TRIM(city) as city,
            toss_winner,
            toss_decision,
            winner,
            result_type,
            result_margin,
            player_of_match,
            event_name,
            event_stage,
            overs_per_side,
            source,
            -- Derived: did toss winner also win the match?
            (toss_winner = winner) as toss_win_match_win,
            -- Bat first team
            CASE
                WHEN toss_decision = 'bat' THEN toss_winner
                WHEN toss_decision = 'field' AND toss_winner = team1 THEN team2
                WHEN toss_decision = 'field' AND toss_winner = team2 THEN team1
                ELSE NULL
            END as bat_first_team,
            -- Chase team
            CASE
                WHEN toss_decision = 'field' THEN toss_winner
                WHEN toss_decision = 'bat' AND toss_winner = team1 THEN team2
                WHEN toss_decision = 'bat' AND toss_winner = team2 THEN team1
                ELSE NULL
            END as chase_team,
            -- Is World Cup match?
            (LOWER(COALESCE(event_name, '')) LIKE '%world cup%'
             OR LOWER(COALESCE(event_name, '')) LIKE '%world twenty20%'
             OR LOWER(COALESCE(event_name, '')) LIKE '%icc men%t20%') as is_world_cup
        FROM bronze.raw_matches
        WHERE match_id IS NOT NULL
          AND team1 IS NOT NULL
          AND team2 IS NOT NULL;
    """)

    silver_match_count = conn.execute("SELECT COUNT(*) FROM silver.matches").fetchone()[0]
    print(f"  ✅ Cleaned {silver_match_count} matches → silver.matches")

    silver_mat = conn.execute("SELECT * FROM silver.matches").pl()
    silver_mat.write_parquet(SILVER_DIR / "matches.parquet")

    # ── Build unique players table ──
    conn.execute("DELETE FROM silver.players;")
    conn.execute("""
        INSERT OR IGNORE INTO silver.players
        SELECT DISTINCT cricsheet_id, player_name
        FROM bronze.raw_player_registry
        WHERE player_name IS NOT NULL;
    """)

    player_count = conn.execute("SELECT COUNT(*) FROM silver.players").fetchone()[0]
    print(f"  ✅ Built {player_count} unique players → silver.players")


# ══════════════════════════════════════════════════════════════
# GOLD LAYER: Aggregations for analytics
# ══════════════════════════════════════════════════════════════

def load_gold(conn: duckdb.DuckDBPyConnection):
    """Transform silver → gold with aggregations and dimension building."""
    print("\n" + "=" * 60)
    print("🥇 GOLD LAYER: Star Schema Aggregation")
    print("=" * 60)

    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build dim_team ──
    conn.execute("DELETE FROM gold.dim_team;")
    conn.execute("""
        INSERT INTO gold.dim_team
        WITH all_teams AS (
            SELECT DISTINCT team1 AS team_name FROM silver.matches
            UNION
            SELECT DISTINCT team2 AS team_name FROM silver.matches
        )
        SELECT
            ROW_NUMBER() OVER (ORDER BY team_name) as team_key,
            team_name,
            team_name IN (
                'India', 'Australia', 'England', 'South Africa', 'New Zealand',
                'Pakistan', 'Sri Lanka', 'West Indies', 'Bangladesh', 'Afghanistan',
                'Zimbabwe', 'Ireland'
            ) as is_icc_full_member
        FROM all_teams
        WHERE team_name IS NOT NULL;
    """)
    team_count = conn.execute("SELECT COUNT(*) FROM gold.dim_team").fetchone()[0]
    print(f"  ✅ Built {team_count} teams → gold.dim_team")

    # ── Build dim_venue ──
    conn.execute("DELETE FROM gold.dim_venue;")
    conn.execute("""
        INSERT INTO gold.dim_venue
        WITH venue_stats AS (
            SELECT
                m.venue as venue_name,
                m.city,
                AVG(CASE WHEN i.innings = 1 THEN i.total ELSE NULL END) as avg_first,
                AVG(CASE WHEN i.innings = 2 THEN i.total ELSE NULL END) as avg_second,
                COUNT(DISTINCT m.match_id) as matches
            FROM silver.matches m
            LEFT JOIN (
                SELECT match_id, innings, SUM(total_runs) as total
                FROM silver.deliveries
                GROUP BY match_id, innings
            ) i ON m.match_id = i.match_id
            WHERE m.venue IS NOT NULL AND TRIM(m.venue) != ''
            GROUP BY m.venue, m.city
        )
        SELECT
            ROW_NUMBER() OVER (ORDER BY venue_name) as venue_key,
            venue_name,
            city,
            NULL as country,
            ROUND(avg_first, 1) as avg_first_innings_score,
            ROUND(avg_second, 1) as avg_second_innings_score,
            matches as matches_hosted
        FROM venue_stats;
    """)
    venue_count = conn.execute("SELECT COUNT(*) FROM gold.dim_venue").fetchone()[0]
    print(f"  ✅ Built {venue_count} venues → gold.dim_venue")

    # ── Build dim_date ──
    conn.execute("DELETE FROM gold.dim_date;")
    conn.execute("""
        INSERT INTO gold.dim_date
        WITH all_dates AS (
            SELECT DISTINCT match_date as full_date
            FROM silver.matches
            WHERE match_date IS NOT NULL
        )
        SELECT
            ROW_NUMBER() OVER (ORDER BY full_date) as date_key,
            full_date,
            EXTRACT(YEAR FROM full_date) as year,
            EXTRACT(MONTH FROM full_date) as month,
            EXTRACT(DAY FROM full_date) as day,
            DAYNAME(full_date) as day_of_week,
            EXTRACT(QUARTER FROM full_date) as quarter,
            DAYOFWEEK(full_date) IN (0, 6) as is_weekend
        FROM all_dates;
    """)
    date_count = conn.execute("SELECT COUNT(*) FROM gold.dim_date").fetchone()[0]
    print(f"  ✅ Built {date_count} dates → gold.dim_date")

    # ── Build dim_tournament ──
    conn.execute("DELETE FROM gold.dim_tournament;")
    conn.execute("""
        INSERT INTO gold.dim_tournament
        WITH tournaments AS (
            SELECT DISTINCT
                event_name,
                season
            FROM silver.matches
            WHERE event_name IS NOT NULL AND TRIM(event_name) != ''
        )
        SELECT
            ROW_NUMBER() OVER (ORDER BY event_name, season) as tournament_key,
            event_name,
            season,
            (LOWER(event_name) LIKE '%world cup%'
             OR LOWER(event_name) LIKE '%world twenty20%') as is_world_cup,
            (LOWER(event_name) LIKE '%ipl%'
             OR LOWER(event_name) LIKE '%premier league%'
             OR LOWER(event_name) LIKE '%big bash%'
             OR LOWER(event_name) LIKE '%caribbean%') as is_league
        FROM tournaments;
    """)
    tournament_count = conn.execute("SELECT COUNT(*) FROM gold.dim_tournament").fetchone()[0]
    print(f"  ✅ Built {tournament_count} tournaments → gold.dim_tournament")

    # ── Build dim_player (from registry + delivery data) ──
    conn.execute("DELETE FROM gold.dim_player;")
    conn.execute("""
        INSERT INTO gold.dim_player
        WITH player_teams AS (
            SELECT
                batter as player_name,
                batting_team as team,
                COUNT(DISTINCT match_id) as matches
            FROM silver.deliveries
            GROUP BY batter, batting_team
        ),
        player_agg AS (
            SELECT
                player_name,
                STRING_AGG(DISTINCT team, ', ') as teams_played,
                SUM(matches) as total_matches
            FROM player_teams
            GROUP BY player_name
        )
        SELECT
            ROW_NUMBER() OVER (ORDER BY pa.player_name) as player_key,
            sp.cricsheet_id,
            pa.player_name,
            pa.teams_played,
            pa.total_matches as matches_played,
            NULL as batting_style,
            NULL as bowling_style,
            NULL as role
        FROM player_agg pa
        LEFT JOIN silver.players sp ON pa.player_name = sp.player_name;
    """)
    player_count = conn.execute("SELECT COUNT(*) FROM gold.dim_player").fetchone()[0]
    print(f"  ✅ Built {player_count} players → gold.dim_player")

    # ── Build fact_innings_summary ──
    conn.execute("DELETE FROM gold.fact_innings_summary;")
    conn.execute("""
        INSERT INTO gold.fact_innings_summary
        SELECT
            d.match_id,
            d.innings,
            t.team_key,
            v.venue_key,
            SUM(d.total_runs) as total_runs,
            SUM(CASE WHEN d.is_wicket THEN 1 ELSE 0 END) as total_wickets,
            ROUND(MAX(d.over) + 1.0, 1) as total_overs,
            ROUND(CAST(SUM(d.total_runs) AS DOUBLE) / NULLIF(MAX(d.over) + 1, 0), 2) as run_rate,
            -- Powerplay
            SUM(CASE WHEN d.phase = 'powerplay' THEN d.total_runs ELSE 0 END) as pp_runs,
            SUM(CASE WHEN d.phase = 'powerplay' AND d.is_wicket THEN 1 ELSE 0 END) as pp_wickets,
            ROUND(CAST(SUM(CASE WHEN d.phase = 'powerplay' THEN d.total_runs ELSE 0 END) AS DOUBLE) / 6.0, 2) as pp_run_rate,
            -- Middle overs
            SUM(CASE WHEN d.phase = 'middle' THEN d.total_runs ELSE 0 END) as middle_runs,
            SUM(CASE WHEN d.phase = 'middle' AND d.is_wicket THEN 1 ELSE 0 END) as middle_wickets,
            ROUND(CAST(SUM(CASE WHEN d.phase = 'middle' THEN d.total_runs ELSE 0 END) AS DOUBLE) / 9.0, 2) as middle_run_rate,
            -- Death overs
            SUM(CASE WHEN d.phase = 'death' THEN d.total_runs ELSE 0 END) as death_runs,
            SUM(CASE WHEN d.phase = 'death' AND d.is_wicket THEN 1 ELSE 0 END) as death_wickets,
            ROUND(CAST(SUM(CASE WHEN d.phase = 'death' THEN d.total_runs ELSE 0 END) AS DOUBLE) / 5.0, 2) as death_run_rate,
            -- Boundaries
            SUM(CASE WHEN d.is_boundary_four OR d.is_boundary_six THEN d.batter_runs ELSE 0 END) as boundary_runs,
            SUM(CASE WHEN d.is_boundary_four THEN 1 ELSE 0 END) as fours,
            SUM(CASE WHEN d.is_boundary_six THEN 1 ELSE 0 END) as sixes,
            SUM(CASE WHEN d.is_dot_ball THEN 1 ELSE 0 END) as dot_balls,
            SUM(d.extras_runs) as extras,
            d.source
        FROM silver.deliveries d
        LEFT JOIN gold.dim_team t ON d.batting_team = t.team_name
        LEFT JOIN silver.matches m ON d.match_id = m.match_id
        LEFT JOIN gold.dim_venue v ON m.venue = v.venue_name
        GROUP BY d.match_id, d.innings, t.team_key, v.venue_key, d.source;
    """)
    innings_count = conn.execute("SELECT COUNT(*) FROM gold.fact_innings_summary").fetchone()[0]
    print(f"  ✅ Built {innings_count} innings summaries → gold.fact_innings_summary")

    # Export gold tables as parquet
    for table in ["dim_team", "dim_venue", "dim_date", "dim_tournament", "dim_player",
                   "fact_innings_summary"]:
        df = conn.execute(f"SELECT * FROM gold.{table}").pl()
        df.write_parquet(GOLD_DIR / f"{table}.parquet")

    print(f"\n  📁 Gold parquet files saved to: {GOLD_DIR}")


# ══════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATION
# ══════════════════════════════════════════════════════════════

def run_full_pipeline():
    """Run the complete ETL pipeline."""
    print("=" * 60)
    print("🚀 ICC T20 WC PREDICTOR — ETL PIPELINE")
    print("=" * 60)

    # Initialize warehouse
    initialize_warehouse()

    conn = get_connection()

    try:
        load_bronze(conn)
        load_silver(conn)
        load_gold(conn)
    finally:
        conn.close()

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run ETL pipeline")
    parser.add_argument(
        "--stage",
        choices=["bronze", "silver", "gold", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    if args.stage == "all":
        run_full_pipeline()
    else:
        # Initialize if needed
        if not DUCKDB_PATH.exists():
            initialize_warehouse()

        conn = get_connection()
        try:
            if args.stage == "bronze":
                load_bronze(conn)
            elif args.stage == "silver":
                load_silver(conn)
            elif args.stage == "gold":
                load_gold(conn)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
