"""
Power BI Export — Export Gold-layer data as flat files with pre-calculated KPI columns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import GOLD_DIR, DATA_DIR


EXPORT_DIR = DATA_DIR / "powerbi_export"


def export_for_powerbi(conn):
    """Export all Gold tables + pre-computed KPIs as CSV for Power BI import."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Direct dimension exports
    for table in ["dim_team", "dim_player", "dim_venue", "dim_date", "dim_tournament"]:
        df = conn.execute(f"SELECT * FROM gold.{table}").df()
        df.to_csv(EXPORT_DIR / f"{table}.csv", index=False)
        print(f"  ✅ Exported {table} ({len(df)} rows)")

    # 2. Fact tables
    for table in ["fact_innings_summary", "fact_batting_innings",
                   "fact_bowling_innings", "fact_match_results"]:
        try:
            df = conn.execute(f"SELECT * FROM gold.{table}").df()
            df.to_csv(EXPORT_DIR / f"{table}.csv", index=False)
            print(f"  ✅ Exported {table} ({len(df)} rows)")
        except Exception as e:
            print(f"  ⚠️ {table}: {e}")

    # 3. Pre-computed KPI tables for each persona

    # Coach KPIs: team-level win rates
    try:
        coach_kpis = conn.execute("""
            SELECT t.team_name,
                   COUNT(*) AS total_matches,
                   SUM(CASE WHEN m.winner = t.team_name THEN 1 ELSE 0 END) AS wins,
                   ROUND(SUM(CASE WHEN m.winner = t.team_name THEN 1.0 ELSE 0 END)
                         / COUNT(*) * 100, 1) AS win_pct,
                   ROUND(AVG(fi.total_runs), 1) AS avg_score,
                   ROUND(AVG(fi.run_rate), 2) AS avg_rr,
                   ROUND(AVG(fi.pp_run_rate), 2) AS avg_pp_rr,
                   ROUND(AVG(fi.death_run_rate), 2) AS avg_death_rr,
                   ROUND(AVG(fi.boundary_runs) * 100.0 / NULLIF(AVG(fi.total_runs), 0), 1) AS boundary_pct,
                   ROUND(AVG(fi.dot_balls), 1) AS avg_dots,
                   m.source
            FROM gold.dim_team t
            JOIN silver.matches m ON (m.team1 = t.team_name OR m.team2 = t.team_name)
            LEFT JOIN gold.fact_innings_summary fi ON fi.team_key = t.team_key
            WHERE m.winner IS NOT NULL
            GROUP BY t.team_name, m.source
            ORDER BY win_pct DESC
        """).df()
        coach_kpis.to_csv(EXPORT_DIR / "kpi_coach_team_performance.csv", index=False)
        print(f"  ✅ Exported kpi_coach_team_performance ({len(coach_kpis)} rows)")
    except Exception as e:
        print(f"  ⚠️ Coach KPIs: {e}")

    # Analyst KPIs: player impact scores
    try:
        analyst_kpis = conn.execute("""
            WITH batting AS (
                SELECT player_key, SUM(runs_scored) AS runs, SUM(fours) f, SUM(sixes) s, COUNT(*) inn
                FROM gold.fact_batting_innings GROUP BY player_key
            ),
            bowling AS (
                SELECT player_key, SUM(wickets_taken) wkts, COUNT(*) inn
                FROM gold.fact_bowling_innings GROUP BY player_key
            )
            SELECT p.player_name,
                   COALESCE(ba.runs, 0) AS total_runs,
                   COALESCE(bo.wkts, 0) AS total_wickets,
                   COALESCE(ba.f, 0) AS fours, COALESCE(ba.s, 0) AS sixes,
                   GREATEST(COALESCE(ba.inn, 0), COALESCE(bo.inn, 0)) AS matches,
                   ROUND((COALESCE(ba.runs, 0) + COALESCE(bo.wkts, 0) * 25
                          + COALESCE(ba.f, 0) * 2 + COALESCE(ba.s, 0) * 3)
                         * 1.0 / GREATEST(COALESCE(ba.inn, 0), COALESCE(bo.inn, 0), 1), 1) AS impact_score
            FROM gold.dim_player p
            LEFT JOIN batting ba ON p.player_key = ba.player_key
            LEFT JOIN bowling bo ON p.player_key = bo.player_key
            WHERE GREATEST(COALESCE(ba.inn, 0), COALESCE(bo.inn, 0)) >= 5
            ORDER BY impact_score DESC
        """).df()
        analyst_kpis.to_csv(EXPORT_DIR / "kpi_analyst_player_impact.csv", index=False)
        print(f"  ✅ Exported kpi_analyst_player_impact ({len(analyst_kpis)} rows)")
    except Exception as e:
        print(f"  ⚠️ Analyst KPIs: {e}")

    # H2H records for broadcaster
    try:
        h2h = conn.execute("""
            SELECT team1, team2, winner,
                   COUNT(*) AS matches,
                   source
            FROM silver.matches
            WHERE winner IS NOT NULL
            GROUP BY team1, team2, winner, source
        """).df()
        h2h.to_csv(EXPORT_DIR / "kpi_broadcaster_h2h.csv", index=False)
        print(f"  ✅ Exported kpi_broadcaster_h2h ({len(h2h)} rows)")
    except Exception as e:
        print(f"  ⚠️ Broadcaster KPIs: {e}")

    print(f"\n📁 All exports saved to: {EXPORT_DIR}")
    print("📊 Import these CSV files into Power BI Desktop to build reports.")


if __name__ == "__main__":
    from src.warehouse.schema import get_connection

    conn = get_connection()
    print("=" * 60)
    print("📊 POWER BI DATA EXPORT")
    print("=" * 60)
    export_for_powerbi(conn)
    conn.close()
