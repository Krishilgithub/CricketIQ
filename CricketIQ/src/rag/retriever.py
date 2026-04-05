"""
src/rag/retriever.py
─────────────────────
Entity extraction and database retrieval logic.
"""

import duckdb
from typing import Dict, Any
from thefuzz import process, fuzz
from langsmith import traceable

from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

def get_con() -> duckdb.DuckDBPyConnection:
    cfg = get_config()
    return duckdb.connect(str(resolve_path(cfg["paths"]["duckdb_path"])), read_only=True)

@traceable(run_type="retriever", name="Entity & Schema Retrieval")
def extract_entities(query: str, con: duckdb.DuckDBPyConnection) -> dict:
    q_lower = query.lower()

    teams_df = con.execute("SELECT DISTINCT team_1 FROM main_gold.fact_matches").df()
    teams = [str(x) for x in teams_df["team_1"].tolist() if x]
    matched_teams = [t for t in teams if t.lower() in q_lower]
    if not matched_teams:
        fuz_teams = process.extract(q_lower, teams, scorer=fuzz.token_set_ratio, limit=2)
        matched_teams = [t[0] for t in fuz_teams if t[1] > 82]

    players_df = con.execute("SELECT DISTINCT batter as name FROM main_gold.fact_deliveries UNION SELECT DISTINCT bowler FROM main_gold.fact_deliveries").df()
    players = [str(x) for x in players_df["name"].tolist() if x]
    matched_players = [p for p in players if p.lower() in q_lower]
    if not matched_players:
        fuz_players = process.extract(q_lower, players, scorer=fuzz.token_set_ratio, limit=2)
        matched_players = [p[0] for p in fuz_players if p[1] > 82]

    venues_df = con.execute("SELECT DISTINCT venue FROM main_gold.fact_matches").df()
    venues = [str(x) for x in venues_df["venue"].tolist() if x]
    matched_venues = [v for v in venues if v.lower() in q_lower]
    if not matched_venues:
        fuz_venues = process.extract(q_lower, venues, scorer=fuzz.token_set_ratio, limit=1)
        matched_venues = [v[0] for v in fuz_venues if v[1] > 82]

    return {
        "teams": matched_teams,
        "players": matched_players,
        "venues": matched_venues,
    }

@traceable(run_type="tool", name="Execute DuckDB SQL")
def execute_sql(query: str) -> str:
    """Executes a Read-Only DuckDB query and returns the result as a string."""
    con = get_con()
    try:
        df = con.execute(query).df()
        if df.empty:
            return "No results found for that query."
        return df.to_string(index=False)
    except Exception as e:
        log.warning(f"SQL Execution Failed: {e}")
        return f"SQL Error: {str(e)}\nHint: Ensure tables are prefixed with 'main_gold.' (e.g., main_gold.fact_matches)"
    finally:
        con.close()
