"""
src/genai/rag_context.py
───────────────────────────
RAG (Retrieval-Augmented Generation) context builder for the CricketIQ chatbot.

Converts natural language queries into DuckDB SQL to fetch grounded cricket facts,
which are then passed as context to the LLM for accurate, hallucination-free answers.

Supported query intents:
  - team_stats   → win rate, form, venue breakdown for a team
  - player_stats → batting and bowling career figures
  - venue_stats  → scoring patterns, chase rates
  - h2h          → head-to-head record between two teams
  - top_n        → leaderboard queries (scorer, wicket-taker etc.)
  - prediction   → pre-match win probability
"""

import re
import duckdb
from typing import Optional
from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)


def get_con() -> duckdb.DuckDBPyConnection:
    cfg = get_config()
    return duckdb.connect(str(resolve_path(cfg["paths"]["duckdb_path"])), read_only=True)


# ── Intent Detection ─────────────────────────────────────────────────────────

def detect_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["win rate", "form", "record of", "how has", "performance of"]):
        return "team_stats"
    if any(w in q for w in ["batting", "bowling", "player", "batter", "bowler", "scorer", "wicket taker"]):
        return "player_stats"
    if any(w in q for w in ["venue", "ground", "stadium", "pitch"]):
        return "venue_stats"
    if any(w in q for w in ["vs ", "versus", "head to head", "h2h", "against"]):
        return "h2h"
    if any(w in q for w in ["top ", "best ", "most runs", "most wickets", "highest", "lowest"]):
        return "top_n"
    if any(w in q for w in ["predict", "who will win", "probability", "chance", "likely"]):
        return "prediction"
    return "general"


def extract_entities(query: str, con: duckdb.DuckDBPyConnection) -> dict:
    """Fuzzy-match team and player names from the query against the DB."""
    q_lower = query.lower()

    teams = con.execute("SELECT DISTINCT team_1 FROM main_gold.fact_matches").df()["team_1"].tolist()
    matched_teams = [t for t in teams if t.lower() in q_lower]

    players = con.execute("""
        SELECT DISTINCT batter as name FROM main_gold.fact_deliveries
        UNION SELECT DISTINCT bowler FROM main_gold.fact_deliveries
    """).df()["name"].tolist()
    matched_players = [p for p in players if p.lower() in q_lower]

    venues = con.execute("SELECT DISTINCT venue FROM main_gold.fact_matches").df()["venue"].tolist()
    matched_venues = [v for v in venues if v.lower() in q_lower]

    return {
        "teams": matched_teams[:2],
        "players": matched_players[:1],
        "venues": matched_venues[:1],
    }


# ── Context Retrievers ───────────────────────────────────────────────────────

def fetch_team_context(con, team: str) -> str:
    row = con.execute(f"""
        SELECT COUNT(*) matches,
               ROUND(AVG(CASE WHEN winner=? THEN 1.0 ELSE 0 END)*100,1) win_pct,
               ROUND(AVG(CASE WHEN toss_winner=? AND winner=? THEN 1.0 ELSE 0 END)*100,1) toss_win_pct
        FROM main_gold.fact_matches m
        JOIN main_silver.slv_match_teams t ON m.match_id=t.match_id
        WHERE t.team=?
    """, [team, team, team, team]).fetchone()

    recent = con.execute(f"""
        SELECT m.match_date, m.venue, m.winner
        FROM main_gold.fact_matches m
        JOIN main_silver.slv_match_teams t ON m.match_id=t.match_id
        WHERE t.team=? ORDER BY m.match_date DESC LIMIT 5
    """, [team]).df().to_string(index=False)

    return (
        f"Team: {team}\n"
        f"Total matches: {row[0]}, Win %: {row[1]}%, Win % when winning toss: {row[2]}%\n"
        f"Last 5 matches:\n{recent}"
    )


def fetch_player_context(con, player: str) -> str:
    bat = con.execute(f"""
        SELECT SUM(runs_batter) runs, SUM(is_legal_ball) balls,
               ROUND(SUM(runs_batter)::FLOAT/NULLIF(SUM(is_legal_ball),0)*100,1) sr,
               COUNT(DISTINCT match_id) innings
        FROM main_gold.fact_deliveries WHERE batter=?
    """, [player]).fetchone()

    bowl = con.execute(f"""
        SELECT SUM(is_wicket) wkts,
               ROUND(SUM(runs_batter+runs_extras)::FLOAT/NULLIF(SUM(is_legal_ball),0)*6,2) econ
        FROM main_gold.fact_deliveries WHERE bowler=?
    """, [player]).fetchone()

    return (
        f"Player: {player}\n"
        f"Batting → {bat[3]} innings, {bat[0]} runs, SR: {bat[2]}\n"
        f"Bowling → {bowl[0]} wickets, Economy: {bowl[1]}"
    )


def fetch_venue_context(con, venue: str) -> str:
    row = con.execute(f"""
        SELECT COUNT(*) matches,
               ROUND(AVG(i1.total_runs)::FLOAT,1) avg_1st,
               ROUND(AVG(i2.total_runs)::FLOAT,1) avg_2nd
        FROM main_gold.fact_matches m
        JOIN main_gold.fact_innings i1 ON m.match_id=i1.match_id AND i1.innings_number=1
        JOIN main_gold.fact_innings i2 ON m.match_id=i2.match_id AND i2.innings_number=2
        WHERE m.venue=?
    """, [venue]).fetchone()

    return (
        f"Venue: {venue}\n"
        f"Matches played: {row[0]}, Avg 1st innings: {row[1]}, Avg 2nd innings: {row[2]}"
    )


def fetch_h2h_context(con, team1: str, team2: str) -> str:
    rows = con.execute(f"""
        SELECT m.match_date, m.venue, m.winner, m.result_margin, m.result_type
        FROM main_gold.fact_matches m
        JOIN main_silver.slv_match_teams t1 ON m.match_id=t1.match_id
        JOIN main_silver.slv_match_teams t2 ON m.match_id=t2.match_id
        WHERE t1.team=? AND t2.team=?
        ORDER BY m.match_date DESC LIMIT 8
    """, [team1, team2]).df()

    t1_wins = (rows["winner"] == team1).sum()
    t2_wins = (rows["winner"] == team2).sum()
    return (
        f"H2H: {team1} vs {team2} — Last {len(rows)} matches\n"
        f"{team1} wins: {t1_wins}, {team2} wins: {t2_wins}\n"
        + rows.to_string(index=False)
    )


def fetch_top_n_context(con, query: str) -> str:
    q = query.lower()
    if "wicket" in q or "bowl" in q:
        df = con.execute("""
            SELECT bowler, SUM(is_wicket) wkts
            FROM main_gold.fact_deliveries
            GROUP BY bowler ORDER BY wkts DESC LIMIT 10
        """).df()
        return f"Top 10 wicket-takers:\n{df.to_string(index=False)}"
    else:
        df = con.execute("""
            SELECT batter, SUM(runs_batter) runs
            FROM main_gold.fact_deliveries
            GROUP BY batter ORDER BY runs DESC LIMIT 10
        """).df()
        return f"Top 10 run scorers:\n{df.to_string(index=False)}"


# ── Main Entry Point ─────────────────────────────────────────────────────────

def build_rag_context(query: str) -> str:
    """
    Given a natural language cricket query, returns a structured
    context string pulled from the DuckDB Gold Layer for LLM grounding.
    """
    con = get_con()
    intent = detect_intent(query)
    entities = extract_entities(query, con)

    log.info(f"RAG intent={intent} entities={entities}")

    try:
        if intent == "team_stats" and entities["teams"]:
            return fetch_team_context(con, entities["teams"][0])

        elif intent == "player_stats" and entities["players"]:
            return fetch_player_context(con, entities["players"][0])

        elif intent == "venue_stats" and entities["venues"]:
            return fetch_venue_context(con, entities["venues"][0])

        elif intent == "h2h" and len(entities["teams"]) >= 2:
            return fetch_h2h_context(con, entities["teams"][0], entities["teams"][1])

        elif intent == "top_n":
            return fetch_top_n_context(con, query)

        else:
            # Fallback: general T20I summary
            summary = con.execute("""
                SELECT COUNT(*) total_matches,
                       COUNT(DISTINCT team_1) teams,
                       MIN(match_date) since,
                       MAX(match_date) latest
                FROM main_gold.fact_matches
            """).fetchone()
            return (
                f"CricketIQ DB covers {summary[0]:,} T20I matches across {summary[1]} teams "
                f"from {summary[2]} to {summary[3]}."
            )
    except Exception as e:
        log.warning(f"RAG context build failed: {e}")
        return "No specific data found for this query in the CricketIQ database."
    finally:
        con.close()
