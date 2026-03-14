"""
Natural Language to SQL — Translate user questions into DuckDB SQL queries.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import GEMINI_API_KEY, DUCKDB_PATH


SCHEMA_DESCRIPTION = """
Database: DuckDB with 3 schemas (bronze, silver, gold).

Key tables in the Gold layer (Star Schema):

gold.dim_team: team_key (PK), team_name, is_icc_full_member
gold.dim_player: player_key (PK), cricsheet_id, player_name, teams_played, matches_played, role
gold.dim_venue: venue_key (PK), venue_name, city, avg_first_innings_score, avg_second_innings_score, matches_hosted
gold.dim_date: date_key (PK), full_date, year, month, day, day_of_week, quarter, is_weekend
gold.dim_tournament: tournament_key (PK), event_name, season, is_world_cup, is_league

gold.fact_match_results: match_id (PK), date_key (FK), venue_key (FK), tournament_key (FK), team1_key (FK), team2_key (FK), winner_key (FK), toss_winner_key (FK), toss_decision, result_type, result_margin, player_of_match_key (FK), team1_score, team2_score, source
gold.fact_batting_innings: match_id, innings, player_key (FK), team_key (FK), venue_key (FK), date_key (FK), runs_scored, balls_faced, fours, sixes, dot_balls_faced, strike_rate, is_out, dismissal_type, batting_position, pp_runs, middle_runs, death_runs, source
gold.fact_bowling_innings: match_id, innings, player_key (FK), team_key (FK), venue_key (FK), date_key (FK), overs_bowled, balls_bowled, runs_conceded, wickets_taken, economy_rate, dot_balls, wides, noballs, source
gold.fact_innings_summary: match_id, innings, team_key (FK), venue_key (FK), total_runs, total_wickets, total_overs, run_rate, pp_runs, pp_wickets, pp_run_rate, middle_runs, middle_wickets, death_runs, death_wickets, boundary_runs, fours, sixes, dot_balls, extras, source

Silver layer:
silver.matches: match_id (PK), match_type, season, team1, team2, match_date, venue, city, toss_winner, toss_decision, winner, result_type, result_margin, player_of_match, event_name, source, bat_first_team, chase_team, is_world_cup
silver.deliveries: match_id, innings, batting_team, bowling_team, over, ball, phase, batter, bowler, batter_runs, total_runs, is_wicket, is_boundary_four, is_boundary_six, is_dot_ball, cumulative_runs, cumulative_wickets, run_rate, source, match_date
"""


def nl_to_sql(question: str) -> dict:
    """
    Convert natural language question to SQL and execute on DuckDB.

    Returns:
        dict with keys: 'sql', 'result', 'error'
    """
    if not GEMINI_API_KEY:
        return {"sql": "", "result": None, "error": "GEMINI_API_KEY not set."}

    try:
        import google.generativeai as genai
        import duckdb
    except ImportError:
        return {"sql": "", "result": None, "error": "Required packages not installed."}

    prompt = f"""You are a SQL expert. Generate a DuckDB SQL query for the following question.

Database Schema:
{SCHEMA_DESCRIPTION}

Rules:
1. Return ONLY the SQL query, no explanation.
2. Use DuckDB syntax.
3. Always LIMIT results to 50 rows max.
4. Use Gold layer tables when possible (better aggregated data).
5. Fall back to Silver layer for detailed ball-by-ball queries.
6. Use JOINs to connect fact tables with dimension tables for readable output.
7. Do NOT use markdown code fences or backticks.

Question: {question}

SQL:"""

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    sql = response.text.strip()
    # Clean up any markdown artifacts
    sql = sql.replace("```sql", "").replace("```", "").strip()

    try:
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        result = conn.execute(sql).df()
        conn.close()
        return {"sql": sql, "result": result, "error": None}
    except Exception as e:
        return {"sql": sql, "result": None, "error": str(e)}


if __name__ == "__main__":
    result = nl_to_sql("Which team has won the most T20 World Cup matches?")
    print(f"SQL: {result['sql']}")
    if result['result'] is not None:
        print(f"Result:\n{result['result']}")
    else:
        print(f"Error: {result['error']}")
