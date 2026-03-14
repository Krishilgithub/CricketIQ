"""Shared utilities, DB connections and cached data loaders used across all pages."""
import os, pickle
import streamlit as st
import pandas as pd
import duckdb
from src.config import get_config, resolve_path

# ── DB Connections ─────────────────────────────────────────────────────────

@st.cache_resource
def get_hub_con():
    cfg = get_config()
    return duckdb.connect(str(resolve_path(cfg["paths"]["duckdb_path"])), read_only=True)


@st.cache_resource
def load_model():
    cfg = get_config()
    db_path = str(resolve_path(cfg["paths"]["models_dir"])) + "/champion_model.pkl"
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            return pickle.load(f)
    return None


# ── Data Loaders ────────────────────────────────────────────────────────────

@st.cache_data
def get_teams():
    con = get_hub_con()
    return con.execute("SELECT DISTINCT team_1 FROM main_gold.fact_matches ORDER BY team_1").df()["team_1"].tolist()


@st.cache_data
def get_venues():
    con = get_hub_con()
    return con.execute("SELECT DISTINCT venue FROM main_gold.fact_matches ORDER BY venue").df()["venue"].tolist()


@st.cache_data
def get_global_kpis():
    con = get_hub_con()
    q = """
    SELECT
        (SELECT COUNT(*) FROM main_gold.fact_matches) as total_matches,
        (SELECT SUM(runs_batter + runs_extras) FROM main_gold.fact_deliveries) as total_runs,
        (SELECT COUNT(*) FROM main_gold.fact_wickets) as total_wickets,
        (SELECT COUNT(DISTINCT batter) FROM main_gold.fact_deliveries) as total_players
    """
    try:
        return con.execute(q).df().iloc[0]
    except Exception:
        return pd.Series({"total_matches": 0, "total_runs": 0, "total_wickets": 0, "total_players": 0})


@st.cache_data
def get_h2h_rate(team1):
    con = get_hub_con()
    q = f"""
    SELECT COUNT(*) as total,
           SUM(CASE WHEN winner = '{team1}' THEN 1 ELSE 0 END) as wins
    FROM main_gold.fact_matches
    WHERE (toss_winner = '{team1}' OR winner = '{team1}')
    """
    row = con.execute(q).df().iloc[0]
    return float(row["wins"]) / max(float(row["total"]), 1)


@st.cache_data
def get_venue_avg(venue):
    con = get_hub_con()
    q = f"""
    SELECT AVG(i.total_runs) as avg_runs
    FROM main_gold.fact_innings i
    JOIN main_gold.fact_matches m ON i.match_id = m.match_id
    WHERE m.venue = '{venue}' AND i.innings_number = 1
    """
    row = con.execute(q).df().iloc[0]
    return float(row["avg_runs"] or 150)


@st.cache_data
def get_team_form(team1):
    con = get_hub_con()
    q = f"""
    SELECT ROUND(AVG(team_1_win)::FLOAT, 3) as form
    FROM (
        SELECT team_1_win FROM main_gold.fact_matches
        WHERE toss_winner = '{team1}'
        ORDER BY match_date DESC LIMIT 5
    )
    """
    row = con.execute(q).df().iloc[0]
    return float(row["form"] or 0.5)


@st.cache_data
def get_phase_data(venue_sel):
    con = get_hub_con()
    q = f"""
    SELECT
        CASE WHEN d.over_number <= 5 THEN 'Powerplay (0-5)'
             WHEN d.over_number <= 14 THEN 'Middle (6-14)'
             ELSE 'Death (15-19)' END as phase,
        ROUND(SUM(d.runs_batter + d.runs_extras)::FLOAT / NULLIF(SUM(d.is_legal_ball), 0) * 6, 2) as run_rate,
        ROUND(SUM(d.is_wicket)::FLOAT / NULLIF(SUM(d.is_legal_ball), 0) * 6, 4) as wicket_rate,
        COUNT(*) as balls
    FROM main_gold.fact_deliveries d
    JOIN main_gold.fact_matches m ON d.match_id = m.match_id
    WHERE m.venue = '{venue_sel}'
    GROUP BY 1 ORDER BY 1
    """
    return con.execute(q).df()


@st.cache_data
def get_toss_recommendation(venue_sel):
    con = get_hub_con()
    q = f"""
    SELECT toss_decision,
           COUNT(*) total,
           SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) as toss_wins
    FROM main_gold.fact_matches
    WHERE venue = '{venue_sel}' AND result_type NOT IN ('no result', 'tie')
    GROUP BY toss_decision
    """
    return con.execute(q).df()


@st.cache_data
def get_top_batters():
    con = get_hub_con()
    q = """
    SELECT batter, SUM(runs_batter) as runs,
           SUM(is_legal_ball) as balls,
           ROUND(SUM(runs_batter)::FLOAT / NULLIF(SUM(is_legal_ball),0) * 100, 2) as sr
    FROM main_gold.fact_deliveries d
    JOIN main_gold.fact_matches m ON d.match_id = m.match_id
    WHERE m.match_date >= (CURRENT_DATE - INTERVAL 730 DAY)
    GROUP BY batter HAVING balls > 100
    ORDER BY runs DESC LIMIT 15
    """
    return con.execute(q).df()


@st.cache_data
def get_top_bowlers():
    con = get_hub_con()
    q = """
    SELECT bowler, SUM(is_wicket) as wickets,
           SUM(is_legal_ball) as balls,
           ROUND(SUM(runs_batter + runs_extras)::FLOAT / NULLIF(SUM(is_legal_ball),0) * 6, 2) as econ
    FROM main_gold.fact_deliveries d
    JOIN main_gold.fact_matches m ON d.match_id = m.match_id
    WHERE m.match_date >= (CURRENT_DATE - INTERVAL 730 DAY)
    GROUP BY bowler HAVING balls > 100
    ORDER BY wickets DESC LIMIT 15
    """
    return con.execute(q).df()


@st.cache_data
def get_venue_heatmap():
    con = get_hub_con()
    q = """
    SELECT t.team as team, m.venue as venue,
           ROUND(AVG(CASE WHEN m.winner = t.team THEN 1.0 ELSE 0.0 END) * 100, 0) as win_pct,
           COUNT(m.match_id) as matches
    FROM main_gold.fact_matches m
    JOIN main_silver.slv_match_teams t ON m.match_id = t.match_id
    GROUP BY t.team, m.venue HAVING matches >= 5
    ORDER BY win_pct DESC
    """
    return con.execute(q).df()


@st.cache_data
def get_exciting_matches():
    con = get_hub_con()
    q = """
    SELECT m.match_id, m.match_date, m.event_name, m.venue, m.team_1, m.winner,
           m.result_margin,
           i1.total_runs as target,
           i2.total_runs as chased,
           ABS(i1.total_runs - i2.total_runs) as margin_runs
    FROM main_gold.fact_matches m
    JOIN main_gold.fact_innings i1 ON m.match_id = i1.match_id AND i1.innings_number = 1
    JOIN main_gold.fact_innings i2 ON m.match_id = i2.match_id AND i2.innings_number = 2
    WHERE m.result_type IN ('runs', 'wickets')
    ORDER BY margin_runs ASC
    LIMIT 20
    """
    return con.execute(q).df()


@st.cache_data
def get_highest_scores():
    con = get_hub_con()
    q = """
    SELECT batter, SUM(runs_batter) as innings_runs, m.event_name, m.match_date, team_1, winner
    FROM main_gold.fact_deliveries d
    JOIN main_gold.fact_matches m ON d.match_id = m.match_id
    GROUP BY d.batter, d.match_id, d.innings_number, m.event_name, m.match_date, m.team_1, m.winner
    ORDER BY innings_runs DESC LIMIT 10
    """
    return con.execute(q).df()


@st.cache_data
def get_best_bowling():
    con = get_hub_con()
    q = """
    SELECT bowler, SUM(is_wicket) as wickets, SUM(runs_batter + runs_extras) as runs_conceded,
           m.event_name, m.match_date
    FROM main_gold.fact_deliveries d
    JOIN main_gold.fact_matches m ON d.match_id = m.match_id
    GROUP BY d.bowler, d.match_id, d.innings_number, m.event_name, m.match_date
    HAVING wickets >= 3
    ORDER BY wickets DESC, runs_conceded ASC LIMIT 10
    """
    return con.execute(q).df()


# ── Shared CSS ─────────────────────────────────────────────────────────────

THEME_CSS = """
<style>
    [data-testid="stSidebar"] { background-color: #0b0f19; }
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .metric-card { background: #1e293b; border-radius: 12px; padding: 20px; margin: 8px 0; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
    .section-header { font-size: 1.5rem; font-weight: 700; color: #38bdf8; margin-bottom: 16px; border-bottom: 2px solid #1e293b; padding-bottom: 8px; }
    .chat-msg-user { background: #0284c7; border-radius: 12px 12px 0 12px; padding: 12px 16px; margin: 4px 0; color: white; float: right; clear: both; max-width: 80%; }
    .chat-msg-bot { background: #1e293b; border-radius: 12px 12px 12px 0; padding: 12px 16px; margin: 4px 0; border: 1px solid #334155; color: #f8fafc; float: left; clear: both; max-width: 80%; }
    .stTextInput > div > div > input { background: #1e293b; color: white; border: 1px solid #334155; border-radius: 8px; }
    .st-expander { background-color: #1e293b !important; border: 1px solid #334155 !important; border-radius: 8px !important; }
    .stCodeBlock { background-color: #0b0f19 !important; border-radius: 6px !important; border: 1px solid #334155 !important; }
    .kpi-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 24px; border: 1px solid #334155; text-align: center; }
    .kpi-value { font-size: 2rem; font-weight: 800; color: #38bdf8; }
    .kpi-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .prediction-card { background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); border-radius: 16px; padding: 24px; border: 1px solid #1d4ed8; }
</style>
"""
