"""Shared utilities, DB connections and cached data loaders used across all pages."""
import os, pickle
import streamlit as st
import pandas as pd
import duckdb
from src.config import get_config, resolve_path

# ── DB Availability Check ──────────────────────────────────────────────────

# Module-level cache — avoids @st.cache_resource interfering with exception handling
_DB_CON = None
_DB_CHECKED = False


def get_hub_con():
    """Return a DuckDB connection, or None if the database file doesn't exist."""
    global _DB_CON, _DB_CHECKED
    if _DB_CHECKED:
        return _DB_CON
    _DB_CHECKED = True
    try:
        cfg = get_config()
        db_path = str(resolve_path(cfg["paths"]["duckdb_path"]))
        if not os.path.exists(db_path):
            _DB_CON = None
            return None
        _DB_CON = duckdb.connect(db_path, read_only=True)
        return _DB_CON
    except:  # bare except — catches C-extension duckdb.IOException too
        _DB_CON = None
        return None


def db_available() -> bool:
    """Check whether the DuckDB database is accessible."""
    return get_hub_con() is not None


def show_db_unavailable_warning():
    """Display a consistent warning when the database is not present."""
    st.warning(
        "⚠️ **Database not available on this deployment.**\n\n"
        "The CricketIQ analytics database (`cricketiq.duckdb`) is built locally from "
        "Cricsheet match data using the dbt pipeline. It is not included in the "
        "repository due to its large size.\n\n"
        "**To use the full analytics features:**\n"
        "1. Clone the repo locally\n"
        "2. Run `python -m src.etl.build_db` to build the database\n"
        "3. Run the app with `streamlit run CricketIQ/src/app.py`\n\n"
        "The **AI Chat Bot** and **Match Prediction** pages are available in demo mode.",
        icon="🗄️",
    )


@st.cache_resource(ttl=60)
def load_model():
    """Load champion model. TTL=60s ensures new model is picked up after retraining."""
    try:
        cfg = get_config()
        db_path = str(resolve_path(cfg["paths"]["models_dir"])) + "/champion_model.pkl"
        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


# ── Safe Query Helper ───────────────────────────────────────────────────────

def _safe_query(query: str, fallback=None):
    """Execute a DuckDB query safely; return fallback on any error."""
    con = get_hub_con()
    if con is None:
        return fallback if fallback is not None else pd.DataFrame()
    try:
        return con.execute(query).df()
    except Exception:
        return fallback if fallback is not None else pd.DataFrame()


def _safe_scalar(query: str, default=0.5):
    """Execute a scalar query and return its first cell safely."""
    con = get_hub_con()
    if con is None:
        return default
    try:
        row = con.execute(query).df()
        if row.empty:
            return default
        val = row.iloc[0, 0]
        return float(val) if val is not None else default
    except Exception:
        return default


# ── Data Loaders ────────────────────────────────────────────────────────────

@st.cache_data
def get_teams():
    df = _safe_query("SELECT DISTINCT team_1 FROM main_gold.fact_matches ORDER BY team_1")
    if df.empty or "team_1" not in df.columns:
        return ["India", "Australia", "England", "Pakistan", "New Zealand",
                "South Africa", "West Indies", "Sri Lanka"]
    return df["team_1"].tolist()


@st.cache_data
def get_venues():
    df = _safe_query("SELECT DISTINCT venue FROM main_gold.fact_matches ORDER BY venue")
    if df.empty or "venue" not in df.columns:
        return ["Wankhede Stadium", "MCG", "Lord's", "Eden Gardens",
                "Headingley", "The Oval", "SuperSport Park"]
    return df["venue"].tolist()


@st.cache_data
def get_global_kpis():
    q = """
    SELECT
        (SELECT COUNT(*) FROM main_gold.fact_matches) as total_matches,
        (SELECT SUM(runs_batter + runs_extras) FROM main_gold.fact_deliveries) as total_runs,
        (SELECT COUNT(*) FROM main_gold.fact_wickets) as total_wickets,
        (SELECT COUNT(DISTINCT batter) FROM main_gold.fact_deliveries) as total_players
    """
    con = get_hub_con()
    if con is None:
        return pd.Series({"total_matches": 0, "total_runs": 0, "total_wickets": 0, "total_players": 0})
    try:
        return con.execute(q).df().iloc[0]
    except Exception:
        return pd.Series({"total_matches": 0, "total_runs": 0, "total_wickets": 0, "total_players": 0})


@st.cache_data
def get_h2h_rate(team1, team2=None):
    """Phase 22 FIX: Pairwise H2H win rate of team1 vs team2."""
    if not db_available():
        return 0.5
    try:
        con = get_hub_con()
        if team2:
            q = f"""
            WITH matchup AS (
                SELECT m.match_id, m.winner
                FROM main_gold.fact_matches m
                JOIN main_gold.fact_innings i1 ON m.match_id = i1.match_id AND i1.batting_team = '{team1}'
                JOIN main_gold.fact_innings i2 ON m.match_id = i2.match_id AND i2.batting_team = '{team2}'
                WHERE m.result_type NOT IN ('no result', 'tie')
            )
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN winner = '{team1}' THEN 1 ELSE 0 END) as wins
            FROM matchup
            """
        else:
            q = f"""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN winner = '{team1}' THEN 1 ELSE 0 END) as wins
            FROM main_gold.fact_matches
            WHERE result_type NOT IN ('no result', 'tie')
              AND (toss_winner = '{team1}' OR winner = '{team1}')
            """
        row = con.execute(q).df().iloc[0]
        total = max(float(row["total"]), 1)
        return float(row["wins"]) / total
    except Exception:
        return 0.5


@st.cache_data
def get_venue_avg(venue):
    if not db_available():
        return 150.0
    try:
        con = get_hub_con()
        q = f"""
        SELECT AVG(i.total_runs) as avg_runs
        FROM main_gold.fact_innings i
        JOIN main_gold.fact_matches m ON i.match_id = m.match_id
        WHERE m.venue = '{venue}' AND i.innings_number = 1
        """
        row = con.execute(q).df().iloc[0]
        return float(row["avg_runs"] or 150)
    except Exception:
        return 150.0


@st.cache_data
def get_venue_chase_rate(venue):
    """Phase 22: Chase success rate at venue."""
    if not db_available():
        return 0.5
    try:
        con = get_hub_con()
        q = f"""
        SELECT COUNT(*) as total,
               SUM(CASE
                   WHEN (toss_decision = 'bat' AND team_1_win = 0) OR (toss_decision = 'field' AND team_1_win = 1)
                   THEN 1 ELSE 0 END) as chase_wins
        FROM main_gold.fact_matches
        WHERE venue = '{venue}' AND result_type NOT IN ('no result', 'tie')
        """
        row = con.execute(q).df().iloc[0]
        total = max(float(row["total"]), 1)
        return float(row["chase_wins"]) / total
    except Exception:
        return 0.5


@st.cache_data
def get_team_form(team, window=5):
    """Phase 22 FIX: Independent team form."""
    if not db_available():
        return 0.5
    try:
        con = get_hub_con()
        q = f"""
        WITH team_matches AS (
            SELECT m.match_id, m.match_date, m.winner
            FROM main_gold.fact_matches m
            JOIN main_gold.fact_innings i ON m.match_id = i.match_id AND i.batting_team = '{team}'
            WHERE m.result_type NOT IN ('no result', 'tie')
            GROUP BY m.match_id, m.match_date, m.winner
            ORDER BY m.match_date DESC
            LIMIT {window}
        )
        SELECT ROUND(AVG(CASE WHEN winner = '{team}' THEN 1.0 ELSE 0.0 END)::FLOAT, 3) as form
        FROM team_matches
        """
        row = con.execute(q).df().iloc[0]
        return float(row["form"] if row["form"] is not None else 0.5)
    except Exception:
        return 0.5


@st.cache_data
def get_team_momentum(team):
    """Phase 22: Weighted momentum index (recent 5 matches)."""
    if not db_available():
        return 0.5
    try:
        con = get_hub_con()
        q = f"""
        WITH team_matches AS (
            SELECT m.match_id, m.match_date,
                   CASE WHEN m.winner = '{team}' THEN 1.0 ELSE 0.0 END as won
            FROM main_gold.fact_matches m
            JOIN main_gold.fact_innings i ON m.match_id = i.match_id AND i.batting_team = '{team}'
            WHERE m.result_type NOT IN ('no result', 'tie')
            GROUP BY m.match_id, m.match_date, m.winner
            ORDER BY m.match_date DESC
            LIMIT 5
        )
        SELECT won FROM team_matches ORDER BY match_date ASC
        """
        df = con.execute(q).df()
        if df.empty:
            return 0.5
        wins = df["won"].values
        weights = list(range(1, len(wins) + 1))
        return float(sum(w * v for w, v in zip(weights, wins)) / sum(weights))
    except Exception:
        return 0.5


@st.cache_data
def get_team_venue_win_rate(team, venue):
    """Phase 22: Team's win rate at specific venue."""
    if not db_available():
        return 0.5
    try:
        con = get_hub_con()
        q = f"""
        WITH team_venue AS (
            SELECT m.match_id, m.winner
            FROM main_gold.fact_matches m
            JOIN main_gold.fact_innings i ON m.match_id = i.match_id AND i.batting_team = '{team}'
            WHERE m.venue = '{venue}' AND m.result_type NOT IN ('no result', 'tie')
            GROUP BY m.match_id, m.winner
        )
        SELECT COUNT(*) as total,
               SUM(CASE WHEN winner = '{team}' THEN 1 ELSE 0 END) as wins
        FROM team_venue
        """
        row = con.execute(q).df().iloc[0]
        total = max(float(row["total"]), 1)
        return float(row["wins"]) / total
    except Exception:
        return 0.5


@st.cache_data
def get_phase_data(venue_sel):
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
    return _safe_query(q)


@st.cache_data
def get_toss_recommendation(venue_sel):
    q = f"""
    SELECT toss_decision,
           COUNT(*) total,
           SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) as toss_wins
    FROM main_gold.fact_matches
    WHERE venue = '{venue_sel}' AND result_type NOT IN ('no result', 'tie')
    GROUP BY toss_decision
    """
    return _safe_query(q)


@st.cache_data
def get_top_batters():
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
    return _safe_query(q)


@st.cache_data
def get_top_bowlers():
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
    return _safe_query(q)


@st.cache_data
def get_venue_heatmap():
    q = """
    SELECT t.team as team, m.venue as venue,
           ROUND(AVG(CASE WHEN m.winner = t.team THEN 1.0 ELSE 0.0 END) * 100, 0) as win_pct,
           COUNT(m.match_id) as matches
    FROM main_gold.fact_matches m
    JOIN main_silver.slv_match_teams t ON m.match_id = t.match_id
    GROUP BY t.team, m.venue HAVING matches >= 5
    ORDER BY win_pct DESC
    """
    return _safe_query(q)


@st.cache_data
def get_exciting_matches():
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
    return _safe_query(q)


@st.cache_data
def get_highest_scores():
    q = """
    SELECT batter, SUM(runs_batter) as innings_runs, m.event_name, m.match_date, team_1, winner
    FROM main_gold.fact_deliveries d
    JOIN main_gold.fact_matches m ON d.match_id = m.match_id
    GROUP BY d.batter, d.match_id, d.innings_number, m.event_name, m.match_date, m.team_1, m.winner
    ORDER BY innings_runs DESC LIMIT 10
    """
    return _safe_query(q)


@st.cache_data
def get_best_bowling():
    q = """
    SELECT bowler, SUM(is_wicket) as wickets, SUM(runs_batter + runs_extras) as runs_conceded,
           m.event_name, m.match_date
    FROM main_gold.fact_deliveries d
    JOIN main_gold.fact_matches m ON d.match_id = m.match_id
    GROUP BY d.bowler, d.match_id, d.innings_number, m.event_name, m.match_date
    HAVING wickets >= 3
    ORDER BY wickets DESC, runs_conceded ASC LIMIT 10
    """
    return _safe_query(q)


# ── Shared CSS ─────────────────────────────────────────────────────────────

THEME_CSS = """
<style>
    [data-testid="stSidebar"] { background-color: #0b0f19; }
    .stApp { background-color: #0f172a; color: #f8fafc; font-size: 1.1rem; }
    .metric-card { background: #1e293b; border-radius: 12px; padding: 20px; margin: 8px 0; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); font-size: 1.1rem; }
    .section-header { font-size: 1.75rem; font-weight: 700; color: #38bdf8; margin-bottom: 16px; border-bottom: 2px solid #1e293b; padding-bottom: 8px; }
    .chat-msg-user { background: #0284c7; border-radius: 12px 12px 0 12px; padding: 12px 16px; margin: 4px 0; color: white; float: right; clear: both; max-width: 80%; font-size: 1.05rem; }
    .chat-msg-bot { background: #1e293b; border-radius: 12px 12px 12px 0; padding: 12px 16px; margin: 4px 0; border: 1px solid #334155; color: #f8fafc; float: left; clear: both; max-width: 80%; font-size: 1.05rem; }
    .stTextInput > div > div > input { background: #1e293b; color: white; border: 1px solid #334155; border-radius: 8px; }
    .st-expander { background-color: #1e293b !important; border: 1px solid #334155 !important; border-radius: 8px !important; }
    .stCodeBlock { background-color: #0b0f19 !important; border-radius: 6px !important; border: 1px solid #334155 !important; }
    .kpi-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 24px; border: 1px solid #334155; text-align: center; }
    .kpi-value { font-size: 2.25rem; font-weight: 800; color: #38bdf8; }
    .kpi-label { font-size: 1rem; color: #94a3b8; margin-top: 4px; }
    .prediction-card { background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); border-radius: 16px; padding: 24px; border: 1px solid #1d4ed8; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }

    /* ── Sidebar radio: hide dots, style as clean nav items ── */
    [data-testid="stSidebar"] .stRadio > label { display: none; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 2px; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        font-size: 1rem;
        font-weight: 500;
        color: #94a3b8;
        padding: 10px 14px;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.15s, color 0.15s;
        margin: 0;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        margin: 0;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: #1e293b;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
        background: #1e293b;
        color: #38bdf8;
        font-weight: 600;
        border-left: 3px solid #38bdf8;
    }
    /* Hide the radio dot circle */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    /* Remove default Streamlit radio dot/indicator */
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] { pointer-events: none; }
    [data-testid="stSidebar"] .stRadio input[type="radio"] { display: none !important; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div[data-testid="stRadioOptionIndicator"],
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span[data-testid="stRadioOptionIndicator"] {
        display: none !important;
    }
</style>
"""
