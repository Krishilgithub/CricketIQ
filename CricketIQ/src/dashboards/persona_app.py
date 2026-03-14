import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pickle
import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.config import get_config, resolve_path

st.set_page_config(
    page_title="CricketIQ — Intelligence Hub",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0d1117; }
    .metric-card { background: #161b22; border-radius: 12px; padding: 16px; margin: 8px 0; border: 1px solid #30363d; }
    .section-header { font-size: 1.4rem; font-weight: 700; color: #58a6ff; margin-bottom: 12px; }
    .win-prob-bar { height: 28px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Data Connections ─────────────────────────────────────────────────────────
@st.cache_resource
def get_con():
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

con = get_con()
champion = load_model()

# ── Teams List ───────────────────────────────────────────────────────────────
@st.cache_data
def get_teams():
    return con.execute("SELECT DISTINCT team_1 FROM main_gold.fact_matches ORDER BY team_1").df()["team_1"].tolist()

@st.cache_data
def get_venues():
    return con.execute("SELECT DISTINCT venue FROM main_gold.fact_matches ORDER BY venue").df()["venue"].tolist()

teams = get_teams()
venues = get_venues()

# ── Sidebar Navigation ───────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/5/5c/ICC_Cricket_World_Cup_Logo.svg", width=80)
st.sidebar.markdown("## 🏏 CricketIQ")
st.sidebar.markdown("*Intelligence Hub*")
persona = st.sidebar.radio(
    "Select View",
    ["🎯 Team Analyst", "👨‍💼 Coach / Captain", "📊 Management", "📺 Fan / Media"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Data**: 1.14M deliveries | 3,211 T20I matches")
st.sidebar.markdown("**Model**: Champion Logistic Regression")
st.sidebar.markdown("**MLflow**: [localhost:5050](http://localhost:5050)")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TEAM ANALYST: Pre-match Win Probability
# ═══════════════════════════════════════════════════════════════════════════════
if persona == "🎯 Team Analyst":
    st.title("🎯 Team Analyst — Pre-Match Intelligence")
    st.markdown("Compute win probability and venue-adjusted expected score using the CricketIQ model.")

    col_t1, col_t2, col_v = st.columns(3)
    with col_t1:
        team1 = st.selectbox("Team 1 (Toss Winner)", teams, index=teams.index("India") if "India" in teams else 0)
    with col_t2:
        others = [t for t in teams if t != team1]
        team2 = st.selectbox("Team 2", others, index=0)
    with col_v:
        venue = st.selectbox("Match Venue", venues)

    toss_decision = st.radio("Toss Decision", ["Bat", "Field"], horizontal=True)

    # Pull live features from DB
    h2h_q = f"""
    SELECT COUNT(*) as total,
           SUM(CASE WHEN winner = '{team1}' THEN 1 ELSE 0 END) as wins
    FROM main_gold.fact_matches
    WHERE (toss_winner = '{team1}' OR winner = '{team1}')
    """
    h2h_df = con.execute(h2h_q).df()
    h2h_rate = float(h2h_df.iloc[0]["wins"]) / max(float(h2h_df.iloc[0]["total"]), 1)

    venue_q = f"""
    SELECT AVG(i.total_runs) as avg_runs
    FROM main_gold.fact_innings i
    JOIN main_gold.fact_matches m ON i.match_id = m.match_id
    WHERE m.venue = '{venue}' AND i.innings_number = 1
    """
    venue_df = con.execute(venue_q).df()
    venue_avg = float(venue_df.iloc[0]["avg_runs"] or 150)

    form_q = f"""
    SELECT ROUND(AVG(team_1_win)::FLOAT, 3) as form
    FROM (
        SELECT team_1_win FROM main_gold.fact_matches
        WHERE toss_winner = '{team1}'
        ORDER BY match_date DESC LIMIT 5
    )
    """
    form_df = con.execute(form_q).df()
    team1_form = float(form_df.iloc[0]["form"] or 0.5)

    # Run model prediction
    win_prob = None
    if champion:
        import numpy as np
        feats = pd.DataFrame([{
            "toss_bat": 1 if toss_decision == "Bat" else 0,
            "venue_avg_1st_inns_runs": venue_avg,
            "team_1_h2h_win_rate": h2h_rate,
            "team_1_form_last5": team1_form,
            "team_2_form_last5": 1 - team1_form,
        }])
        win_prob = float(champion["model"].predict_proba(feats)[0][1])
    
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win Probability", f"{win_prob*100:.1f}%" if win_prob else "N/A", delta=f"{(win_prob-0.5)*100:+.1f}% vs 50/50" if win_prob else None)
    c2.metric("Venue Avg 1st Innings", f"{venue_avg:.0f} runs")
    c3.metric("H2H Win Rate", f"{h2h_rate*100:.1f}%")
    c4.metric("Last 5-Match Form", f"{team1_form*100:.1f}%")

    if win_prob is not None:
        fig = go.Figure(go.Bar(
            x=[win_prob * 100, (1 - win_prob) * 100],
            y=[team1, team2],
            orientation="h",
            marker_color=["#2ea043", "#da3633"],
        ))
        fig.update_layout(title="Win Probability", xaxis_title="Win % →", template="plotly_dark", height=200)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COACH/CAPTAIN: Phase Performance KPIs
# ═══════════════════════════════════════════════════════════════════════════════
elif persona == "👨‍💼 Coach / Captain":
    st.title("👨‍💼 Coach / Captain — Match Phase Intelligence")
    st.markdown("Powerplay index, middle-overs control, death-overs risk, and toss recommendation.")
    
    team_sel = st.selectbox("Team", teams, index=teams.index("India") if "India" in teams else 0)
    venue_sel = st.selectbox("Venue", venues)

    phase_q = f"""
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
    phase_df = con.execute(phase_q).df()

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if not phase_df.empty:
            fig2 = px.bar(phase_df, x="phase", y="run_rate", title=f"Run Rate by Phase at {venue_sel[:25]}",
                          color="run_rate", color_continuous_scale="Blues", template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
    with col_p2:
        if not phase_df.empty:
            fig3 = px.bar(phase_df, x="phase", y="wicket_rate", title="Wicket Fall Rate by Phase",
                          color="wicket_rate", color_continuous_scale="Reds", template="plotly_dark")
            st.plotly_chart(fig3, use_container_width=True)

    # Toss Recommendation at Venue
    toss_q = f"""
    SELECT toss_decision,
           COUNT(*) total,
           SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) as toss_wins
    FROM main_gold.fact_matches
    WHERE venue = '{venue_sel}' AND result_type NOT IN ('no result', 'tie')
    GROUP BY toss_decision
    """
    toss_df = con.execute(toss_q).df()
    if not toss_df.empty:
        toss_df["win_pct"] = (toss_df["toss_wins"] / toss_df["total"] * 100).round(1)
        best = toss_df.loc[toss_df["win_pct"].idxmax()]
        st.success(f"🪙 **Toss Recommendation at {venue_sel[:30]}**: Choose to **{best['toss_decision'].upper()}** — {best['win_pct']:.0f}% win conversion when toss won and chose to {best['toss_decision']}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MANAGEMENT: Player Form & Heatmaps
# ═══════════════════════════════════════════════════════════════════════════════
elif persona == "📊 Management":
    st.title("📊 Management — Talent & Form Analytics")
    st.markdown("Player consistency, form momentum, and opposition weakness heatmaps.")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("🏏 Top Run Scorers (Last 2 Years)")
        bat_q = """
        SELECT batter, SUM(runs_batter) as runs,
               SUM(is_legal_ball) as balls,
               ROUND(SUM(runs_batter)::FLOAT / NULLIF(SUM(is_legal_ball),0) * 100, 2) as sr
        FROM main_gold.fact_deliveries d
        JOIN main_gold.fact_matches m ON d.match_id = m.match_id
        WHERE m.match_date >= (CURRENT_DATE - INTERVAL 730 DAY)
        GROUP BY batter HAVING balls > 100
        ORDER BY runs DESC LIMIT 15
        """
        bat_df = con.execute(bat_q).df()
        fig4 = px.bar(bat_df, x="batter", y="runs", color="sr", 
                      title="Runs (colored by Strike Rate)",
                      color_continuous_scale="Viridis", template="plotly_dark")
        fig4.update_xaxes(tickangle=45)
        st.plotly_chart(fig4, use_container_width=True)

    with col_m2:
        st.subheader("🎯 Top Wicket Takers (Last 2 Years)")
        bowl_q = """
        SELECT bowler, SUM(is_wicket) as wickets,
               SUM(is_legal_ball) as balls,
               ROUND(SUM(runs_batter + runs_extras)::FLOAT / NULLIF(SUM(is_legal_ball),0) * 6, 2) as econ
        FROM main_gold.fact_deliveries d
        JOIN main_gold.fact_matches m ON d.match_id = m.match_id
        WHERE m.match_date >= (CURRENT_DATE - INTERVAL 730 DAY)
        GROUP BY bowler HAVING balls > 100
        ORDER BY wickets DESC LIMIT 15
        """
        bowl_df = con.execute(bowl_q).df()
        fig5 = px.bar(bowl_df, x="bowler", y="wickets", color="econ",
                      title="Wickets (colored by Economy)",
                      color_continuous_scale="RdYlGn_r", template="plotly_dark")
        fig5.update_xaxes(tickangle=45)
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("📍 Team vs Venue Win Heatmap")
    heatmap_q = """
    SELECT team_1 as team, venue,
           ROUND(AVG(team_1_win::FLOAT) * 100, 0) as win_pct,
           COUNT(*) as matches
    FROM main_gold.fact_matches
    JOIN main_silver.slv_match_teams t ON main_gold.fact_matches.match_id = t.match_id
    GROUP BY team, venue HAVING matches >= 5
    ORDER BY win_pct DESC
    """
    hm_df = con.execute(heatmap_q).df()
    if not hm_df.empty:
        top_teams = hm_df.groupby("team")["matches"].sum().nlargest(12).index.tolist()
        top_venues = hm_df.groupby("venue")["matches"].sum().nlargest(15).index.tolist()
        hm_pivot = hm_df[hm_df["team"].isin(top_teams) & hm_df["venue"].isin(top_venues)]\
            .pivot_table(index="team", columns="venue", values="win_pct")
        if not hm_pivot.empty:
            fig6 = px.imshow(hm_pivot, title="Win % by Team × Venue (min 5 matches)",
                             color_continuous_scale="RdYlGn", template="plotly_dark", aspect="auto")
            st.plotly_chart(fig6, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FAN / MEDIA: Excitement Index & Key Players
# ═══════════════════════════════════════════════════════════════════════════════
elif persona == "📺 Fan / Media":
    st.title("📺 Fan & Media — Entertainment Analytics")
    st.markdown("Match excitement scores, key player trackers, and historic record-chasing moments.")

    st.subheader("🔥 Most Exciting T20I Matches (Super Overs / Ties / Close Chases)")
    excitement_q = """
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
    ex_df = con.execute(excitement_q).df()
    if not ex_df.empty:
        ex_df["excitement_score"] = (200 - ex_df["margin_runs"]).clip(0, 200)
        ex_df["label"] = ex_df["event_name"].fillna("T20I") + " " + ex_df["match_date"].astype(str).str[:4]
        fig7 = px.bar(ex_df, x="label", y="excitement_score",
                      color="excitement_score", color_continuous_scale="Inferno",
                      title="Top 20 Most Exciting T20I Matches (lower margin = more exciting)",
                      template="plotly_dark")
        fig7.update_xaxes(tickangle=45)
        st.plotly_chart(fig7, use_container_width=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.subheader("💥 Highest Individual Scores in T20Is")
        top_bat_q = """
        SELECT batter, SUM(runs_batter) as innings_runs, m.event_name, m.match_date, team_1, winner
        FROM main_gold.fact_deliveries d
        JOIN main_gold.fact_matches m ON d.match_id = m.match_id
        GROUP BY d.batter, d.match_id, d.innings_number, m.event_name, m.match_date, m.team_1, m.winner
        ORDER BY innings_runs DESC LIMIT 10
        """
        top_bat_df = con.execute(top_bat_q).df()
        st.dataframe(top_bat_df[["batter", "innings_runs", "event_name", "match_date"]].rename(columns={
            "batter": "Batter", "innings_runs": "Runs", "event_name": "Tournament", "match_date": "Date"
        }), use_container_width=True)

    with col_f2:
        st.subheader("🎯 Best Bowling Spells in T20Is")
        top_bowl_q = """
        SELECT bowler, SUM(is_wicket) as wickets, SUM(runs_batter + runs_extras) as runs_conceded,
               m.event_name, m.match_date
        FROM main_gold.fact_deliveries d
        JOIN main_gold.fact_matches m ON d.match_id = m.match_id
        GROUP BY d.bowler, d.match_id, d.innings_number, m.event_name, m.match_date
        HAVING wickets >= 3
        ORDER BY wickets DESC, runs_conceded ASC LIMIT 10
        """
        top_bowl_df = con.execute(top_bowl_q).df()
        st.dataframe(top_bowl_df[["bowler", "wickets", "runs_conceded", "event_name", "match_date"]].rename(columns={
            "bowler": "Bowler", "wickets": "Wickets", "runs_conceded": "Runs", "event_name": "Tournament", "match_date": "Date"
        }), use_container_width=True)
