import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
from pathlib import Path
from src.config import get_config, resolve_path
from src.warehouse.data_quality import build_and_run_dq_suite

# Set page config
st.set_page_config(page_title="CricketIQ - EDA Dashboard", page_icon="🏏", layout="wide")

st.title("🏏 CricketIQ - Explosive Data Analytics")
st.markdown("Explore Team Trends, Venue Behavior, Matchups, and Live Data Quality.")

@st.cache_resource
def get_db_connection():
    cfg = get_config()
    db_path = str(resolve_path(cfg["paths"]["duckdb_path"]))
    return duckdb.connect(db_path, read_only=True)

con = get_db_connection()

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Team & Venue Trends", 
    "🪙 Toss & Venue Behavior", 
    "⚔️ Player Matchups", 
    "🛡️ Data Quality Status"
])

# ─── TAB 1: Team Performance Trends ────────────────────────
with tab1:
    st.header("Team Performance Trends")
    # Fetch all canonical teams
    teams_df = con.execute("SELECT DISTINCT team_1 AS team FROM main_gold.fact_matches UNION SELECT DISTINCT toss_winner FROM main_gold.fact_matches ORDER BY team").df()
    selected_team = st.selectbox("Select Team", teams_df["team"].tolist(), index=teams_df["team"].tolist().index("India") if "India" in teams_df["team"].tolist() else 0)

    st.subheader(f"Win Rate by Venue for {selected_team}")
    
    query1 = f"""
    SELECT 
        venue, 
        COUNT(*) as total_matches,
        SUM(CASE WHEN winner = '{selected_team}' THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN winner = '{selected_team}' THEN 1.0 ELSE 0.0 END) / COUNT(*) as win_rate
    FROM main_gold.fact_matches
    WHERE '{selected_team}' IN (team_1, CASE WHEN team_1 = toss_winner THEN toss_winner /* fallback logic */ END) 
       OR toss_winner = '{selected_team}' OR winner = '{selected_team}'
       -- A better filter: where the selected team played
       -- Since we know winner and toss_winner, we can join slv_match_teams but it's easier to just query fact_matches + slv_match_teams
    """
    
    # Better joining teams
    query_teams_venues = f"""
    SELECT 
        m.venue,
        COUNT(m.match_id) as total_matches,
        SUM(CASE WHEN m.winner = '{selected_team}' THEN 1 ELSE 0 END) as wins,
        ROUND((SUM(CASE WHEN m.winner = '{selected_team}' THEN 1.0 ELSE 0.0 END) / COUNT(m.match_id)) * 100, 2) as win_rate_pct
    FROM main_gold.fact_matches m
    JOIN main_silver.slv_match_teams t ON m.match_id = t.match_id
    WHERE t.team = '{selected_team}'
    GROUP BY m.venue
    HAVING COUNT(m.match_id) >= 3
    ORDER BY win_rate_pct DESC
    LIMIT 15
    """
    
    venue_perf_df = con.execute(query_teams_venues).df()
    
    if not venue_perf_df.empty:
        fig1 = px.bar(venue_perf_df, x="venue", y="win_rate_pct", hover_data=["total_matches", "wins"],
                      title=f"Top Venues for {selected_team} (Min 3 Matches)", labels={"win_rate_pct": "Win Rate (%)"},
                      color="win_rate_pct", color_continuous_scale="Viridis")
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Not enough data for the selected team across distinct venues.")


# ─── TAB 2: Toss & Venue Behavior ──────────────────────────
with tab2:
    st.header("Toss Impact & Venue Behaviors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Does winning the toss matter?")
        toss_query = """
        SELECT 
            toss_decision,
            SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) as toss_and_match_wins,
            COUNT(*) as total_toss_wins,
            ROUND(SUM(CASE WHEN toss_winner = winner THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) as win_conversion_pct
        FROM main_gold.fact_matches
        WHERE toss_decision IN ('bat', 'field')
        GROUP BY toss_decision
        """
        toss_df = con.execute(toss_query).df()
        
        fig2 = px.pie(toss_df, values="total_toss_wins", names="toss_decision", title="Overall Toss Decisions")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.dataframe(toss_df, use_container_width=True)

    with col2:
        st.subheader("Chase Success vs Defend Success by Venue")
        # Define chase success as winning when batting second
        # If toss_decision=bat and toss_winner=winner -> Defend Win.
        # This is a proxy. Better proxy is looking at innings.
        venue_query = """
        WITH InningsStats AS (
            SELECT 
                match_id,
                MAX(CASE WHEN innings_number = 1 THEN total_runs ELSE 0 END) as first_innings_runs
            FROM main_gold.fact_innings
            GROUP BY match_id
        )
        SELECT 
            m.venue,
            COUNT(m.match_id) as total_matches,
            ROUND(AVG(i.first_innings_runs), 0) as avg_1st_inns_score,
            -- Roughly, if method isn't null or ties occur, ignore. 
            SUM(CASE WHEN m.toss_decision = 'field' AND m.toss_winner = m.winner THEN 1 
                     WHEN m.toss_decision = 'bat' AND m.toss_winner != m.winner THEN 1 ELSE 0 END) as chasing_wins
        FROM main_gold.fact_matches m
        JOIN InningsStats i ON m.match_id = i.match_id
        GROUP BY m.venue
        HAVING total_matches >= 10
        ORDER BY total_matches DESC
        LIMIT 20
        """
        v_df = con.execute(venue_query).df()
        v_df['chase_win_pct'] = round((v_df['chasing_wins'] / v_df['total_matches']) * 100, 1)
        
        fig3 = px.scatter(v_df, x="avg_1st_inns_score", y="chase_win_pct", size="total_matches", color="chase_win_pct",
                          hover_name="venue", title="Avg 1st Innings Score vs Chase Win % (Min 10 Matches)",
                          color_continuous_scale="RdBu")
        st.plotly_chart(fig3, use_container_width=True)


# ─── TAB 3: Player Matchups ──────────────────────────────
with tab3:
    st.header("Batting vs Bowling Matchup Patterns")
    st.markdown("Select a Batter and a Bowler to see their historical head-to-head stats.")
    
    col3, col4 = st.columns(2)
    with col3:
        batters_df = con.execute("SELECT DISTINCT batter FROM main_gold.fact_deliveries ORDER BY batter").df()
        batter_sel = st.selectbox("Batter", batters_df['batter'].tolist(), index=batters_df['batter'].tolist().index("V Kohli") if "V Kohli" in batters_df['batter'].tolist() else 0)
    with col4:
        bowlers_df = con.execute(f"SELECT DISTINCT bowler FROM main_gold.fact_deliveries WHERE batter = '{batter_sel}' ORDER BY bowler").df()
        if not bowlers_df.empty:
            bowler_sel = st.selectbox("Bowler", bowlers_df['bowler'].tolist())
        else:
            bowler_sel = None
            st.warning("No bowlers found for this batter.")

    if bowler_sel:
        matchup_query = f"""
        SELECT 
            SUM(runs_batter) as total_runs,
            SUM(CASE WHEN is_legal_ball = 1 THEN 1 ELSE 0 END) as balls_faced,
            SUM(is_wicket) as dismissals
        FROM main_gold.fact_deliveries
        WHERE batter = '{batter_sel}' AND bowler = '{bowler_sel}'
        """
        matchup_df = con.execute(matchup_query).df()
        
        m_runs = int(matchup_df.iloc[0]['total_runs'])
        m_balls = int(matchup_df.iloc[0]['balls_faced'])
        m_outs = int(matchup_df.iloc[0]['dismissals'])
        m_sr = round((m_runs / m_balls) * 100, 2) if m_balls > 0 else 0
        
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Runs", m_runs)
        mc2.metric("Balls Faced", m_balls)
        mc3.metric("Dismissals", m_outs)
        mc4.metric("Strike Rate", m_sr)


# ─── TAB 4: Data Quality Status ───────────────────────────
with tab4:
    st.header("Data Quality Status Panel")
    st.markdown("Run the production Gold layer Data Quality assertions.")
    
    if st.button("Run DQ Suite Now", type="primary"):
        with st.spinner("Executing Data Quality tests..."):
            cfg = get_config()
            db_path = str(resolve_path(cfg["paths"]["duckdb_path"]))
            passed = build_and_run_dq_suite(db_path)
            
            if passed:
                st.success("✅ All Data Quality constraints passed! Data is structurally sound.")
                st.balloons()
            else:
                st.error("❌ Data Quality validations failed. See logs for details.")

    # Show some DB stats
    st.markdown("### Database Statistics")
    st_df_m = con.execute("SELECT count(*) as count FROM main_gold.fact_matches").df()
    st_df_d = con.execute("SELECT count(*) as count FROM main_gold.fact_deliveries").df()
    sc1, sc2 = st.columns(2)
    sc1.metric("Total T20 Matches Processed", f"{st_df_m.iloc[0]['count']:,}")
    sc2.metric("Total Deliveries Processed", f"{st_df_d.iloc[0]['count']:,}")

