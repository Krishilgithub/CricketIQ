"""
Coach / Captain Dashboard — Match strategy, team performance, toss analysis.
"""

import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_conn():
    from src.warehouse.schema import get_connection
    return get_connection()


def render_coach_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>🏏 Coach / Captain Dashboard</h1>
        <p>Match strategy, team performance, toss impact, phase-wise analysis</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        conn = get_conn()
    except Exception as e:
        st.error(f"Cannot connect to warehouse: {e}")
        return

    # Team selector
    teams = conn.execute("""
        SELECT team_name FROM gold.dim_team
        WHERE is_icc_full_member = TRUE ORDER BY team_name
    """).fetchall()
    team_list = [t[0] for t in teams]

    with st.sidebar:
        st.markdown("### 🎛️ Coach Filters")
        selected_team = st.selectbox("Select Team", team_list, key="coach_team")
        source_filter = st.multiselect(
            "Data Source", ["t20i", "ipl", "bbl", "cpl"],
            default=["t20i"], key="coach_src"
        )

    if not source_filter:
        st.warning("Select at least one source.")
        conn.close()
        return

    src_clause = ", ".join(f"'{s}'" for s in source_filter)

    # ── KPI Cards ──
    st.markdown("### 📊 Key Performance Indicators")

    try:
        team_stats = conn.execute(f"""
            SELECT
                COUNT(*) AS total_matches,
                SUM(CASE WHEN winner = '{selected_team}' THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN toss_winner = '{selected_team}' AND toss_decision = 'bat'
                    AND winner = '{selected_team}' THEN 1 ELSE 0 END) AS bat_first_wins,
                SUM(CASE WHEN toss_winner = '{selected_team}' AND toss_decision = 'bat' THEN 1 ELSE 0 END) AS bat_first_total,
                SUM(CASE WHEN (toss_winner != '{selected_team}' AND toss_decision = 'bat')
                    OR (toss_winner = '{selected_team}' AND toss_decision = 'field')
                    THEN CASE WHEN winner = '{selected_team}' THEN 1 ELSE 0 END ELSE 0 END) AS chase_wins,
                SUM(CASE WHEN (toss_winner != '{selected_team}' AND toss_decision = 'bat')
                    OR (toss_winner = '{selected_team}' AND toss_decision = 'field') THEN 1 ELSE 0 END) AS chase_total,
                SUM(CASE WHEN toss_winner = '{selected_team}' AND winner = '{selected_team}' THEN 1 ELSE 0 END) AS toss_match_wins,
                SUM(CASE WHEN toss_winner = '{selected_team}' THEN 1 ELSE 0 END) AS toss_wins
            FROM silver.matches
            WHERE (team1 = '{selected_team}' OR team2 = '{selected_team}')
              AND source IN ({src_clause})
              AND winner IS NOT NULL
        """).fetchone()

        total, wins = team_stats[0], team_stats[1]
        win_rate = (wins / total * 100) if total > 0 else 0
        bat_first_wr = (team_stats[2] / team_stats[3] * 100) if team_stats[3] > 0 else 0
        chase_wr = (team_stats[4] / team_stats[5] * 100) if team_stats[5] > 0 else 0
        toss_impact = (team_stats[6] / team_stats[7] * 100) if team_stats[7] > 0 else 0

        cols = st.columns(6)
        with cols[0]:
            st.metric("Matches", total)
        with cols[1]:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with cols[2]:
            st.metric("Bat First Win%", f"{bat_first_wr:.1f}%")
        with cols[3]:
            st.metric("Chase Win%", f"{chase_wr:.1f}%")
        with cols[4]:
            st.metric("Toss→Win%", f"{toss_impact:.1f}%")
        with cols[5]:
            st.metric("Wins", wins)

    except Exception as e:
        st.error(f"Error loading KPIs: {e}")

    # ── Phase-wise Performance ──
    st.markdown("### ⚡ Phase-wise Performance")
    try:
        phase_stats = conn.execute(f"""
            SELECT
                ROUND(AVG(pp_runs), 1) AS avg_pp_runs,
                ROUND(AVG(pp_wickets), 1) AS avg_pp_wkts,
                ROUND(AVG(pp_run_rate), 2) AS avg_pp_rr,
                ROUND(AVG(middle_runs), 1) AS avg_mid_runs,
                ROUND(AVG(middle_wickets), 1) AS avg_mid_wkts,
                ROUND(AVG(middle_run_rate), 2) AS avg_mid_rr,
                ROUND(AVG(death_runs), 1) AS avg_death_runs,
                ROUND(AVG(death_wickets), 1) AS avg_death_wkts,
                ROUND(AVG(death_run_rate), 2) AS avg_death_rr,
                ROUND(AVG(boundary_runs * 100.0 / NULLIF(total_runs, 0)), 1) AS boundary_pct,
                ROUND(AVG(dot_balls * 100.0 / NULLIF(total_runs + dot_balls, 0)), 1) AS dot_pct
            FROM gold.fact_innings_summary fi
            JOIN gold.dim_team t ON fi.team_key = t.team_key
            WHERE t.team_name = '{selected_team}'
              AND fi.source IN ({src_clause})
        """).fetchone()

        col1, col2 = st.columns(2)
        with col1:
            # Phase-wise runs
            fig = go.Figure(go.Bar(
                x=["Powerplay", "Middle", "Death"],
                y=[phase_stats[0], phase_stats[3], phase_stats[6]],
                marker_color=["#818cf8", "#34d399", "#f472b6"],
                text=[f"{v}" for v in [phase_stats[0], phase_stats[3], phase_stats[6]]],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"{selected_team} — Avg Runs by Phase",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Phase-wise run rate
            fig = go.Figure(go.Bar(
                x=["Powerplay", "Middle", "Death"],
                y=[phase_stats[2], phase_stats[5], phase_stats[8]],
                marker_color=["#818cf8", "#34d399", "#f472b6"],
                text=[f"{v}" for v in [phase_stats[2], phase_stats[5], phase_stats[8]]],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"{selected_team} — Avg Run Rate by Phase",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(2)
        with cols[0]:
            st.metric("Boundary Dependency", f"{phase_stats[9]:.1f}%")
        with cols[1]:
            st.metric("Dot Ball %", f"{phase_stats[10]:.1f}%")

    except Exception as e:
        st.error(f"Error loading phase stats: {e}")

    # ── Opposition Analysis ──
    st.markdown("### 🆚 Performance vs Opposition")
    try:
        opp_df = conn.execute(f"""
            SELECT
                CASE WHEN team1 = '{selected_team}' THEN team2 ELSE team1 END AS opposition,
                COUNT(*) AS matches,
                SUM(CASE WHEN winner = '{selected_team}' THEN 1 ELSE 0 END) AS wins,
                ROUND(SUM(CASE WHEN winner = '{selected_team}' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) AS win_pct
            FROM silver.matches
            WHERE (team1 = '{selected_team}' OR team2 = '{selected_team}')
              AND source IN ({src_clause})
              AND winner IS NOT NULL
            GROUP BY opposition
            HAVING COUNT(*) >= 3
            ORDER BY win_pct DESC
        """).df()

        if not opp_df.empty:
            fig = px.bar(
                opp_df, x="opposition", y="win_pct",
                color="win_pct", color_continuous_scale="RdYlGn",
                title=f"{selected_team} — Win % vs Each Opposition",
                text="win_pct",
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-45, yaxis_range=[0, 100],
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading opposition stats: {e}")

    conn.close()
