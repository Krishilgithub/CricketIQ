"""
Team Selector Dashboard — Player comparison, consistency, matchup analysis.
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


def render_selector_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Team Selector Dashboard</h1>
        <p>Player consistency, matchup analysis, squad balance, and role coverage</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        conn = get_conn()
    except Exception as e:
        st.error(f"Cannot connect to warehouse: {e}")
        return

    with st.sidebar:
        st.markdown("### 🎛️ Selector Filters")
        source_filter = st.multiselect(
            "Data Source", ["t20i", "ipl", "bbl", "cpl"],
            default=["t20i"], key="selector_src"
        )
        min_innings = st.slider("Min Innings", 5, 30, 10, key="selector_min")

    if not source_filter:
        st.warning("Select at least one source.")
        conn.close()
        return

    src_clause = ", ".join(f"'{s}'" for s in source_filter)

    tab1, tab2, tab3 = st.tabs([
        "📊 Player Consistency", "🆚 Matchup Analysis", "👥 Squad Builder"
    ])

    with tab1:
        _render_consistency(conn, src_clause, min_innings)
    with tab2:
        _render_matchups(conn, src_clause)
    with tab3:
        _render_squad_builder(conn, src_clause, min_innings)

    conn.close()


def _render_consistency(conn, src_clause, min_innings):
    """Player consistency index analysis."""
    st.markdown("### 📊 Batting Consistency Index")
    st.info("Consistency = 1 − (StdDev / Mean). Closer to 1 = more reliable performer.")

    try:
        batters = conn.execute(f"""
            SELECT p.player_name,
                   COUNT(*) AS innings,
                   ROUND(AVG(b.runs_scored), 1) AS avg_runs,
                   ROUND(STDDEV(b.runs_scored), 1) AS std_runs,
                   ROUND(AVG(b.strike_rate), 1) AS avg_sr,
                   ROUND(STDDEV(b.strike_rate), 1) AS std_sr,
                   SUM(b.runs_scored) AS total_runs,
                   SUM(b.fours) AS fours,
                   SUM(b.sixes) AS sixes
            FROM gold.fact_batting_innings b
            JOIN gold.dim_player p ON b.player_key = p.player_key
            WHERE b.source IN ({src_clause})
              AND b.balls_faced >= 5
            GROUP BY p.player_name
            HAVING COUNT(*) >= {min_innings}
        """).df()

        if batters.empty:
            st.info("No batting data available.")
            return

        batters["consistency"] = np.round(
            1 - (batters["std_runs"] / batters["avg_runs"].replace(0, np.nan)), 3
        )
        batters = batters.sort_values("consistency", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                batters.head(20), x="player_name", y="consistency",
                color="avg_runs", color_continuous_scale="Viridis",
                title="Most Consistent Batters (Top 20)",
                text="consistency",
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-45, yaxis_range=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                batters, x="avg_runs", y="consistency",
                size="innings", color="avg_sr",
                hover_name="player_name",
                color_continuous_scale="Plasma",
                title="Average vs Consistency",
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(batters.head(30), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")


def _render_matchups(conn, src_clause):
    """Head-to-head team matchup analysis."""
    st.markdown("### 🆚 Team Head-to-Head Records")

    try:
        teams = conn.execute("""
            SELECT team_name FROM gold.dim_team
            WHERE is_icc_full_member = TRUE ORDER BY team_name
        """).fetchall()
        team_list = [t[0] for t in teams]

        col1, col2 = st.columns(2)
        with col1:
            team_a = st.selectbox("Team A", team_list, key="sel_team_a")
        with col2:
            team_b = st.selectbox("Team B", team_list, index=1, key="sel_team_b")

        if team_a == team_b:
            st.warning("Please select different teams.")
            return

        h2h = conn.execute(f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN winner = '{team_a}' THEN 1 ELSE 0 END) AS a_wins,
                SUM(CASE WHEN winner = '{team_b}' THEN 1 ELSE 0 END) AS b_wins,
                SUM(CASE WHEN winner IS NOT NULL AND winner != '{team_a}' AND winner != '{team_b}' THEN 1
                    WHEN winner IS NULL THEN 1 ELSE 0 END) AS draws_nr
            FROM silver.matches
            WHERE ((team1 = '{team_a}' AND team2 = '{team_b}')
                OR (team1 = '{team_b}' AND team2 = '{team_a}'))
              AND source IN ({src_clause})
        """).fetchone()

        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Matches", h2h[0])
        with cols[1]:
            st.metric(f"{team_a} Wins", h2h[1])
        with cols[2]:
            st.metric(f"{team_b} Wins", h2h[2])
        with cols[3]:
            st.metric("No Result", h2h[3])

        if h2h[0] > 0:
            fig = go.Figure(go.Pie(
                labels=[team_a, team_b, "No Result"],
                values=[h2h[1], h2h[2], h2h[3]],
                marker_colors=["#818cf8", "#f472b6", "#6b7280"],
                hole=0.4,
            ))
            fig.update_layout(
                title=f"{team_a} vs {team_b} — Head to Head",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")


def _render_squad_builder(conn, src_clause, min_innings):
    """Interactive squad builder with balance meter."""
    st.markdown("### 👥 Squad Builder — Role Analysis")

    try:
        # Classify players by role based on stats
        players = conn.execute(f"""
            WITH batting AS (
                SELECT player_key, SUM(runs_scored) AS runs, COUNT(*) AS bat_inn
                FROM gold.fact_batting_innings
                WHERE source IN ({src_clause})
                GROUP BY player_key
            ),
            bowling AS (
                SELECT player_key, SUM(wickets_taken) AS wkts, COUNT(*) AS bowl_inn,
                       ROUND(AVG(economy_rate), 2) AS econ
                FROM gold.fact_bowling_innings
                WHERE source IN ({src_clause})
                GROUP BY player_key
            )
            SELECT p.player_name, p.teams_played,
                   COALESCE(ba.runs, 0) AS runs,
                   COALESCE(bo.wkts, 0) AS wickets,
                   COALESCE(ba.bat_inn, 0) AS bat_inn,
                   COALESCE(bo.bowl_inn, 0) AS bowl_inn,
                   COALESCE(bo.econ, 0) AS economy,
                   CASE
                       WHEN COALESCE(ba.bat_inn, 0) >= {min_innings} AND COALESCE(bo.bowl_inn, 0) >= {min_innings}
                           THEN 'All-Rounder'
                       WHEN COALESCE(bo.bowl_inn, 0) >= {min_innings}
                           THEN 'Bowler'
                       WHEN COALESCE(ba.bat_inn, 0) >= {min_innings}
                           THEN 'Batter'
                       ELSE 'Other'
                   END AS role
            FROM gold.dim_player p
            LEFT JOIN batting ba ON p.player_key = ba.player_key
            LEFT JOIN bowling bo ON p.player_key = bo.player_key
            WHERE GREATEST(COALESCE(ba.bat_inn, 0), COALESCE(bo.bowl_inn, 0)) >= {min_innings}
            ORDER BY runs + wickets * 25 DESC
        """).df()

        if players.empty:
            st.info("No player data available.")
            return

        # Role distribution
        role_counts = players["role"].value_counts().reset_index()
        role_counts.columns = ["Role", "Count"]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                role_counts, names="Role", values="Count",
                title="Player Pool — Role Distribution",
                color_discrete_sequence=["#818cf8", "#34d399", "#f472b6", "#6b7280"],
                hole=0.4,
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Top All-Rounders:**")
            ars = players[players["role"] == "All-Rounder"].head(10)
            if not ars.empty:
                st.dataframe(ars[["player_name", "runs", "wickets", "economy"]],
                            use_container_width=True, hide_index=True)
            else:
                st.info("No all-rounders found with enough data.")

        st.markdown("#### Full Player Pool (by Impact)")
        st.dataframe(players.head(50), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")
