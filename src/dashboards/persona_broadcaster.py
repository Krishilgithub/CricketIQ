"""
Broadcaster / Commentator Dashboard — Records, milestones, stories, battles.
"""

import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_conn():
    from src.warehouse.schema import get_connection
    return get_connection()


def render_broadcaster_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>📺 Broadcaster / Commentator Dashboard</h1>
        <p>Records, milestones, head-to-head battles, and story hooks</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        conn = get_conn()
    except Exception as e:
        st.error(f"Cannot connect to warehouse: {e}")
        return

    with st.sidebar:
        st.markdown("### 🎛️ Broadcaster Filters")
        source_filter = st.multiselect(
            "Data Source", ["t20i", "ipl", "bbl", "cpl"],
            default=["t20i"], key="bcast_src"
        )

    if not source_filter:
        st.warning("Select at least one source.")
        conn.close()
        return

    src_clause = ", ".join(f"'{s}'" for s in source_filter)

    tab1, tab2, tab3 = st.tabs([
        "🏆 Records & Milestones", "⚔️ Player Battles", "📊 Clutch Performers"
    ])

    with tab1:
        _render_records(conn, src_clause)
    with tab2:
        _render_battles(conn, src_clause)
    with tab3:
        _render_clutch(conn, src_clause)

    conn.close()


def _render_records(conn, src_clause):
    """All-time records and milestones."""
    st.markdown("### 🏆 Records & Milestones")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏏 Highest Individual Scores")
        try:
            top_scores = conn.execute(f"""
                SELECT p.player_name, b.runs_scored, b.balls_faced,
                       b.fours, b.sixes,
                       ROUND(b.strike_rate, 1) AS sr,
                       b.match_id
                FROM gold.fact_batting_innings b
                JOIN gold.dim_player p ON b.player_key = p.player_key
                WHERE b.source IN ({src_clause})
                ORDER BY b.runs_scored DESC
                LIMIT 15
            """).df()
            if not top_scores.empty:
                st.dataframe(top_scores, use_container_width=True, hide_index=True)
        except Exception as e:
            st.info(f"Not available: {e}")

    with col2:
        st.markdown("#### 🎯 Best Bowling Figures")
        try:
            top_bowling = conn.execute(f"""
                SELECT p.player_name, b.wickets_taken, b.runs_conceded,
                       ROUND(b.economy_rate, 2) AS economy,
                       b.dot_balls, b.match_id
                FROM gold.fact_bowling_innings b
                JOIN gold.dim_player p ON b.player_key = p.player_key
                WHERE b.source IN ({src_clause})
                ORDER BY b.wickets_taken DESC, b.runs_conceded ASC
                LIMIT 15
            """).df()
            if not top_bowling.empty:
                st.dataframe(top_bowling, use_container_width=True, hide_index=True)
        except Exception as e:
            st.info(f"Not available: {e}")

    st.markdown("#### 📈 Highest Team Totals")
    try:
        top_totals = conn.execute(f"""
            SELECT t.team_name, fi.total_runs, fi.total_wickets,
                   fi.fours, fi.sixes, fi.match_id
            FROM gold.fact_innings_summary fi
            JOIN gold.dim_team t ON fi.team_key = t.team_key
            WHERE fi.source IN ({src_clause})
            ORDER BY fi.total_runs DESC
            LIMIT 15
        """).df()
        if not top_totals.empty:
            fig = px.bar(
                top_totals, x="team_name", y="total_runs",
                color="sixes", color_continuous_scale="Hot",
                title="Top 15 Highest Team Totals",
                text="total_runs",
                hover_data=["fours", "sixes", "match_id"],
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Not available: {e}")

    # Six-hitting records
    st.markdown("#### 💥 Most Sixes (Career)")
    try:
        six_hitters = conn.execute(f"""
            SELECT p.player_name, SUM(b.sixes) AS total_sixes,
                   SUM(b.runs_scored) AS total_runs, COUNT(*) AS innings
            FROM gold.fact_batting_innings b
            JOIN gold.dim_player p ON b.player_key = p.player_key
            WHERE b.source IN ({src_clause})
            GROUP BY p.player_name
            HAVING SUM(b.sixes) > 0
            ORDER BY total_sixes DESC
            LIMIT 15
        """).df()
        if not six_hitters.empty:
            fig = px.bar(
                six_hitters, x="player_name", y="total_sixes",
                color="total_runs", color_continuous_scale="Viridis",
                title="Top 15 Six-Hitters",
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Not available: {e}")


def _render_battles(conn, src_clause):
    """Batter vs bowler battles from ball-by-ball data."""
    st.markdown("### ⚔️ Batter vs Bowler Battles")

    try:
        battles = conn.execute(f"""
            SELECT d.batter, d.bowler,
                   COUNT(*) AS balls,
                   SUM(d.batter_runs) AS runs,
                   SUM(CASE WHEN d.is_wicket AND d.player_out = d.batter THEN 1 ELSE 0 END) AS dismissals,
                   ROUND(SUM(d.batter_runs) * 100.0 / NULLIF(COUNT(*), 0), 1) AS sr,
                   SUM(CASE WHEN d.is_dot_ball THEN 1 ELSE 0 END) AS dots,
                   SUM(CASE WHEN d.is_boundary_six THEN 1 ELSE 0 END) AS sixes
            FROM silver.deliveries d
            WHERE d.source IN ({src_clause})
            GROUP BY d.batter, d.bowler
            HAVING COUNT(*) >= 20
            ORDER BY balls DESC
            LIMIT 50
        """).df()

        if battles.empty:
            st.info("Not enough ball-by-ball data for battle analysis.")
            return

        st.dataframe(battles, use_container_width=True, hide_index=True)

        # Top battles by volume
        fig = px.scatter(
            battles.head(30), x="runs", y="dismissals",
            size="balls", color="sr",
            hover_name=battles.head(30).apply(
                lambda r: f"{r['batter']} vs {r['bowler']}", axis=1
            ),
            color_continuous_scale="RdYlGn",
            title="Batter vs Bowler (size=balls, color=SR)",
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")


def _render_clutch(conn, src_clause):
    """Clutch performers in close matches."""
    st.markdown("### 📊 Clutch Performers (Close Matches)")
    st.info("Close matches: won by ≤ 15 runs or ≤ 3 wickets")

    try:
        clutch = conn.execute(f"""
            WITH close_matches AS (
                SELECT match_id FROM silver.matches
                WHERE source IN ({src_clause})
                  AND winner IS NOT NULL
                  AND ((result_type = 'runs' AND result_margin <= 15)
                    OR (result_type = 'wickets' AND result_margin <= 3))
            )
            SELECT p.player_name,
                   COUNT(*) AS close_innings,
                   ROUND(AVG(b.runs_scored), 1) AS avg_in_close,
                   ROUND(AVG(b.strike_rate), 1) AS avg_sr_close,
                   SUM(b.runs_scored) AS total_in_close
            FROM gold.fact_batting_innings b
            JOIN gold.dim_player p ON b.player_key = p.player_key
            WHERE b.match_id IN (SELECT match_id FROM close_matches)
              AND b.balls_faced >= 5
            GROUP BY p.player_name
            HAVING COUNT(*) >= 5
            ORDER BY avg_in_close DESC
            LIMIT 20
        """).df()

        if clutch.empty:
            st.info("No clutch data available.")
            return

        fig = px.bar(
            clutch, x="player_name", y="avg_in_close",
            color="avg_sr_close", color_continuous_scale="Plasma",
            title="Top Clutch Performers (Avg in Close Matches)",
            text="avg_in_close",
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(clutch, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")
