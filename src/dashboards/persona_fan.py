"""
Fan / ICC / Tournament Organizer Dashboard — Tournament health, entertainment, rankings.
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


def render_fan_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Fan / ICC Dashboard</h1>
        <p>Tournament health, competitive balance, entertainment metrics, power rankings</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        conn = get_conn()
    except Exception as e:
        st.error(f"Cannot connect to warehouse: {e}")
        return

    with st.sidebar:
        st.markdown("### 🎛️ Fan Filters")
        source_filter = st.multiselect(
            "Data Source", ["t20i", "ipl", "bbl", "cpl"],
            default=["t20i"], key="fan_src"
        )

    if not source_filter:
        st.warning("Select at least one source.")
        conn.close()
        return

    src_clause = ", ".join(f"'{s}'" for s in source_filter)

    tab1, tab2, tab3 = st.tabs([
        "🏆 Power Rankings", "🎭 Entertainment Metrics", "📊 Tournament Overview"
    ])

    with tab1:
        _render_power_rankings(conn, src_clause)
    with tab2:
        _render_entertainment(conn, src_clause)
    with tab3:
        _render_tournament_overview(conn, src_clause)

    conn.close()


def _render_power_rankings(conn, src_clause):
    """Team power rankings based on weighted recent performance."""
    st.markdown("### 🏆 Team Power Rankings")
    st.info("Weighted by recency: recent matches count more. Only teams with 10+ matches.")

    try:
        # Get all matches with dates
        matches_df = conn.execute(f"""
            SELECT m.match_id, m.team1, m.team2, m.winner, m.match_date
            FROM silver.matches m
            WHERE m.source IN ({src_clause})
              AND m.winner IS NOT NULL
              AND m.match_date IS NOT NULL
            ORDER BY m.match_date
        """).df()

        if matches_df.empty:
            st.info("No match data available.")
            return

        # Calculate weighted win rate (recent matches weigh more)
        all_teams = set(matches_df["team1"].tolist() + matches_df["team2"].tolist())
        rankings = []

        for team in all_teams:
            team_matches = matches_df[
                (matches_df["team1"] == team) | (matches_df["team2"] == team)
            ].copy()

            if len(team_matches) < 10:
                continue

            team_matches = team_matches.sort_values("match_date")
            n = len(team_matches)
            weights = np.linspace(0.5, 1.5, n)  # older=0.5, newest=1.5
            wins = (team_matches["winner"] == team).astype(float).values

            weighted_wr = np.average(wins, weights=weights) * 100
            raw_wr = wins.mean() * 100

            rankings.append({
                "Team": team,
                "Matches": n,
                "Wins": int(wins.sum()),
                "Win Rate %": round(raw_wr, 1),
                "Power Rating": round(weighted_wr, 1),
            })

        rank_df = pd.DataFrame(rankings).sort_values("Power Rating", ascending=False)
        rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))

        # Top 15 bar chart
        top = rank_df.head(15)
        fig = px.bar(
            top, x="Team", y="Power Rating",
            color="Power Rating", color_continuous_scale="Turbo",
            title="Top 15 Teams by Power Rating",
            text="Power Rating",
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45, yaxis_range=[0, 100],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(rank_df.head(25), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")


def _render_entertainment(conn, src_clause):
    """Entertainment and competitiveness metrics."""
    st.markdown("### 🎭 Entertainment Metrics")

    try:
        stats = conn.execute(f"""
            SELECT
                COUNT(DISTINCT m.match_id) AS total_matches,
                ROUND(AVG(fi.total_runs), 1) AS avg_runs_per_innings,
                ROUND(AVG(fi.fours + fi.sixes), 1) AS avg_boundaries,
                ROUND(AVG(fi.sixes), 1) AS avg_sixes,
                ROUND(AVG(fi.run_rate), 2) AS avg_run_rate
            FROM gold.fact_innings_summary fi
            JOIN silver.matches m ON fi.match_id = m.match_id
            WHERE fi.source IN ({src_clause})
        """).fetchone()

        close_match_stats = conn.execute(f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN (result_type = 'runs' AND result_margin <= 15)
                         OR (result_type = 'wickets' AND result_margin <= 3) THEN 1 ELSE 0 END) AS close,
                SUM(CASE WHEN (result_type = 'runs' AND result_margin <= 5)
                         OR (result_type = 'wickets' AND result_margin <= 1) THEN 1 ELSE 0 END) AS nail_biter
            FROM silver.matches
            WHERE source IN ({src_clause}) AND winner IS NOT NULL
        """).fetchone()

        close_pct = (close_match_stats[1] / close_match_stats[0] * 100) if close_match_stats[0] > 0 else 0
        nail_pct = (close_match_stats[2] / close_match_stats[0] * 100) if close_match_stats[0] > 0 else 0

        cols = st.columns(6)
        with cols[0]:
            st.metric("Matches", stats[0])
        with cols[1]:
            st.metric("Avg Runs/Innings", stats[1])
        with cols[2]:
            st.metric("Avg Boundaries", stats[2])
        with cols[3]:
            st.metric("Avg Sixes", stats[3])
        with cols[4]:
            st.metric("Close Match %", f"{close_pct:.1f}%")
        with cols[5]:
            st.metric("Nail-biters %", f"{nail_pct:.1f}%")

        # Excitement metrics over time
        yearly = conn.execute(f"""
            SELECT d.year,
                   ROUND(AVG(fi.total_runs), 1) AS avg_score,
                   ROUND(AVG(fi.sixes), 1) AS avg_sixes,
                   ROUND(AVG(fi.run_rate), 2) AS avg_rr
            FROM gold.fact_innings_summary fi
            JOIN silver.matches m ON fi.match_id = m.match_id
            LEFT JOIN gold.dim_date d ON m.match_date = d.full_date
            WHERE fi.source IN ({src_clause}) AND d.year IS NOT NULL
            GROUP BY d.year ORDER BY d.year
        """).df()

        if not yearly.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly["year"], y=yearly["avg_score"],
                mode="lines+markers", name="Avg Score",
                line=dict(color="#818cf8", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=yearly["year"], y=yearly["avg_sixes"] * 10,
                mode="lines+markers", name="Avg Sixes (×10)",
                line=dict(color="#f472b6", width=2),
            ))
            fig.update_layout(
                title="Entertainment Trend Over Years",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")


def _render_tournament_overview(conn, src_clause):
    """Tournament-specific overview."""
    st.markdown("### 📊 Tournament Overview")

    try:
        tournaments = conn.execute("""
            SELECT tournament_key, event_name, season, is_world_cup, is_league
            FROM gold.dim_tournament
            ORDER BY season DESC, event_name
        """).df()

        if tournaments.empty:
            st.info("No tournament data available.")
            return

        # World Cup tournaments
        wc = tournaments[tournaments["is_world_cup"] == True]
        leagues = tournaments[tournaments["is_league"] == True]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🏆 World Cup Tournaments")
            st.dataframe(wc[["event_name", "season"]], use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### 🏟️ League Tournaments")
            league_summary = leagues.groupby("event_name").agg(
                seasons=("season", "count")
            ).reset_index().sort_values("seasons", ascending=False)
            st.dataframe(league_summary, use_container_width=True, hide_index=True)

        # Venue utilization
        st.markdown("#### 📍 Venue Utilization")
        venue_usage = conn.execute("""
            SELECT venue_name, city, matches_hosted
            FROM gold.dim_venue
            WHERE matches_hosted >= 5
            ORDER BY matches_hosted DESC
            LIMIT 20
        """).df()

        if not venue_usage.empty:
            fig = px.bar(
                venue_usage, x="venue_name", y="matches_hosted",
                color="city", title="Top 20 Most Used Venues",
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
