"""
Exploratory Data Analysis (EDA) Dashboard — distributions, patterns, trends.
"""

import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_conn():
    from src.warehouse.schema import get_connection
    return get_connection()


def render_eda():
    st.markdown("""
    <div class="main-header">
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Discover patterns, trends, and insights across T20 cricket data</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        conn = get_conn()
    except Exception as e:
        st.error(f"Cannot connect to warehouse: {e}")
        return

    # Sidebar filters
    with st.sidebar:
        st.markdown("### 🎛️ EDA Filters")
        sources = conn.execute(
            "SELECT DISTINCT source FROM silver.matches ORDER BY source"
        ).fetchall()
        source_list = [s[0] for s in sources]
        selected_sources = st.multiselect(
            "Data Source", source_list, default=source_list, key="eda_src"
        )

    if not selected_sources:
        st.warning("Select at least one source.")
        conn.close()
        return

    src_clause = ", ".join(f"'{s}'" for s in selected_sources)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏏 Score Distributions", "🏆 Win Patterns",
        "📍 Venue Analysis", "⭐ Top Performers", "📈 Trends Over Time"
    ])

    with tab1:
        _render_score_distributions(conn, src_clause)
    with tab2:
        _render_win_patterns(conn, src_clause)
    with tab3:
        _render_venue_analysis(conn, src_clause)
    with tab4:
        _render_top_performers(conn, src_clause)
    with tab5:
        _render_trends(conn, src_clause)

    conn.close()


# ──────────────────────────────────────────────────────────────
# TAB 1: Score Distributions
# ──────────────────────────────────────────────────────────────
def _render_score_distributions(conn, src_clause):
    st.markdown("### 🏏 Score Distributions")

    innings_df = conn.execute(f"""
        SELECT
            fi.match_id, fi.innings, fi.total_runs, fi.total_wickets,
            fi.total_overs, fi.run_rate, fi.pp_runs, fi.middle_runs,
            fi.death_runs, fi.fours, fi.sixes, fi.dot_balls,
            fi.boundary_runs, fi.source
        FROM gold.fact_innings_summary fi
        WHERE fi.source IN ({src_clause})
          AND fi.total_runs IS NOT NULL
    """).df()

    if innings_df.empty:
        st.info("No innings data available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # 1st vs 2nd innings distributions
        fig = px.histogram(
            innings_df, x="total_runs", color="innings",
            nbins=40, barmode="overlay", opacity=0.7,
            color_discrete_map={1: "#818cf8", 2: "#f472b6"},
            title="Score Distribution: 1st vs 2nd Innings",
            labels={"total_runs": "Total Runs", "innings": "Innings"},
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Run rate distribution
        fig = px.histogram(
            innings_df[innings_df["run_rate"] > 0],
            x="run_rate", nbins=40,
            color_discrete_sequence=["#34d399"],
            title="Run Rate Distribution",
            labels={"run_rate": "Run Rate"},
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Phase-wise scoring
    st.markdown("#### Phase-wise Scoring Breakdown")
    phase_means = innings_df[["pp_runs", "middle_runs", "death_runs"]].mean()
    fig = go.Figure(go.Bar(
        x=["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"],
        y=[phase_means["pp_runs"], phase_means["middle_runs"], phase_means["death_runs"]],
        marker_color=["#818cf8", "#34d399", "#f472b6"],
        text=[f"{v:.1f}" for v in [phase_means["pp_runs"], phase_means["middle_runs"], phase_means["death_runs"]]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Average Runs by Phase",
        yaxis_title="Avg Runs", template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Boundary analysis
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            innings_df, x="fours", y="sixes",
            color="total_runs", size="total_runs",
            color_continuous_scale="Viridis",
            title="Fours vs Sixes (sized by total runs)",
            labels={"fours": "Fours", "sixes": "Sixes", "total_runs": "Total Runs"},
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        boundary_pct = innings_df["boundary_runs"] / innings_df["total_runs"].replace(0, np.nan) * 100
        fig = px.histogram(
            boundary_pct.dropna(), nbins=30,
            title="Boundary Dependency (% runs from boundaries)",
            color_discrete_sequence=["#fbbf24"],
            labels={"value": "Boundary %"},
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 2: Win Patterns
# ──────────────────────────────────────────────────────────────
def _render_win_patterns(conn, src_clause):
    st.markdown("### 🏆 Win Patterns & Toss Analysis")

    matches_df = conn.execute(f"""
        SELECT match_id, team1, team2, winner, toss_winner, toss_decision,
               result_type, result_margin, bat_first_team, chase_team,
               toss_win_match_win, source
        FROM silver.matches
        WHERE source IN ({src_clause}) AND winner IS NOT NULL
    """).df()

    if matches_df.empty:
        st.info("No match data available.")
        return

    col1, col2, col3 = st.columns(3)

    # Toss decision breakdown
    with col1:
        toss_dec = matches_df["toss_decision"].value_counts().reset_index()
        toss_dec.columns = ["Decision", "Count"]
        fig = px.pie(
            toss_dec, names="Decision", values="Count",
            title="Toss Decision Split",
            color_discrete_sequence=["#818cf8", "#f472b6"],
            hole=0.4,
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Toss win → match win
    with col2:
        toss_match = matches_df["toss_win_match_win"].value_counts().reset_index()
        toss_match.columns = ["Toss Winner Won?", "Count"]
        toss_match["Toss Winner Won?"] = toss_match["Toss Winner Won?"].map(
            {True: "Yes", False: "No"}
        )
        fig = px.pie(
            toss_match, names="Toss Winner Won?", values="Count",
            title="Toss Winner → Match Winner?",
            color_discrete_sequence=["#34d399", "#ef4444"],
            hole=0.4,
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Bat first vs chase win rate
    with col3:
        bat_first_wins = len(matches_df[matches_df["winner"] == matches_df["bat_first_team"]])
        chase_wins = len(matches_df[matches_df["winner"] == matches_df["chase_team"]])
        fig = px.pie(
            pd.DataFrame({"Strategy": ["Bat First Win", "Chase Win"],
                          "Count": [bat_first_wins, chase_wins]}),
            names="Strategy", values="Count",
            title="Bat First vs Chase Win Rate",
            color_discrete_sequence=["#fbbf24", "#818cf8"],
            hole=0.4,
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Result margin distributions
    st.markdown("#### Result Margin Analysis")
    col1, col2 = st.columns(2)
    with col1:
        run_wins = matches_df[matches_df["result_type"] == "runs"]
        if not run_wins.empty:
            fig = px.histogram(
                run_wins, x="result_margin", nbins=30,
                title="Win Margin (Runs)", color_discrete_sequence=["#818cf8"],
                labels={"result_margin": "Margin (Runs)"},
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        wkt_wins = matches_df[matches_df["result_type"] == "wickets"]
        if not wkt_wins.empty:
            fig = px.histogram(
                wkt_wins, x="result_margin", nbins=10,
                title="Win Margin (Wickets)", color_discrete_sequence=["#f472b6"],
                labels={"result_margin": "Margin (Wickets)"},
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Top winning teams
    st.markdown("#### Top Winning Teams")
    top_winners = matches_df["winner"].value_counts().head(15).reset_index()
    top_winners.columns = ["Team", "Wins"]
    fig = px.bar(
        top_winners, x="Team", y="Wins", color="Wins",
        color_continuous_scale="Viridis",
        title="Most Match Wins",
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB 3: Venue Analysis
# ──────────────────────────────────────────────────────────────
def _render_venue_analysis(conn, src_clause):
    st.markdown("### 📍 Venue Analysis")

    venue_df = conn.execute("""
        SELECT venue_name, city, avg_first_innings_score, avg_second_innings_score,
               matches_hosted
        FROM gold.dim_venue
        WHERE matches_hosted >= 3
        ORDER BY matches_hosted DESC
    """).df()

    if venue_df.empty:
        st.info("No venue data available.")
        return

    # Top venues by matches
    top_venues = venue_df.head(20)

    fig = px.bar(
        top_venues, x="venue_name", y="matches_hosted",
        color="avg_first_innings_score",
        color_continuous_scale="Turbo",
        title="Top Venues by Matches Hosted (colored by avg 1st innings score)",
        labels={"venue_name": "Venue", "matches_hosted": "Matches",
                "avg_first_innings_score": "Avg 1st Inn"},
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-45, height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 1st vs 2nd innings avg scores
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            venue_df[venue_df["matches_hosted"] >= 5],
            x="avg_first_innings_score", y="avg_second_innings_score",
            size="matches_hosted", hover_name="venue_name",
            title="1st vs 2nd Innings Avg Score by Venue",
            color="matches_hosted", color_continuous_scale="Viridis",
        )
        fig.add_shape(type="line", x0=80, x1=220, y0=80, y1=220,
                      line=dict(dash="dash", color="grey"))
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Highest scoring venues
        venue_df["score_diff"] = (
            venue_df["avg_first_innings_score"] - venue_df["avg_second_innings_score"]
        )
        sorted_v = venue_df[venue_df["matches_hosted"] >= 5].sort_values(
            "avg_first_innings_score", ascending=False
        ).head(15)
        fig = px.bar(
            sorted_v, x="venue_name", y="avg_first_innings_score",
            title="Highest Scoring Venues (Avg 1st Innings)",
            color_discrete_sequence=["#34d399"],
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        venue_df.sort_values("matches_hosted", ascending=False).head(30),
        use_container_width=True, hide_index=True,
    )


# ──────────────────────────────────────────────────────────────
# TAB 4: Top Performers
# ──────────────────────────────────────────────────────────────
def _render_top_performers(conn, src_clause):
    st.markdown("### ⭐ Top Performers")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏏 Top Run Scorers")
        try:
            batters = conn.execute(f"""
                SELECT p.player_name, SUM(b.runs_scored) AS total_runs,
                       COUNT(*) AS innings,
                       ROUND(AVG(b.strike_rate), 1) AS avg_sr,
                       SUM(b.fours) AS fours, SUM(b.sixes) AS sixes
                FROM gold.fact_batting_innings b
                JOIN gold.dim_player p ON b.player_key = p.player_key
                WHERE b.source IN ({src_clause})
                GROUP BY p.player_name
                HAVING COUNT(*) >= 10
                ORDER BY total_runs DESC
                LIMIT 20
            """).df()
            if not batters.empty:
                fig = px.bar(
                    batters, x="player_name", y="total_runs",
                    color="avg_sr", color_continuous_scale="Plasma",
                    title="Top 20 Run Scorers (colored by Avg SR)",
                )
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(batters, use_container_width=True, hide_index=True)
            else:
                st.info("No batting data available. Ensure Gold layer fact tables are populated.")
        except Exception as e:
            st.info(f"Batting stats not available: {e}")

    with col2:
        st.markdown("#### 🎯 Top Wicket Takers")
        try:
            bowlers = conn.execute(f"""
                SELECT p.player_name, SUM(b.wickets_taken) AS total_wickets,
                       COUNT(*) AS innings,
                       ROUND(AVG(b.economy_rate), 2) AS avg_econ,
                       SUM(b.dot_balls) AS dots
                FROM gold.fact_bowling_innings b
                JOIN gold.dim_player p ON b.player_key = p.player_key
                WHERE b.source IN ({src_clause})
                GROUP BY p.player_name
                HAVING COUNT(*) >= 10
                ORDER BY total_wickets DESC
                LIMIT 20
            """).df()
            if not bowlers.empty:
                fig = px.bar(
                    bowlers, x="player_name", y="total_wickets",
                    color="avg_econ", color_continuous_scale="RdYlGn_r",
                    title="Top 20 Wicket Takers (colored by Avg Economy)",
                )
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(bowlers, use_container_width=True, hide_index=True)
            else:
                st.info("No bowling data available. Ensure Gold layer fact tables are populated.")
        except Exception as e:
            st.info(f"Bowling stats not available: {e}")


# ──────────────────────────────────────────────────────────────
# TAB 5: Trends Over Time
# ──────────────────────────────────────────────────────────────
def _render_trends(conn, src_clause):
    st.markdown("### 📈 Trends Over Time")

    try:
        yearly = conn.execute(f"""
            SELECT
                d.year,
                COUNT(DISTINCT f.match_id) AS matches,
                ROUND(AVG(f.total_runs), 1) AS avg_score,
                ROUND(AVG(f.run_rate), 2) AS avg_rr,
                ROUND(AVG(f.pp_run_rate), 2) AS avg_pp_rr,
                ROUND(AVG(f.death_run_rate), 2) AS avg_death_rr,
                ROUND(AVG(f.fours + f.sixes), 1) AS avg_boundaries,
                ROUND(AVG(f.dot_balls), 1) AS avg_dots
            FROM gold.fact_innings_summary f
            JOIN gold.dim_venue v ON f.venue_key = v.venue_key
            LEFT JOIN silver.matches m ON f.match_id = m.match_id
            LEFT JOIN gold.dim_date d ON m.match_date = d.full_date
            WHERE f.source IN ({src_clause})
              AND d.year IS NOT NULL
            GROUP BY d.year
            ORDER BY d.year
        """).df()

        if yearly.empty:
            st.info("No time-series data available.")
            return

        # Average score trends
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(
                yearly, x="year", y="avg_score",
                title="Average Innings Score Over Years",
                markers=True, color_discrete_sequence=["#818cf8"],
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                yearly, x="year", y="matches",
                title="Matches Played Per Year",
                color_discrete_sequence=["#34d399"],
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Phase-wise run rate trends
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly["year"], y=yearly["avg_pp_rr"],
            mode="lines+markers", name="Powerplay RR",
            line=dict(color="#818cf8", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=yearly["year"], y=yearly["avg_rr"],
            mode="lines+markers", name="Overall RR",
            line=dict(color="#34d399", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=yearly["year"], y=yearly["avg_death_rr"],
            mode="lines+markers", name="Death RR",
            line=dict(color="#f472b6", width=2),
        ))
        fig.update_layout(
            title="Run Rate Trends by Phase",
            xaxis_title="Year", yaxis_title="Run Rate",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Boundaries trend
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(
                yearly, x="year", y="avg_boundaries",
                title="Avg Boundaries per Innings Over Years",
                markers=True, color_discrete_sequence=["#fbbf24"],
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                yearly, x="year", y="avg_dots",
                title="Avg Dot Balls per Innings Over Years",
                markers=True, color_discrete_sequence=["#ef4444"],
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading trends: {e}")
