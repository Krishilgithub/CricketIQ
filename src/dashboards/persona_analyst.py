"""
Data Analyst / Statistician Dashboard — Advanced stats, form tracking, correlations.
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


def render_analyst_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>📈 Data Analyst Dashboard</h1>
        <p>Advanced statistical analysis, correlations, form indices, and deep-dive metrics</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        conn = get_conn()
    except Exception as e:
        st.error(f"Cannot connect to warehouse: {e}")
        return

    with st.sidebar:
        st.markdown("### 🎛️ Analyst Filters")
        source_filter = st.multiselect(
            "Data Source", ["t20i", "ipl", "bbl", "cpl"],
            default=["t20i"], key="analyst_src"
        )
        min_innings = st.slider("Min Innings (player filter)", 5, 50, 15, key="analyst_min_inn")

    if not source_filter:
        st.warning("Select at least one source.")
        conn.close()
        return

    src_clause = ", ".join(f"'{s}'" for s in source_filter)

    tab1, tab2, tab3 = st.tabs([
        "📊 Feature Correlations", "📈 Player Form Index", "🎯 Impact Scores"
    ])

    with tab1:
        _render_correlations(conn, src_clause)
    with tab2:
        _render_form_index(conn, src_clause, min_innings)
    with tab3:
        _render_impact_scores(conn, src_clause, min_innings)

    conn.close()


def _render_correlations(conn, src_clause):
    """Feature correlation analysis for match outcomes."""
    st.markdown("### 📊 Feature Correlations")

    try:
        innings_df = conn.execute(f"""
            SELECT total_runs, total_wickets, total_overs, run_rate,
                   pp_runs, pp_wickets, pp_run_rate,
                   middle_runs, middle_wickets, middle_run_rate,
                   death_runs, death_wickets, death_run_rate,
                   boundary_runs, fours, sixes, dot_balls, extras
            FROM gold.fact_innings_summary
            WHERE source IN ({src_clause})
              AND total_runs IS NOT NULL
        """).df()

        if innings_df.empty:
            st.info("No data available.")
            return

        corr = innings_df.corr()

        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix",
            aspect="auto",
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Key insights
        st.markdown("#### 🔑 Key Correlations with Total Runs")
        total_corr = corr["total_runs"].drop("total_runs").sort_values(ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Strongest Positive:**")
            for feat, val in total_corr.head(5).items():
                st.markdown(f"- `{feat}`: **{val:.3f}**")
        with col2:
            st.markdown("**Strongest Negative:**")
            for feat, val in total_corr.tail(3).items():
                st.markdown(f"- `{feat}`: **{val:.3f}**")

    except Exception as e:
        st.error(f"Error: {e}")


def _render_form_index(conn, src_clause, min_innings):
    """Player form index based on recent performances."""
    st.markdown("### 📈 Player Form Index (Batting)")
    st.info("Form Index = Weighted average of last N innings (recent innings weighted higher)")

    n_innings = st.slider("Form window (last N innings)", 3, 20, 10, key="analyst_form_n")

    try:
        # Get batting data with dates
        batters = conn.execute(f"""
            SELECT p.player_name, b.runs_scored, b.strike_rate,
                   b.balls_faced, d.full_date
            FROM gold.fact_batting_innings b
            JOIN gold.dim_player p ON b.player_key = p.player_key
            LEFT JOIN gold.dim_date d ON b.date_key = d.date_key
            WHERE b.source IN ({src_clause})
              AND b.balls_faced >= 5
            ORDER BY p.player_name, d.full_date DESC
        """).df()

        if batters.empty:
            st.info("No batting data available.")
            return

        # Calculate form index
        form_records = []
        for player, grp in batters.groupby("player_name"):
            if len(grp) < min_innings:
                continue
            recent = grp.head(n_innings)
            weights = np.arange(1, len(recent) + 1)  # newer = higher weight
            weighted_avg = np.average(recent["runs_scored"].values, weights=weights)
            overall_avg = grp["runs_scored"].mean()
            avg_sr = recent["strike_rate"].mean()
            form_records.append({
                "Player": player,
                "Form Index": round(weighted_avg, 1),
                "Overall Avg": round(overall_avg, 1),
                "Form vs Avg": round(weighted_avg - overall_avg, 1),
                "Recent SR": round(avg_sr, 1) if not pd.isna(avg_sr) else 0,
                "Total Innings": len(grp),
            })

        form_df = pd.DataFrame(form_records).sort_values("Form Index", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔥 Hottest Form (Top 15)")
            top = form_df.head(15)
            fig = px.bar(
                top, x="Player", y="Form Index",
                color="Form vs Avg", color_continuous_scale="RdYlGn",
                title="Top Players by Form Index",
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### ❄️ Coldest Form (Bottom 15)")
            bottom = form_df.tail(15).sort_values("Form Index", ascending=True)
            fig = px.bar(
                bottom, x="Player", y="Form Index",
                color="Form vs Avg", color_continuous_scale="RdYlGn",
                title="Lowest Form Index Players",
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(form_df.head(50), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")


def _render_impact_scores(conn, src_clause, min_innings):
    """Player impact scores combining batting + bowling contributions."""
    st.markdown("### 🎯 Player Impact Scores")
    st.info("Impact = (Runs + Wickets×25 + Fours×2 + Sixes×3) / Matches")

    try:
        impact_df = conn.execute(f"""
            WITH batting AS (
                SELECT player_key,
                       SUM(runs_scored) AS total_runs,
                       SUM(fours) AS total_fours,
                       SUM(sixes) AS total_sixes,
                       COUNT(*) AS bat_innings
                FROM gold.fact_batting_innings
                WHERE source IN ({src_clause})
                GROUP BY player_key
            ),
            bowling AS (
                SELECT player_key,
                       SUM(wickets_taken) AS total_wickets,
                       SUM(dot_balls) AS total_dots,
                       COUNT(*) AS bowl_innings
                FROM gold.fact_bowling_innings
                WHERE source IN ({src_clause})
                GROUP BY player_key
            )
            SELECT p.player_name,
                   COALESCE(ba.total_runs, 0) AS runs,
                   COALESCE(bo.total_wickets, 0) AS wickets,
                   COALESCE(ba.total_fours, 0) AS fours,
                   COALESCE(ba.total_sixes, 0) AS sixes,
                   COALESCE(ba.bat_innings, 0) AS bat_inn,
                   COALESCE(bo.bowl_innings, 0) AS bowl_inn,
                   GREATEST(COALESCE(ba.bat_innings, 0), COALESCE(bo.bowl_innings, 0)) AS max_inn,
                   ROUND((COALESCE(ba.total_runs, 0) + COALESCE(bo.total_wickets, 0) * 25
                          + COALESCE(ba.total_fours, 0) * 2 + COALESCE(ba.total_sixes, 0) * 3)
                         * 1.0 / GREATEST(COALESCE(ba.bat_innings, 0), COALESCE(bo.bowl_innings, 0), 1), 1) AS impact
            FROM gold.dim_player p
            LEFT JOIN batting ba ON p.player_key = ba.player_key
            LEFT JOIN bowling bo ON p.player_key = bo.player_key
            WHERE GREATEST(COALESCE(ba.bat_innings, 0), COALESCE(bo.bowl_innings, 0)) >= {min_innings}
            ORDER BY impact DESC
            LIMIT 50
        """).df()

        if impact_df.empty:
            st.info("No data available.")
            return

        fig = px.bar(
            impact_df.head(25), x="player_name", y="impact",
            color="impact", color_continuous_scale="Plasma",
            title="Top 25 Players by Impact Score",
            hover_data=["runs", "wickets", "fours", "sixes"],
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Scatter: runs vs wickets
        fig2 = px.scatter(
            impact_df, x="runs", y="wickets",
            size="impact", color="impact",
            hover_name="player_name",
            color_continuous_scale="Viridis",
            title="All-rounder Map: Runs vs Wickets",
        )
        fig2.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(impact_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")
