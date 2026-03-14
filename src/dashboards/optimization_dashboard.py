"""
Optimization Dashboard — Batting order optimizer, squad selection, balance analysis.
"""

import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render_optimization_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Optimization Engine</h1>
        <p>Optimal batting order, squad selection, and team balance analysis</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🏏 Batting Order", "👥 Squad Selection"])

    with tab1:
        _render_batting_order()
    with tab2:
        _render_squad_selection()


def _render_batting_order():
    st.markdown("### 🏏 Batting Order Optimizer")

    try:
        from src.warehouse.schema import get_connection
        conn = get_connection()

        teams = conn.execute("""
            SELECT team_name FROM gold.dim_team
            WHERE is_icc_full_member = TRUE ORDER BY team_name
        """).fetchall()
        team_list = [t[0] for t in teams]

        col1, col2 = st.columns(2)
        with col1:
            team = st.selectbox("Select Team", team_list, key="opt_bat_team")
        with col2:
            source = st.selectbox("Data Source", ["t20i", "ipl", "bbl", "cpl"], key="opt_bat_src")

        if st.button("⚡ Optimize Batting Order", key="opt_bat_btn"):
            with st.spinner("Computing optimal order..."):
                from src.optimization.batting_order import (
                    get_player_phase_stats, optimize_batting_order,
                )

                stats = get_player_phase_stats(conn, team, source=source)

                if stats.empty:
                    st.warning("Not enough data for this team.")
                else:
                    order = optimize_batting_order(stats)

                    st.markdown(f"#### Optimized Batting Order — {team}")

                    # Color-coded by role
                    role_colors = {
                        "Opener": "#818cf8",
                        "Top Order": "#34d399",
                        "Middle Order": "#fbbf24",
                        "Finisher": "#f472b6",
                        "Lower Order": "#6b7280",
                    }

                    fig = go.Figure()
                    for _, row in order.iterrows():
                        fig.add_trace(go.Bar(
                            x=[row["Position"]],
                            y=[row["Avg Runs"]],
                            name=row["Player"],
                            marker_color=role_colors.get(row["Role"], "#818cf8"),
                            text=f"{row['Player']}<br>{row['Role']}",
                            textposition="inside",
                            hovertemplate=(
                                f"<b>{row['Player']}</b><br>"
                                f"Role: {row['Role']}<br>"
                                f"Avg: {row['Avg Runs']}<br>"
                                f"SR: {row['Avg SR']}<br>"
                                f"PP Runs: {row['PP Runs']}<br>"
                                f"Death Runs: {row['Death Runs']}"
                            ),
                        ))

                    fig.update_layout(
                        title="Batting Order (by avg runs, colored by role)",
                        xaxis_title="Position", yaxis_title="Avg Runs",
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False, height=450,
                        xaxis=dict(dtick=1),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(order, use_container_width=True, hide_index=True)

        conn.close()

    except Exception as e:
        st.error(f"Error: {e}")


def _render_squad_selection():
    st.markdown("### 👥 Squad Selection Optimizer")

    try:
        from src.warehouse.schema import get_connection
        conn = get_connection()

        teams = conn.execute("""
            SELECT team_name FROM gold.dim_team
            WHERE is_icc_full_member = TRUE ORDER BY team_name
        """).fetchall()
        team_list = [t[0] for t in teams]

        col1, col2 = st.columns(2)
        with col1:
            team = st.selectbox("Select Team", team_list, key="opt_squad_team")
        with col2:
            source = st.selectbox("Data Source", ["t20i", "ipl", "bbl", "cpl"], key="opt_squad_src")

        # Constraints
        st.markdown("#### Constraints")
        c1, c2, c3 = st.columns(3)
        with c1:
            min_batters = st.number_input("Min Batters", 2, 7, 4, key="opt_min_bat")
            max_batters = st.number_input("Max Batters", min_batters, 8, 6, key="opt_max_bat")
        with c2:
            min_bowlers = st.number_input("Min Bowlers", 2, 6, 4, key="opt_min_bowl")
            max_bowlers = st.number_input("Max Bowlers", min_bowlers, 7, 5, key="opt_max_bowl")
        with c3:
            min_ar = st.number_input("Min All-rounders", 0, 4, 1, key="opt_min_ar")
            squad_size = st.number_input("Squad Size", 11, 15, 11, key="opt_squad_size")

        if st.button("⚡ Select Optimal Squad", key="opt_squad_btn"):
            with st.spinner("Selecting squad..."):
                from src.optimization.squad_selection import get_player_pool, select_squad

                pool = get_player_pool(conn, team, source=source)

                if pool.empty:
                    st.warning("Not enough data for this team.")
                else:
                    squad = select_squad(
                        pool,
                        squad_size=squad_size,
                        min_batters=min_batters,
                        min_bowlers=min_bowlers,
                        min_allrounders=min_ar,
                        max_batters=max_batters,
                        max_bowlers=max_bowlers,
                    )

                    st.markdown(f"#### Optimal Playing XI — {team}")

                    # Role distribution pie chart
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        role_dist = squad["role"].value_counts().reset_index()
                        role_dist.columns = ["Role", "Count"]
                        fig = px.pie(
                            role_dist, names="Role", values="Count",
                            title="Squad Balance",
                            color_discrete_sequence=["#818cf8", "#34d399", "#f472b6"],
                            hole=0.4,
                        )
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        fig = px.bar(
                            squad, x="player_name", y="value",
                            color="role", title="Selected Squad by Value Score",
                            color_discrete_sequence=["#818cf8", "#34d399", "#f472b6"],
                        )
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            xaxis_tickangle=-45,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(squad, use_container_width=True, hide_index=True)

                    # Squad strength meter
                    avg_value = squad["value"].mean()
                    st.markdown("#### Squad Strength")
                    st.progress(min(avg_value * 2, 1.0))
                    st.metric("Avg Player Value", f"{avg_value:.3f}")

        conn.close()

    except Exception as e:
        st.error(f"Error: {e}")
