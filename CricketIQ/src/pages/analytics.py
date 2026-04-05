"""
CricketIQ — Advanced Analytics Page
Power BI / Tableau-style interactive cricket analytics dashboard.
Sections: Team Intelligence | Player Intelligence | Match Intelligence | Venue Intelligence
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.pages.shared import get_hub_con

# ─────────────────────────────────────────────────────────────────────────────
# SQL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _q(sql: str) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame, silently returning empty on error."""
    try:
        return get_hub_con().execute(sql).df()
    except Exception:
        return pd.DataFrame()


def _years() -> list[int]:
    df = _q("SELECT DISTINCT YEAR(match_date) as yr FROM main_gold.fact_matches WHERE match_date IS NOT NULL ORDER BY yr")
    return df["yr"].dropna().astype(int).tolist() if not df.empty else list(range(2007, 2025))


def _teams() -> list[str]:
    """Return only teams that have played >= 50 matches (major T20I nations)."""
    df = _q("""
        SELECT team, COUNT(*) as matches
        FROM main_silver.slv_match_teams t
        JOIN main_gold.fact_matches m ON t.match_id = m.match_id
        GROUP BY team HAVING matches >= 50
        ORDER BY team
    """)
    return df["team"].tolist() if not df.empty else []


def _venues() -> list[str]:
    df = _q("SELECT DISTINCT venue FROM main_gold.fact_matches ORDER BY venue")
    return df["venue"].tolist() if not df.empty else []


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DARK = "plotly_dark"

def _empty_chart(msg="No data available for these filters."):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=14, color="#94a3b8"))
    fig.update_layout(template=DARK, height=300, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
    return fig


def _bar(df, x, y, color=None, title="", orientation="v", height=340, text=None, **kwargs):
    if df.empty:
        return _empty_chart()
    fig = px.bar(df, x=x, y=y, color=color, title=title, template=DARK,
                 orientation=orientation, text=text, **kwargs)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=60))
    if orientation == "v":
        fig.update_xaxes(tickangle=-40)
    return fig


def _line(df, x, y, color=None, title="", height=340, **kwargs):
    if df.empty:
        return _empty_chart()
    fig = px.line(df, x=x, y=y, color=color, title=title, template=DARK,
                  markers=True, **kwargs)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _scatter(df, x, y, color=None, size=None, title="", text=None, height=380, **kwargs):
    if df.empty:
        return _empty_chart()
    fig = px.scatter(df, x=x, y=y, color=color, size=size, title=title, text=text,
                     template=DARK, **kwargs)
    fig.update_traces(textposition="top center")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _heatmap(df, x, y, z, title="", height=420, **kwargs):
    if df.empty:
        return _empty_chart()
    pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
    fig = px.imshow(pivot, title=title, template=DARK, color_continuous_scale="RdYlGn",
                    aspect="auto", **kwargs)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _pie(df, names, values, title="", height=340):
    if df.empty:
        return _empty_chart()
    fig = px.pie(df, names=names, values=values, title=title, template=DARK,
                 hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=height)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL FILTER PANEL
# ─────────────────────────────────────────────────────────────────────────────

def _global_filters():
    all_teams = _teams()
    all_venues = _venues()
    all_years = _years()
    min_yr = min(all_years) if all_years else 2007
    max_yr = max(all_years) if all_years else 2024

    with st.sidebar.expander("🎛️ Global Filters", expanded=True):
        team = st.selectbox("🏏 Team", ["All Teams"] + all_teams, key="ga_team")
        opponent = st.selectbox("⚔️ Opponent", ["All Teams"] + all_teams, key="ga_opp")
        venue = st.selectbox("🏟️ Venue", ["All Venues"] + all_venues, key="ga_venue")
        year_range = st.slider("📅 Year Range", min_yr, max_yr, (max(min_yr, max_yr - 5), max_yr), key="ga_yr")

    return {
        "team": None if team == "All Teams" else team,
        "opponent": None if opponent == "All Teams" else opponent,
        "venue": None if venue == "All Venues" else venue,
        "year_from": year_range[0],
        "year_to": year_range[1],
    }


def _where(f: dict, team_col="toss_winner", venue_col="venue") -> str:
    clauses = [f"YEAR(match_date) BETWEEN {f['year_from']} AND {f['year_to']}"]
    if f["team"]:
        clauses.append(f"{team_col} = '{f['team']}'")
    if f["venue"]:
        clauses.append(f"{venue_col} = '{f['venue']}'")
    return "WHERE " + " AND ".join(clauses)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — TEAM INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

def _section_team(f: dict, st_tab):
    with st_tab:
        st.markdown("<div class='section-header'>🏏 Team Intelligence</div>", unsafe_allow_html=True)

        team_filter = f["team"] or "India"
        yr_from, yr_to = f["year_from"], f["year_to"]
        v_filter = f"AND venue = '{f['venue']}'" if f["venue"] else ""

        # ── 1. Win Rate Over Time (Annual) ─────────────────────────────────
        st.subheader("📈 Win Rate Trend Over Time")
        c1, c2 = st.columns(2)
        with c1:
            team_sel = st.selectbox("Select Team", _teams(),
                                    index=_teams().index(team_filter) if team_filter in _teams() else 0,
                                    key="t_wr_team")
        with c2:
            opponent_sel = st.selectbox("vs Opponent", ["All"] + _teams(), key="t_wr_opp")

        opp_clause = f"AND winner = '{opponent_sel}'" if opponent_sel != "All" else ""
        wr_sql = f"""
        SELECT YEAR(match_date) as year,
               COUNT(*) as total,
               SUM(CASE WHEN winner = '{team_sel}' THEN 1 ELSE 0 END) as wins,
               ROUND(100.0 * SUM(CASE WHEN winner = '{team_sel}' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct
        FROM main_gold.fact_matches
        WHERE (toss_winner = '{team_sel}' OR winner = '{team_sel}')
          AND YEAR(match_date) BETWEEN {yr_from} AND {yr_to}
          {v_filter} {opp_clause}
        GROUP BY 1 ORDER BY 1
        """
        wr_df = _q(wr_sql)
        st.plotly_chart(_line(wr_df, "year", "win_pct", title=f"{team_sel} Annual Win Rate %",
                              labels={"win_pct": "Win %", "year": "Year"}), use_container_width=True)

        # ── 2. Head-to-Head Comparison ─────────────────────────────────────
        st.subheader("⚔️ Head-to-Head Comparison")
        h2h_cols = st.columns(2)
        with h2h_cols[0]:
            h2h_team1 = st.selectbox("Team A", _teams(), key="h2h_a",
                                     index=_teams().index("India") if "India" in _teams() else 0)
        with h2h_cols[1]:
            h2h_team2 = st.selectbox("Team B", [t for t in _teams() if t != h2h_team1], key="h2h_b")

        h2h_sql = f"""
        SELECT
            SUM(CASE WHEN winner = '{h2h_team1}' THEN 1 ELSE 0 END) as team1_wins,
            SUM(CASE WHEN winner = '{h2h_team2}' THEN 1 ELSE 0 END) as team2_wins,
            SUM(CASE WHEN result_type IN ('no result','tie') THEN 1 ELSE 0 END) as other,
            COUNT(*) as total
        FROM main_gold.fact_matches m
        WHERE YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to}
          AND m.match_id IN (
              SELECT match_id FROM main_silver.slv_match_teams WHERE team = '{h2h_team1}'
          )
          AND m.match_id IN (
              SELECT match_id FROM main_silver.slv_match_teams WHERE team = '{h2h_team2}'
          )
        """
        h2h_df = _q(h2h_sql)
        # NaN-safe int helper
        def _si(val):
            try:
                import math
                return 0 if (val is None or (isinstance(val, float) and math.isnan(val))) else int(val)
            except Exception:
                return 0

        if not h2h_df.empty:
            row = h2h_df.iloc[0]
            t1w = _si(row.get("team1_wins", 0))
            t2w = _si(row.get("team2_wins", 0))
            oth = _si(row.get("other", 0))
            tot = _si(row.get("total", 0))
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric(f"🏆 {h2h_team1} Wins", t1w)
            mc2.metric(f"🏆 {h2h_team2} Wins", t2w)
            mc3.metric("🤝 Other", oth)
            mc4.metric("📊 Total Matches", tot)

            pie_data = pd.DataFrame({
                "Outcome": [h2h_team1, h2h_team2, "No Result/Tie"],
                "Count": [t1w, t2w, oth],
            })
            st.plotly_chart(_pie(pie_data, "Outcome", "Count",
                                 title=f"{h2h_team1} vs {h2h_team2} — Head to Head"), use_container_width=True)

        # ── 3. Team Chasing vs Defending Success ───────────────────────────
        st.subheader("🎯 Batting First vs Chasing Success Rate by Team")
        bat_first_sql = f"""
        SELECT i.batting_team as team,
               COUNT(*) as total,
               SUM(CASE WHEN m.winner = i.batting_team THEN 1 ELSE 0 END) as wins,
               ROUND(100.0 * SUM(CASE WHEN m.winner = i.batting_team THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct,
               i.innings_number
        FROM main_gold.fact_innings i
        JOIN main_gold.fact_matches m ON i.match_id = m.match_id
        WHERE YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to}
          AND i.innings_number IN (1, 2)
        GROUP BY i.batting_team, i.innings_number
        HAVING total >= 10
        ORDER BY innings_number, win_pct DESC
        """
        bf_df = _q(bat_first_sql)
        if not bf_df.empty:
            bf_df["Role"] = bf_df["innings_number"].map({1: "Bat First (Defend)", 2: "Chase"})
            top_teams = bf_df.groupby("team")["total"].sum().nlargest(15).index
            bf_df = bf_df[bf_df["team"].isin(top_teams)]
            fig_bf = px.bar(bf_df, x="team", y="win_pct", color="Role",
                            barmode="group", title="Win Rate: Batting First vs Chasing (Top 15 Teams)",
                            template=DARK, color_discrete_map={"Bat First (Defend)": "#38bdf8", "Chase": "#f59e0b"})
            fig_bf.update_layout(height=380, xaxis_tickangle=-40)
            st.plotly_chart(fig_bf, use_container_width=True)

        # ── 4. Team Batting vs Bowling Strength ────────────────────────────
        st.subheader("⚖️ Team Batting Strength vs Bowling Strength")
        bbs_sql = f"""
        SELECT i.batting_team as team,
               ROUND(AVG(i.total_runs), 1) as avg_runs_scored,
               ROUND(AVG(i2.total_runs), 1) as avg_runs_conceded
        FROM main_gold.fact_innings i
        JOIN main_gold.fact_matches m ON i.match_id = m.match_id
        JOIN main_gold.fact_innings i2 ON m.match_id = i2.match_id
            AND i2.innings_number != i.innings_number
        WHERE YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to}
          AND i.innings_number = 1
        GROUP BY i.batting_team
        HAVING COUNT(*) >= 8
        ORDER BY avg_runs_scored DESC
        LIMIT 20
        """
        bbs_df = _q(bbs_sql)
        fig_bbs = _scatter(bbs_df, "avg_runs_scored", "avg_runs_conceded",
                           text="team", color="avg_runs_scored",
                           title="Batting Strength (X) vs Bowling Weakness (Y) — Bigger gap = better",
                           labels={"avg_runs_scored": "Avg Runs Scored (Innings 1)",
                                   "avg_runs_conceded": "Avg Runs Conceded (Innings 2)"},
                           color_continuous_scale="Blues_r")
        # Add quadrant lines
        if not bbs_df.empty:
            mid_x = bbs_df["avg_runs_scored"].mean()
            mid_y = bbs_df["avg_runs_conceded"].mean()
            fig_bbs.add_hline(y=mid_y, line_dash="dot", line_color="#475569",
                              annotation_text="Avg Conceded", annotation_position="bottom right")
            fig_bbs.add_vline(x=mid_x, line_dash="dot", line_color="#475569",
                              annotation_text="Avg Scored", annotation_position="top left")
        st.plotly_chart(fig_bbs, use_container_width=True)

        # ── 5. Toss Impact by Team ─────────────────────────────────────────
        st.subheader("🪙 Toss Win Impact on Match Outcome")
        toss_sql = f"""
        SELECT
            team_1 as team,
            toss_decision,
            COUNT(*) as matches,
            SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) as toss_then_won,
            ROUND(100.0 * SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) / COUNT(*), 1) as toss_win_pct
        FROM main_gold.fact_matches
        WHERE YEAR(match_date) BETWEEN {yr_from} AND {yr_to}
          {v_filter}
        GROUP BY team_1, toss_decision
        HAVING matches >= 5
        ORDER BY toss_win_pct DESC
        LIMIT 30
        """
        toss_df = _q(toss_sql)
        fig_toss = _bar(toss_df, "team", "toss_win_pct", color="toss_decision",
                        title="Win % After Winning Toss (by Decision & Team)",
                        labels={"toss_win_pct": "Win % After Toss Win", "toss_decision": "Decision"},
                        color_discrete_map={"bat": "#38bdf8", "field": "#f59e0b"}, barmode="group")
        st.plotly_chart(fig_toss, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PLAYER INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

def _section_player(f: dict, st_tab):
    with st_tab:
        st.markdown("<div class='section-header'>👤 Player Intelligence</div>", unsafe_allow_html=True)
        yr_from, yr_to = f["year_from"], f["year_to"]
        v_filter = f"AND m.venue = '{f['venue']}'" if f["venue"] else ""
        t_filter = f"AND m.toss_winner = '{f['team']}'" if f["team"] else ""

        # ── Batting Leaderboard ────────────────────────────────────────────
        st.subheader("🏆 Batting Leaderboard — Strike Rate vs Average")
        min_balls = st.slider("Min Balls Faced (filter for quality batters)", 100, 1000, 200, step=50, key="p_min_balls")
        bat_ldr_sql = f"""
        SELECT d.batter,
               SUM(d.runs_batter) as total_runs,
               SUM(d.is_legal_ball) as balls,
               COUNT(DISTINCT d.match_id) as matches,
               ROUND(SUM(d.runs_batter)::FLOAT / NULLIF(SUM(d.is_legal_ball), 0) * 100, 1) as strike_rate,
               ROUND(SUM(d.runs_batter)::FLOAT / NULLIF(COUNT(DISTINCT d.match_id), 0), 1) as avg_per_match
        FROM main_gold.fact_deliveries d
        JOIN main_gold.fact_matches m ON d.match_id = m.match_id
        WHERE YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to} {v_filter} {t_filter}
        GROUP BY d.batter
        HAVING balls >= {min_balls}
        ORDER BY total_runs DESC
        LIMIT 30
        """
        bat_ldr_df = _q(bat_ldr_sql)
        fig_bat = _scatter(bat_ldr_df, "avg_per_match", "strike_rate",
                           color="total_runs", size="matches",
                           text="batter", title="Batters: Avg per Match (X) vs Strike Rate (Y)",
                           labels={"avg_per_match": "Avg Runs / Match", "strike_rate": "Strike Rate"},
                           color_continuous_scale="Viridis")
        st.plotly_chart(fig_bat, use_container_width=True)

        # ── Top Run Scorers Bar ────────────────────────────────────────────
        st.subheader("📊 Top 15 Run Scorers Leaderboard")
        if not bat_ldr_df.empty:
            top15 = bat_ldr_df.nlargest(15, "total_runs")
            fig_top15 = _bar(top15, "batter", "total_runs", color="strike_rate",
                             title="Top 15 Run Scorers",
                             labels={"total_runs": "Total Runs", "strike_rate": "Strike Rate"},
                             color_continuous_scale="Blues")
            st.plotly_chart(fig_top15, use_container_width=True)

        # ── Player Runs Over Time ──────────────────────────────────────────
        st.subheader("📈 Player Run Scoring Trend Over Years")
        all_batters = bat_ldr_df["batter"].tolist() if not bat_ldr_df.empty else []
        if all_batters:
            sel_player = st.selectbox("Select Batter", all_batters, key="p_trend_player")
            player_trend_sql = f"""
            SELECT YEAR(m.match_date) as year,
                   SUM(d.runs_batter) as runs,
                   COUNT(DISTINCT d.match_id) as matches,
                   ROUND(SUM(d.runs_batter)::FLOAT / NULLIF(COUNT(DISTINCT d.match_id),0), 1) as avg
            FROM main_gold.fact_deliveries d
            JOIN main_gold.fact_matches m ON d.match_id = m.match_id
            WHERE d.batter = '{sel_player}'
              AND YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to}
            GROUP BY 1 ORDER BY 1
            """
            pt_df = _q(player_trend_sql)
            if not pt_df.empty:
                fig_pt = make_subplots(specs=[[{"secondary_y": True}]])
                fig_pt.add_trace(go.Bar(x=pt_df["year"], y=pt_df["runs"], name="Total Runs",
                                        marker_color="#38bdf8"), secondary_y=False)
                fig_pt.add_trace(go.Scatter(x=pt_df["year"], y=pt_df["avg"], name="Avg/Match",
                                            line=dict(color="#f59e0b", width=3), mode="lines+markers"), secondary_y=True)
                fig_pt.update_layout(title=f"{sel_player} — Annual Runs & Average",
                                     template=DARK, height=360)
                fig_pt.update_yaxes(title_text="Total Runs", secondary_y=False)
                fig_pt.update_yaxes(title_text="Average / Match", secondary_y=True)
                st.plotly_chart(fig_pt, use_container_width=True)

        # ── Bowling Leaderboard ────────────────────────────────────────────
        st.subheader("🎳 Bowling Intelligence — Economy vs Wickets")
        min_overs = st.slider("Min Overs Bowled", 20, 500, 60, step=20, key="p_min_overs")
        bowl_sql = f"""
        SELECT d.bowler,
               SUM(d.is_wicket) as wickets,
               ROUND(SUM(d.is_legal_ball)::FLOAT / 6, 1) as overs,
               ROUND(SUM(d.runs_batter + d.runs_extras)::FLOAT / NULLIF(SUM(d.is_legal_ball),0) * 6, 2) as economy,
               ROUND(SUM(d.is_legal_ball)::FLOAT / NULLIF(SUM(d.is_wicket),0), 1) as bowling_sr,
               COUNT(DISTINCT d.match_id) as matches
        FROM main_gold.fact_deliveries d
        JOIN main_gold.fact_matches m ON d.match_id = m.match_id
        WHERE YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to} {v_filter} {t_filter}
        GROUP BY d.bowler
        HAVING overs >= {min_overs}
        ORDER BY wickets DESC LIMIT 25
        """
        bowl_df = _q(bowl_sql)
        fig_bowl = _scatter(bowl_df, "economy", "wickets", color="bowling_sr",
                            size="matches", text="bowler",
                            title="Bowlers: Economy (X) vs Wickets (Y) — ideal = low economy, high wickets",
                            labels={"economy": "Economy Rate", "wickets": "Total Wickets"},
                            color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig_bowl, use_container_width=True)

        # ── Player Performance by Venue Heatmap ───────────────────────────
        st.subheader("🗺️ Batter Performance by Venue (Top Runs)")
        pvh_sql = f"""
        SELECT d.batter, m.venue,
               ROUND(SUM(d.runs_batter)::FLOAT / NULLIF(COUNT(DISTINCT d.match_id),0), 1) as avg_runs
        FROM main_gold.fact_deliveries d
        JOIN main_gold.fact_matches m ON d.match_id = m.match_id
        WHERE YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to}
        GROUP BY d.batter, m.venue
        HAVING COUNT(DISTINCT d.match_id) >= 3
        """
        pvh_df = _q(pvh_sql)
        if not pvh_df.empty and not bowl_df.empty:
            top_batters_list = bat_ldr_df["batter"].head(10).tolist() if not bat_ldr_df.empty else []
            top_venues_list = _venues()[:15]
            pvh_filtered = pvh_df[pvh_df["batter"].isin(top_batters_list) & pvh_df["venue"].isin(top_venues_list)]
            if not pvh_filtered.empty:
                pivot = pvh_filtered.pivot_table(index="batter", columns="venue", values="avg_runs")
                fig_pvh = px.imshow(pivot, title="Avg Runs per Match by Batter × Venue",
                                    color_continuous_scale="YlOrRd", template=DARK, aspect="auto")
                fig_pvh.update_layout(height=420)
                st.plotly_chart(fig_pvh, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MATCH INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

def _section_match(f: dict, st_tab):
    with st_tab:
        st.markdown("<div class='section-header'>📋 Match Intelligence</div>", unsafe_allow_html=True)
        yr_from, yr_to = f["year_from"], f["year_to"]
        t_filter = f"AND m.toss_winner = '{f['team']}'" if f["team"] else ""
        v_filter = f"AND m.venue = '{f['venue']}'" if f["venue"] else ""

        # ── 1. Run Distribution by Over ────────────────────────────────────
        st.subheader("🏏 Run Rate Heatmap by Over Number")
        ov_sql = f"""
        SELECT d.over_number,
               ROUND(SUM(d.runs_batter + d.runs_extras)::FLOAT / NULLIF(COUNT(DISTINCT d.match_id), 0), 2) as avg_runs,
               ROUND(SUM(d.is_wicket)::FLOAT / NULLIF(COUNT(DISTINCT d.match_id), 0), 3) as wkts_per_match
        FROM main_gold.fact_deliveries d
        JOIN main_gold.fact_matches m ON d.match_id = m.match_id
        WHERE YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to} {t_filter} {v_filter}
        GROUP BY d.over_number ORDER BY 1
        """
        ov_df = _q(ov_sql)
        if not ov_df.empty:
            fig_ov = make_subplots(specs=[[{"secondary_y": True}]])
            phase_colors = ["#1d4ed8"] * 6 + ["#0891b2"] * 9 + ["#dc2626"] * 5
            fig_ov.add_trace(go.Bar(x=ov_df["over_number"], y=ov_df["avg_runs"],
                                    name="Avg Runs/Match", marker_color=phase_colors[:len(ov_df)]),
                             secondary_y=False)
            fig_ov.add_trace(go.Scatter(x=ov_df["over_number"], y=ov_df["wkts_per_match"],
                                        name="Wickets/Match", line=dict(color="#fbbf24", width=2.5),
                                        mode="lines+markers"), secondary_y=True)
            fig_ov.update_layout(title="Run Distribution & Wickets by Over (Blue=PP, Teal=Middle, Red=Death)",
                                 template=DARK, height=380)
            fig_ov.add_vrect(x0=-0.5, x1=5.5, fillcolor="#1d4ed8", opacity=0.06, line_width=0)
            fig_ov.add_vrect(x0=5.5, x1=14.5, fillcolor="#0891b2", opacity=0.06, line_width=0)
            fig_ov.add_vrect(x0=14.5, x1=19.5, fillcolor="#dc2626", opacity=0.06, line_width=0)
            st.plotly_chart(fig_ov, use_container_width=True)

        # ── 2. Score Distribution Histogram ───────────────────────────────
        st.subheader("📊 First Innings Score Distribution")
        hist_sql = f"""
        SELECT i.total_runs
        FROM main_gold.fact_innings i
        JOIN main_gold.fact_matches m ON i.match_id = m.match_id
        WHERE i.innings_number = 1
          AND YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to} {v_filter}
          AND i.total_runs > 50
        """
        hist_df = _q(hist_sql)
        if not hist_df.empty:
            fig_hist = px.histogram(hist_df, x="total_runs", nbins=30,
                                    title="Distribution of 1st Innings Scores",
                                    template=DARK, color_discrete_sequence=["#38bdf8"])
            fig_hist.add_vline(x=hist_df["total_runs"].mean(), line_dash="dash",
                               line_color="#f59e0b", annotation_text=f"Mean: {hist_df['total_runs'].mean():.0f}")
            fig_hist.update_layout(height=340, bargap=0.02)
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── 3. Powerplay vs Final Score Correlation ────────────────────────
        st.subheader("⚡ Powerplay Score vs Final Score Correlation")
        pp_sql = f"""
        SELECT
            pp_runs.match_id,
            pp_runs.pp_runs,
            i.total_runs as final_runs,
            m.team_1,
            CASE WHEN i.batting_team = m.winner THEN 'Won' ELSE 'Lost' END as outcome
        FROM (
            SELECT match_id, SUM(runs_batter + runs_extras) as pp_runs
            FROM main_gold.fact_deliveries
            WHERE over_number < 6 AND innings_number = 1
            GROUP BY match_id
        ) pp_runs
        JOIN main_gold.fact_innings i ON i.match_id = pp_runs.match_id AND i.innings_number = 1
        JOIN main_gold.fact_matches m ON m.match_id = pp_runs.match_id
        WHERE YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to} {v_filter}
        """
        pp_df = _q(pp_sql)
        if not pp_df.empty:
            fig_pp = px.scatter(pp_df, x="pp_runs", y="final_runs", color="outcome",
                                title="Powerplay Score vs Final Score (1st Innings)",
                                template=DARK, trendline="ols",
                                color_discrete_map={"Won": "#22c55e", "Lost": "#ef4444"},
                                labels={"pp_runs": "Powerplay Runs (0-5)", "final_runs": "Final Score"})
            fig_pp.update_layout(height=360)
            st.plotly_chart(fig_pp, use_container_width=True)

        # ── 4. Match Outcome by Margin Type ───────────────────────────────
        st.subheader("🏆 Match Outcomes — Win by Runs vs Win by Wickets")
        margin_sql = f"""
        SELECT result_type,
               ROUND(AVG(result_margin), 1) as avg_margin,
               COUNT(*) as matches
        FROM main_gold.fact_matches
        WHERE result_type IN ('runs', 'wickets')
          AND YEAR(match_date) BETWEEN {yr_from} AND {yr_to} {v_filter}
        GROUP BY result_type
        """
        outcome_sql = f"""
        SELECT result_type, COUNT(*) as count
        FROM main_gold.fact_matches
        WHERE YEAR(match_date) BETWEEN {yr_from} AND {yr_to} {v_filter}
        GROUP BY result_type
        """
        outcome_df = _q(outcome_sql)
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.plotly_chart(_pie(outcome_df, "result_type", "count",
                                 title="Match Result Types"), use_container_width=True)
        with mcol2:
            margin_df = _q(margin_sql)
            st.plotly_chart(_bar(margin_df, "result_type", "avg_margin",
                                 title="Average Win Margin by Result Type",
                                 color="result_type",
                                 color_discrete_map={"runs": "#38bdf8", "wickets": "#22c55e"}),
                            use_container_width=True)

        # ── 5. Momentum Chart (run race) ──────────────────────────────────
        st.subheader("📈 Match Momentum — Run Scoring by Over (Choose a Match)")
        momentum_sql = f"""
        SELECT m.match_id, m.match_date, m.team_1, m.winner, m.event_name
        FROM main_gold.fact_matches m
        WHERE YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to} {t_filter} {v_filter}
        ORDER BY m.match_date DESC LIMIT 50
        """
        match_list_df = _q(momentum_sql)
        if not match_list_df.empty:
            match_list_df["label"] = match_list_df.apply(
                lambda r: f"{str(r['match_date'])[:10]} — {r['team_1']} ({r['event_name'] or 'T20I'})", axis=1)
            sel_match = st.selectbox("Select Match", match_list_df["label"].tolist(), key="mo_match")
            sel_match_id = match_list_df[match_list_df["label"] == sel_match]["match_id"].iloc[0]

            run_race_sql = f"""
            SELECT innings_number, over_number,
                   SUM(runs_batter + runs_extras) as over_runs,
                   SUM(SUM(runs_batter + runs_extras)) OVER (
                       PARTITION BY innings_number ORDER BY over_number
                   ) as cumulative_runs
            FROM main_gold.fact_deliveries
            WHERE match_id = '{sel_match_id}'
            GROUP BY innings_number, over_number ORDER BY 1, 2
            """
            rr_df = _q(run_race_sql)
            if not rr_df.empty:
                rr_df["Innings"] = rr_df["innings_number"].map({1: "1st Innings", 2: "2nd Innings"})
                fig_mo = px.line(rr_df, x="over_number", y="cumulative_runs", color="Innings",
                                 title="Match Momentum — Cumulative Runs Over Progress",
                                 template=DARK, markers=True,
                                 color_discrete_map={"1st Innings": "#38bdf8", "2nd Innings": "#f59e0b"},
                                 labels={"over_number": "Over", "cumulative_runs": "Cumulative Runs"})
                fig_mo.add_vrect(x0=0, x1=5, fillcolor="#1d4ed8", opacity=0.08, line_width=0, annotation_text="PP")
                fig_mo.add_vrect(x0=15, x1=19, fillcolor="#dc2626", opacity=0.08, line_width=0, annotation_text="Death")
                fig_mo.update_layout(height=360)
                st.plotly_chart(fig_mo, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — VENUE INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

def _section_venue(f: dict, st_tab):
    with st_tab:
        st.markdown("<div class='section-header'>🏟️ Venue Intelligence</div>", unsafe_allow_html=True)
        yr_from, yr_to = f["year_from"], f["year_to"]
        t_filter = f"AND m.toss_winner = '{f['team']}'" if f["team"] else ""

        # ── 1. Average 1st Innings Score by Venue ─────────────────────────
        st.subheader("📊 Average 1st Innings Score by Venue")
        v_top = st.slider("Top N Venues", 10, 40, 20, key="v_n")
        avg_score_sql = f"""
        SELECT m.venue,
               ROUND(AVG(i.total_runs), 1) as avg_score,
               ROUND(AVG(i2.total_runs), 1) as avg_chase,
               COUNT(*) as matches
        FROM main_gold.fact_innings i
        JOIN main_gold.fact_matches m ON i.match_id = m.match_id
        JOIN main_gold.fact_innings i2 ON m.match_id = i2.match_id AND i2.innings_number = 2
        WHERE i.innings_number = 1
          AND YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to} {t_filter}
        GROUP BY m.venue HAVING matches >= 3
        ORDER BY avg_score DESC LIMIT {v_top}
        """
        avg_score_df = _q(avg_score_sql)
        if not avg_score_df.empty:
            avg_score_long = pd.melt(avg_score_df, id_vars=["venue", "matches"],
                                     value_vars=["avg_score", "avg_chase"],
                                     var_name="Innings", value_name="Avg Runs")
            avg_score_long["Innings"] = avg_score_long["Innings"].map({"avg_score": "1st Innings", "avg_chase": "2nd Innings"})
            fig_vs = px.bar(avg_score_long, x="venue", y="Avg Runs", color="Innings",
                            barmode="group", title="Avg 1st vs 2nd Innings Score by Venue",
                            template=DARK,
                            color_discrete_map={"1st Innings": "#38bdf8", "2nd Innings": "#f59e0b"})
            fig_vs.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_vs, use_container_width=True)

        # ── 2. Chase Success Rate by Venue ─────────────────────────────────
        st.subheader("✅ Chase Success Rate by Venue")
        chase_sql = f"""
        SELECT m.venue,
               COUNT(*) as matches,
               ROUND(100.0 * SUM(CASE WHEN i2.batting_team = m.winner THEN 1 ELSE 0 END) / COUNT(*), 1) as chase_win_pct
        FROM main_gold.fact_matches m
        JOIN main_gold.fact_innings i2 ON m.match_id = i2.match_id AND i2.innings_number = 2
        WHERE YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to}
          AND m.result_type IN ('runs', 'wickets')
        GROUP BY m.venue HAVING matches >= 3
        ORDER BY chase_win_pct DESC LIMIT {v_top}
        """
        chase_df = _q(chase_sql)
        if not chase_df.empty:
            chase_df["bat_first_win_pct"] = 100 - chase_df["chase_win_pct"]
            chase_long = pd.melt(chase_df, id_vars=["venue"], value_vars=["chase_win_pct", "bat_first_win_pct"],
                                 var_name="Strategy", value_name="Win %")
            chase_long["Strategy"] = chase_long["Strategy"].map({
                "chase_win_pct": "Chase", "bat_first_win_pct": "Bat First"})
            fig_ch = px.bar(chase_long, x="venue", y="Win %", color="Strategy",
                            title="Chase vs Bat-First Win Rate by Venue",
                            barmode="stack", template=DARK,
                            color_discrete_map={"Chase": "#22c55e", "Bat First": "#38bdf8"})
            fig_ch.add_hline(y=50, line_dash="dot", line_color="#ef4444",
                             annotation_text="50% threshold")
            fig_ch.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_ch, use_container_width=True)

        # ── 3. Team vs Venue Win Heatmap ───────────────────────────────────
        st.subheader("🗺️ Team × Venue Win Rate Heatmap")
        hmv_sql = f"""
        SELECT t.team, m.venue,
               ROUND(100.0 * AVG(CASE WHEN m.winner = t.team THEN 1.0 ELSE 0.0 END), 0) as win_pct,
               COUNT(m.match_id) as matches
        FROM main_gold.fact_matches m
        JOIN main_silver.slv_match_teams t ON m.match_id = t.match_id
        WHERE YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to}
        GROUP BY t.team, m.venue HAVING matches >= 4
        """
        hmv_df = _q(hmv_sql)
        if not hmv_df.empty:
            top_teams_hm = hmv_df.groupby("team")["matches"].sum().nlargest(12).index.tolist()
            top_venues_hm = hmv_df.groupby("venue")["matches"].sum().nlargest(15).index.tolist()
            hmv_filtered = hmv_df[hmv_df["team"].isin(top_teams_hm) & hmv_df["venue"].isin(top_venues_hm)]
            if not hmv_filtered.empty:
                pivot = hmv_filtered.pivot_table(index="team", columns="venue", values="win_pct")
                fig_hmv = px.imshow(pivot, title="Win % Heatmap: Team × Venue (min 4 matches)",
                                    color_continuous_scale="RdYlGn", template=DARK, aspect="auto",
                                    text_auto=".0f")
                fig_hmv.update_layout(height=440)
                st.plotly_chart(fig_hmv, use_container_width=True)

        # ── 4. Venue Personality: High/Low Scoring ─────────────────────────
        st.subheader("🎭 Venue Character — Pitch & Conditions Analysis")
        char_sql = f"""
        SELECT m.venue,
               ROUND(AVG(i.total_runs), 0) as avg_runs,
               ROUND(100.0 * SUM(CASE WHEN i.batting_team = m.winner THEN 1 ELSE 0 END) / COUNT(*), 0) as batting_wins_pct,
               COUNT(*) as matches
        FROM main_gold.fact_innings i
        JOIN main_gold.fact_matches m ON i.match_id = m.match_id
        WHERE i.innings_number = 1
          AND YEAR(m.match_date) BETWEEN {yr_from} AND {yr_to}
        GROUP BY m.venue HAVING matches >= 5
        ORDER BY avg_runs DESC LIMIT 20
        """
        char_df = _q(char_sql)
        if not char_df.empty:
            fig_char = px.scatter(char_df, x="avg_runs", y="batting_wins_pct",
                                  size="matches", text="venue",
                                  title="Venue Character: High Score (X) vs Bat-First Win Rate (Y)",
                                  template=DARK, color="avg_runs", color_continuous_scale="RdYlGn_r",
                                  labels={"avg_runs": "Avg 1st Innings Score",
                                          "batting_wins_pct": "Batting First Win %"})
            fig_char.update_traces(textposition="top center")
            fig_char.add_hline(y=50, line_dash="dot", line_color="#94a3b8")
            fig_char.update_layout(height=440)
            st.plotly_chart(fig_char, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render():
    st.title("📈 Advanced Analytics")
    st.markdown("""<div class='metric-card'>
    Deep-dive sports analytics across Teams, Players, Matches, and Venues.
    Use the <b>Global Filters</b> in the sidebar to dynamically update all charts.
    </div>""", unsafe_allow_html=True)

    # Load global filter values
    f = _global_filters()

    # Filter summary banner
    active_filters = []
    if f["team"]: active_filters.append(f"🏏 Team: **{f['team']}**")
    if f["opponent"]: active_filters.append(f"⚔️ Opponent: **{f['opponent']}**")
    if f["venue"]: active_filters.append(f"🏟️ Venue: **{f['venue']}**")
    active_filters.append(f"📅 {f['year_from']}–{f['year_to']}")
    st.info("  |  ".join(active_filters))

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏏 Team Intelligence",
        "👤 Player Intelligence",
        "📋 Match Intelligence",
        "🏟️ Venue Intelligence",
    ])

    _section_team(f, tab1)
    _section_player(f, tab2)
    _section_match(f, tab3)
    _section_venue(f, tab4)
