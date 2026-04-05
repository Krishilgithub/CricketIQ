"""Intelligence Hub dashboard page — KPI overview + 3 persona tabs."""
import streamlit as st
import plotly.express as px

from src.pages.shared import (
    get_global_kpis, get_venues,
    get_phase_data, get_toss_recommendation,
    get_top_batters, get_top_bowlers, get_venue_heatmap,
    get_exciting_matches, get_highest_scores, get_best_bowling,
    db_available, show_db_unavailable_warning,
)


def render():
    st.title("📊 Intelligence Hub")
    st.markdown("""<div class='metric-card'>
    Explore pre-match predictions, venue behaviors, and strategic toss recommendations.
    </div>""", unsafe_allow_html=True)

    if not db_available():
        show_db_unavailable_warning()
        return

    # ── Global KPIs ─────────────────────────────────────────────────────────
    kpis = get_global_kpis()
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("🏟️ Total Matches", f"{int(kpis['total_matches']):,}")
    g2.metric("🏏 Total Runs", f"{int(kpis['total_runs']):,}")
    g3.metric("🎯 Total Wickets", f"{int(kpis['total_wickets']):,}")
    g4.metric("👤 Players Tracked", f"{int(kpis['total_players']):,}")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["👨‍💼 Coach View", "📊 Management", "📺 Fan / Media"])
    venues_list = get_venues()

    # ── Tab 1: Coach View ────────────────────────────────────────────────
    with tab1:
        st.markdown("<div class='section-header'>Phase Run Rates & Risk Analysis</div>", unsafe_allow_html=True)
        v_sel = st.selectbox("Analyze Venue", venues_list, key="coach_v")
        phase_df = get_phase_data(v_sel)

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            if not phase_df.empty:
                fig2 = px.bar(phase_df, x="phase", y="run_rate", title="Run Rate by Phase",
                              color="run_rate", color_continuous_scale="Blues", template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)
        with c_p2:
            if not phase_df.empty:
                fig3 = px.bar(phase_df, x="phase", y="wicket_rate", title="Wicket Rate by Phase",
                              color="wicket_rate", color_continuous_scale="Reds", template="plotly_dark")
                st.plotly_chart(fig3, use_container_width=True)

        toss_df = get_toss_recommendation(v_sel)
        if not toss_df.empty:
            toss_df["win_pct"] = (toss_df["toss_wins"] / toss_df["total"] * 100).round(1)
            best = toss_df.loc[toss_df["win_pct"].idxmax()]
            st.success(f"🪙 **Toss Recommendation**: Choose to **{best['toss_decision'].upper()}** — {best['win_pct']:.0f}% win conversion at this venue.")
            st.dataframe(toss_df[["toss_decision", "total", "toss_wins", "win_pct"]].rename(columns={
                "toss_decision": "Decision", "total": "Matches", "toss_wins": "Wins", "win_pct": "Win %"
            }), use_container_width=True, hide_index=True)

    # ── Tab 2: Management ────────────────────────────────────────────────
    with tab2:
        st.markdown("<div class='section-header'>Talent & Form Analytics</div>", unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.subheader("🏏 Top Run Scorers (Last 2 Years)")
            bat_df = get_top_batters()
            fig4 = px.bar(bat_df, x="batter", y="runs", color="sr",
                          title="Runs (colored by Strike Rate)",
                          color_continuous_scale="Viridis", template="plotly_dark")
            fig4.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig4, use_container_width=True)

        with col_m2:
            st.subheader("🎯 Top Wicket Takers (Last 2 Years)")
            bowl_df = get_top_bowlers()
            fig5 = px.bar(bowl_df, x="bowler", y="wickets", color="econ",
                          title="Wickets (colored by Economy)",
                          color_continuous_scale="RdYlGn_r", template="plotly_dark")
            fig5.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig5, use_container_width=True)

        st.subheader("📍 Team vs Venue Win Heatmap")
        hm_df = get_venue_heatmap()
        if not hm_df.empty:
            top_teams = hm_df.groupby("team")["matches"].sum().nlargest(12).index.tolist()
            top_venues = hm_df.groupby("venue")["matches"].sum().nlargest(15).index.tolist()
            hm_pivot = hm_df[hm_df["team"].isin(top_teams) & hm_df["venue"].isin(top_venues)]\
                .pivot_table(index="team", columns="venue", values="win_pct")
            if not hm_pivot.empty:
                fig6 = px.imshow(hm_pivot, title="Win % by Team × Venue (min 5 matches)",
                                 color_continuous_scale="RdYlGn", template="plotly_dark", aspect="auto")
                st.plotly_chart(fig6, use_container_width=True)

    # ── Tab 3: Fan / Media ───────────────────────────────────────────────
    with tab3:
        st.markdown("<div class='section-header'>Entertainment Analytics</div>", unsafe_allow_html=True)

        st.subheader("🔥 Most Exciting T20I Matches (Close Chases)")
        ex_df = get_exciting_matches()
        if not ex_df.empty:
            ex_df["excitement_score"] = (200 - ex_df["margin_runs"]).clip(lower=0, upper=200)
            ex_df["label"] = ex_df["event_name"].fillna("T20I") + " " + ex_df["match_date"].astype(str).str[:4]
            fig7 = px.bar(ex_df, x="label", y="excitement_score",
                          color="excitement_score", color_continuous_scale="Inferno",
                          title="Top 20 Most Exciting Matches (lower margin = more exciting)",
                          template="plotly_dark")
            st.plotly_chart(fig7, use_container_width=True)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.subheader("💥 Highest Individual Scores")
            top_bat_df = get_highest_scores()
            st.dataframe(top_bat_df[["batter", "innings_runs", "event_name", "match_date"]].rename(columns={
                "batter": "Batter", "innings_runs": "Runs", "event_name": "Tournament", "match_date": "Date"
            }), use_container_width=True, hide_index=True)

        with col_f2:
            st.subheader("🎯 Best Bowling Spells")
            top_bowl_df = get_best_bowling()
            st.dataframe(top_bowl_df[["bowler", "wickets", "runs_conceded", "event_name", "match_date"]].rename(columns={
                "bowler": "Bowler", "wickets": "Wickets", "runs_conceded": "Runs", "event_name": "Tournament", "match_date": "Date"
            }), use_container_width=True, hide_index=True)
