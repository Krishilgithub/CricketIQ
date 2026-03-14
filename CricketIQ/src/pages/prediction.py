"""Match Prediction page — Advanced interactive gauge charts and confidence meters."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.pages.shared import (
    get_teams, get_venues, get_h2h_rate, get_venue_avg,
    get_team_form, load_model, get_hub_con,
)


def _gauge(value: float, title: str, max_val: float = 100, suffix: str = "%") -> go.Figure:
    """Create a Plotly gauge chart for a single metric."""
    color = "#22c55e" if value >= 55 else ("#f59e0b" if value >= 45 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        number={"suffix": suffix, "font": {"size": 30, "color": "#f8fafc"}},
        title={"text": title, "font": {"size": 14, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#334155"},
            "bar": {"color": color},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 40], "color": "#1f2937"},
                {"range": [40, 60], "color": "#1e293b"},
                {"range": [60, max_val], "color": "#172554"},
            ],
            "threshold": {
                "line": {"color": "#38bdf8", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="#0f172a", font_color="#f8fafc",
    )
    return fig


def _get_team_additional_stats(team: str) -> dict:
    """Fetch extra stats for display."""
    con = get_hub_con()
    try:
        q = f"""
        SELECT
            COUNT(*) as total_matches,
            SUM(CASE WHEN winner = '{team}' THEN 1 ELSE 0 END) as wins,
            ROUND(AVG(CASE WHEN winner = '{team}' THEN 1.0 ELSE 0.0 END) * 100, 1) as win_pct
        FROM main_gold.fact_matches
        WHERE toss_winner = '{team}' OR winner = '{team}'
        """
        row = con.execute(q).df().iloc[0]
        return {"total": int(row["total_matches"]), "wins": int(row["wins"]), "win_pct": float(row["win_pct"])}
    except Exception:
        return {"total": 0, "wins": 0, "win_pct": 0.0}


def _recent_form_chart(team: str) -> go.Figure | None:
    """Bar chart of team's last 10-match W/L sequence."""
    con = get_hub_con()
    try:
        q = f"""
        SELECT match_date, team_1, winner,
               CASE WHEN winner = '{team}' THEN 1 ELSE 0 END as won
        FROM main_gold.fact_matches
        WHERE toss_winner = '{team}' OR winner = '{team}'
        ORDER BY match_date DESC LIMIT 10
        """
        df = con.execute(q).df()
        if df.empty:
            return None
        df = df.iloc[::-1].reset_index(drop=True)
        df["result"] = df["won"].map({1: "Win", 0: "Loss"})
        color_map = {"Win": "#22c55e", "Loss": "#ef4444"}
        fig = px.bar(df, x=df.index.astype(str), y="won", color="result",
                     color_discrete_map=color_map,
                     title=f"{team} — Last 10 Match Form",
                     template="plotly_dark",
                     labels={"won": "Result", "x": "Match"})
        fig.update_layout(height=220, showlegend=True, margin=dict(l=10, r=10, t=50, b=10))
        return fig
    except Exception:
        return None


def render():
    st.title("🔮 Match Prediction")
    st.markdown("""<div class='metric-card'>
    Advanced win probability prediction powered by our ML champion model.
    Configure the match details below for an in-depth pre-match analysis.
    </div>""", unsafe_allow_html=True)

    teams_list = get_teams()
    venues_list = get_venues()
    champion = load_model()

    if not champion:
        st.warning("⚠️ No champion model found. Please run `python -m src.models.train_models` to train the model first.")
        return

    # ── Match Configuration ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>Match Configuration</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        team1 = st.selectbox("🏏 Team 1 (Toss Winner)", teams_list,
                             index=teams_list.index("India") if "India" in teams_list else 0)
    with c2:
        team2 = st.selectbox("🏏 Team 2", [t for t in teams_list if t != team1])
    with c3:
        venue = st.selectbox("🏟️ Match Venue", venues_list)

    col_td, col_btn = st.columns([2, 1])
    with col_td:
        toss_decision = st.radio("🪙 Toss Decision", ["Bat", "Field"], horizontal=True)
    with col_btn:
        st.write("")
        st.write("")
        predict_clicked = st.button("🔮 Run Full Analysis", type="primary", use_container_width=True)

    if not predict_clicked:
        st.info("👆 Configure the match above and click **Run Full Analysis** to see predictions.")
        return

    # ── Compute Predictions ───────────────────────────────────────────────
    with st.spinner("🧠 Running ML model..."):
        h2h_rate = get_h2h_rate(team1)
        venue_avg = get_venue_avg(venue)
        team1_form = get_team_form(team1)
        team1_stats = _get_team_additional_stats(team1)
        team2_stats = _get_team_additional_stats(team2)

        feats = pd.DataFrame([{
            "toss_bat": 1 if toss_decision == "Bat" else 0,
            "venue_avg_1st_inns_runs": venue_avg,
            "team_1_h2h_win_rate": h2h_rate,
            "team_1_form_last5": team1_form,
            "team_2_form_last5": 1 - team1_form,
        }])
        try:
            win_prob_t1 = float(champion["model"].predict_proba(feats)[0][1])
        except Exception:
            win_prob_t1 = h2h_rate  # Fallback

        win_prob_t2 = 1 - win_prob_t1

    # ── Match Verdict Banner ──────────────────────────────────────────────
    st.markdown("---")
    favourite = team1 if win_prob_t1 >= win_prob_t2 else team2
    fav_prob = max(win_prob_t1, win_prob_t2) * 100
    confidence = "High" if fav_prob >= 62 else ("Moderate" if fav_prob >= 54 else "Low")
    conf_color = "#22c55e" if confidence == "High" else ("#f59e0b" if confidence == "Moderate" else "#ef4444")

    st.markdown(f"""
    <div class='prediction-card' style='text-align:center; margin-bottom:24px;'>
        <div style='font-size:1rem; color:#94a3b8; margin-bottom:6px;'>🏆 Predicted Winner</div>
        <div style='font-size:2.5rem; font-weight:800; color:#38bdf8;'>{favourite}</div>
        <div style='font-size:1.1rem; color:#f8fafc; margin-top:6px;'>{fav_prob:.1f}% win probability</div>
        <div style='margin-top:8px; padding:4px 16px; border-radius:20px; display:inline-block;
                    background:{conf_color}22; border:1px solid {conf_color}; color:{conf_color};
                    font-size:0.85rem; font-weight:600;'>Confidence: {confidence}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Gauge Charts Row ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Win Probability Gauges</div>", unsafe_allow_html=True)
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(_gauge(win_prob_t1 * 100, f"{team1} Win %"), use_container_width=True)
    with g2:
        st.plotly_chart(_gauge(win_prob_t2 * 100, f"{team2} Win %"), use_container_width=True)
    with g3:
        st.plotly_chart(_gauge(h2h_rate * 100, f"H2H Rate ({team1})"), use_container_width=True)
    with g4:
        st.plotly_chart(_gauge(team1_form * 100, f"{team1} Last-5 Form"), use_container_width=True)

    # ── Probability Breakdown Bar ─────────────────────────────────────────
    st.markdown("---")
    fig_bar = go.Figure(go.Bar(
        x=[team1, team2],
        y=[win_prob_t1 * 100, win_prob_t2 * 100],
        marker_color=["#38bdf8", "#f43f5e"],
        text=[f"{win_prob_t1*100:.1f}%", f"{win_prob_t2*100:.1f}%"],
        textposition="auto",
        width=0.5,
    ))
    fig_bar.update_layout(
        title=f"Win Probability: {team1} vs {team2}",
        template="plotly_dark", height=360,
        yaxis=dict(title="Win Probability (%)", range=[0, 100]),
        bargap=0.4,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Team Comparison Stats ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>Team Comparison</div>", unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric(f"🏏 {team1} All-Time Win Rate", f"{team1_stats['win_pct']}%")
    tc2.metric("📊 Venue Avg 1st Innings", f"{venue_avg:.0f} runs")
    tc3.metric(f"🏏 {team2} All-Time Win Rate", f"{team2_stats['win_pct']}%")

    # ── Recent Form Charts ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Recent Form (Last 10 Matches)</div>", unsafe_allow_html=True)
    rf1, rf2 = st.columns(2)
    with rf1:
        fig_form1 = _recent_form_chart(team1)
        if fig_form1:
            st.plotly_chart(fig_form1, use_container_width=True)
    with rf2:
        fig_form2 = _recent_form_chart(team2)
        if fig_form2:
            st.plotly_chart(fig_form2, use_container_width=True)

    # ── Feature Breakdown ─────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🔍 Feature Input Breakdown", expanded=False):
        feat_data = {
            "Feature": ["Toss Decision", "Venue Avg 1st Innings", "H2H Win Rate (Team 1)", "Team 1 Last-5 Form", "Team 2 Last-5 Form"],
            "Value": [
                toss_decision,
                f"{venue_avg:.1f} runs",
                f"{h2h_rate*100:.1f}%",
                f"{team1_form*100:.1f}%",
                f"{(1-team1_form)*100:.1f}%",
            ]
        }
        st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)
