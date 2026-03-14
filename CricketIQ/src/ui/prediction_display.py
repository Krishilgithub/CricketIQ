"""
src/ui/prediction_display.py
────────────────────────────
Streamlit UI components for rendering prediction results in chat.
"""

import streamlit as st
import plotly.graph_objects as go

def render_prediction_result(pred: dict):
    """Renders the prediction dictionary beautifully in Streamlit."""
    
    team1 = pred.get("team1", "Team 1")
    team2 = pred.get("team2", "Team 2")
    venue = pred.get("venue", "Unknown Venue")
    toss = pred.get("toss_decision", "Bat")
    
    favourite = pred.get("favourite", "Unknown")
    fav_prob = pred.get("fav_prob", 50.0)
    confidence = pred.get("confidence", "Moderate")
    
    win_prob_t1 = pred.get("win_prob_t1", 0.5) * 100
    win_prob_t2 = pred.get("win_prob_t2", 0.5) * 100
    
    # Assumptions Note
    st.info(f"🏟️ **Assumptions:** Match at **{venue}**, {team1} wins toss & elects to **{toss}**.")
    
    # Main Header
    st.markdown(f"### 🎯 Prediction: {favourite} is favored to win")
    
    # CSS coloring for Confidence
    conf_color = "🟢" if confidence == "High" else ("🟡" if confidence == "Moderate" else "🔴")
    st.markdown(f"**Confidence Score:** {conf_color} {confidence}")
    
    st.markdown("---")
    
    cols = st.columns(2)
    with cols[0]:
        st.metric(f"{team1} Win Prob", f"{win_prob_t1:.1f}%")
        st.caption(f"Recent Form: {pred.get('team1_form', 0.5):.2f}")
    with cols[1]:
        st.metric(f"{team2} Win Prob", f"{win_prob_t2:.1f}%")
        st.caption(f"H2H Win Rate against {team2}: {pred.get('h2h_rate', 0.5):.2f}")

    # Gauge Chart for Probability
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=win_prob_t1,
        title={'text': f"{team1} Win Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 45], 'color': "rgba(239, 68, 68, 0.2)"},
                {'range': [45, 55], 'color': "rgba(245, 158, 11, 0.2)"},
                {'range': [55, 100], 'color': "rgba(34, 197, 94, 0.2)"}
            ],
            'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': 50}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary
    st.markdown(
        f"Based on historical data from **{pred.get('model_loaded', 'Champion Model')}**, "
        f"*{favourite}* currently holds a statistical advantage in these conditions."
    )
