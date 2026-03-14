"""
src/ml/predictor.py
────────────────────
ML prediction inference pipeline with LangSmith tracing.
"""

import pandas as pd
from langsmith import traceable

from src.pages.shared import get_h2h_rate, get_venue_avg, get_team_form, load_model

@traceable(run_type="tool", name="ML Model Prediction", tags=["mlflow", "xgboost"])
def predict_match(team1: str, team2: str, venue: str, toss_decision: str) -> dict:
    """
    Runs the champion ML model to predict match outcome.
    Logs inputs and model version to LangSmith.
    """
    champion = load_model()
    if not champion:
        raise ValueError("No champion model found for prediction.")

    h2h_rate = get_h2h_rate(team1)
    venue_avg = get_venue_avg(venue)
    team1_form = get_team_form(team1)
    
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
    
    # Calculate confidence
    fav_prob = max(win_prob_t1, win_prob_t2) * 100
    confidence = "High" if fav_prob >= 62 else ("Moderate" if fav_prob >= 54 else "Low")
    favourite = team1 if win_prob_t1 >= win_prob_t2 else team2
    
    result = {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_decision": toss_decision,
        "favourite": favourite,
        "fav_prob": fav_prob,
        "win_prob_t1": win_prob_t1,
        "win_prob_t2": win_prob_t2,
        "confidence": confidence,
        "h2h_rate": h2h_rate,
        "team1_form": team1_form,
        "venue_avg": venue_avg,
        "model_loaded": "champion"
    }
    
    return result
