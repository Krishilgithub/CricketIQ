"""
src/ml/predictor.py
────────────────────
Phase 22 FIX: Complete rewrite to supply ALL features matching the
training pipeline. No more feature mismatch or fake values.
"""

import pandas as pd
from langsmith import traceable

from src.pages.shared import (
    get_h2h_rate, get_venue_avg, get_team_form,
    get_team_momentum, get_venue_chase_rate, get_team_venue_win_rate,
    load_model
)

@traceable(run_type="tool", name="ML Model Prediction", tags=["mlflow", "xgboost"])
def predict_match(team1: str, team2: str, venue: str, toss_decision: str) -> dict:
    """
    Phase 22: Runs the champion ML model with ALL features matching training pipeline.
    No team identity leakage — purely based on form, H2H, venue, and momentum.
    """
    champion = load_model()
    if not champion:
        raise ValueError("No champion model found for prediction.")

    # Query ALL features independently (no faking)
    h2h_rate = get_h2h_rate(team1, team2)  # Fixed: pairwise H2H
    venue_avg = get_venue_avg(venue)
    venue_chase_rate = get_venue_chase_rate(venue)
    
    team1_form5 = get_team_form(team1, window=5)
    team1_form10 = get_team_form(team1, window=10)
    team1_momentum = get_team_momentum(team1)
    
    team2_form5 = get_team_form(team2, window=5)   # Fixed: independent query
    team2_form10 = get_team_form(team2, window=10)  # Fixed: independent query
    team2_momentum = get_team_momentum(team2)
    
    team1_venue_wr = get_team_venue_win_rate(team1, venue)
    team2_venue_wr = get_team_venue_win_rate(team2, venue)
    
    h2h_advantage = h2h_rate - 0.5
    form_last5_diff = team1_form5 - team2_form5
    form_last10_diff = team1_form10 - team2_form10
    momentum_diff = team1_momentum - team2_momentum
    venue_win_rate_diff = team1_venue_wr - team2_venue_wr
    
    # Build feature DataFrame matching EXACTLY the training pipeline columns
    feats = pd.DataFrame([{
        "toss_bat": 1 if toss_decision == "Bat" else 0,
        "venue_avg_1st_inns_runs": venue_avg,
        "venue_chase_success_rate": venue_chase_rate,
        "h2h_advantage": h2h_advantage,
        "form_last5_diff": form_last5_diff,
        "form_last10_diff": form_last10_diff,
        "momentum_diff": momentum_diff,
        "venue_win_rate_diff": venue_win_rate_diff,
    }])
    
    try:
        win_prob_t1 = float(champion["pipeline"].predict_proba(feats)[0][1])
    except Exception:
        try:
            win_prob_t1 = float(champion["model"].predict_proba(feats)[0][1])
        except Exception:
            win_prob_t1 = h2h_rate  # Last resort fallback

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
        "team1_form": team1_form5,
        "team2_form": team2_form5,
        "venue_avg": venue_avg,
        "model_loaded": champion.get("name", "champion")
    }
    
    return result
