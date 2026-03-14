"""
ML API — FastAPI endpoint for match outcome predictions (Docker serving).
"""

import sys
import pickle
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import ML_MODEL_DIR

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    print("⚠️ FastAPI not installed. pip install fastapi uvicorn")
    raise

app = FastAPI(
    title="🏏 ICC T20 WC Prediction API",
    description="Match outcome prediction using trained ML models",
    version="1.0.0",
)


class MatchPredictionRequest(BaseModel):
    team1_recent_win_rate: float = 0.5
    team2_recent_win_rate: float = 0.5
    toss_winner_is_team1: int = 1
    toss_elected_bat: int = 1
    is_world_cup: int = 0
    team1_avg_score: float = 155
    team2_avg_score: float = 155
    team1_avg_rr: float = 8.0
    team2_avg_rr: float = 8.0
    team1_avg_pp_rr: float = 7.5
    team2_avg_pp_rr: float = 7.5
    team1_avg_death_rr: float = 9.0
    team2_avg_death_rr: float = 9.0
    team1_avg_boundaries: float = 15
    team2_avg_boundaries: float = 15
    h2h_team1_win_rate: float = 0.5
    h2h_total_matches: float = 5
    venue_avg_1st_score: float = 155
    venue_avg_2nd_score: float = 145
    venue_matches: float = 10


class PredictionResponse(BaseModel):
    team1_win_probability: float
    team2_win_probability: float
    model_used: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_available": (ML_MODEL_DIR / "match_outcome_model.pkl").exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict_match(req: MatchPredictionRequest):
    model_path = ML_MODEL_DIR / "match_outcome_model.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    feature_values = {
        "toss_winner_is_team1": req.toss_winner_is_team1,
        "toss_elected_bat": req.toss_elected_bat,
        "is_world_cup": req.is_world_cup,
        "team1_recent_win_rate": req.team1_recent_win_rate,
        "team2_recent_win_rate": req.team2_recent_win_rate,
        "team1_avg_score": req.team1_avg_score,
        "team2_avg_score": req.team2_avg_score,
        "team1_avg_rr": req.team1_avg_rr,
        "team2_avg_rr": req.team2_avg_rr,
        "team1_avg_pp_rr": req.team1_avg_pp_rr,
        "team2_avg_pp_rr": req.team2_avg_pp_rr,
        "team1_avg_death_rr": req.team1_avg_death_rr,
        "team2_avg_death_rr": req.team2_avg_death_rr,
        "team1_avg_boundaries": req.team1_avg_boundaries,
        "team2_avg_boundaries": req.team2_avg_boundaries,
        "h2h_team1_win_rate": req.h2h_team1_win_rate,
        "h2h_total_matches": req.h2h_total_matches,
        "venue_avg_1st_score": req.venue_avg_1st_score,
        "venue_avg_2nd_score": req.venue_avg_2nd_score,
        "venue_matches": req.venue_matches,
        "win_rate_diff": req.team1_recent_win_rate - req.team2_recent_win_rate,
        "avg_score_diff": req.team1_avg_score - req.team2_avg_score,
        "avg_rr_diff": req.team1_avg_rr - req.team2_avg_rr,
    }

    X = np.array([[feature_values.get(c, 0) for c in saved["feature_cols"]]])

    if "Logistic" in saved["model_name"]:
        X = saved["scaler"].transform(X)

    prob = saved["model"].predict_proba(X)[0]

    return PredictionResponse(
        team1_win_probability=round(float(prob[1]), 4),
        team2_win_probability=round(float(prob[0]), 4),
        model_used=saved["model_name"],
    )
