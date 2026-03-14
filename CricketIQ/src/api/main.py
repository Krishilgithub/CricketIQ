"""
src/api/main.py
────────────────
CricketIQ FastAPI Application

REST Endpoints:
  GET  /health                          — health check
  POST /predict/prematch                — pre-match win probability
  GET  /predict/live                    — (stub) live ball-by-ball win prob update
  GET  /stats/team/{team_name}          — historical team statistics
  GET  /stats/player/{player_name}      — career batting/bowling stats
  GET  /stats/venue/{venue_name}        — venue scoring patterns

Run:
  uvicorn src.api.main:app --reload --port 8000
"""

import pickle
import os
from functools import lru_cache
from typing import Optional

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Bootstrap project path for module resolution ─────────────────────────────
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

# ── App Factory ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="CricketIQ API",
    description="🏏 Real-time cricket intelligence — match prediction, team/player/venue stats",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy Singletons ──────────────────────────────────────────────────────────
_db_con = None
_champion = None

def get_db():
    global _db_con
    if _db_con is None:
        cfg = get_config()
        path = str(resolve_path(cfg["paths"]["duckdb_path"]))
        _db_con = duckdb.connect(path, read_only=True)
    return _db_con

def get_champion():
    global _champion
    if _champion is None:
        cfg = get_config()
        model_path = str(resolve_path(cfg["paths"]["models_dir"])) + "/champion_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                _champion = pickle.load(f)
    return _champion


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class PreMatchRequest(BaseModel):
    team_1: str = Field(..., example="India")
    team_2: str = Field(..., example="Australia")
    venue: str = Field(..., example="Eden Gardens")
    toss_winner: str = Field(..., example="India")
    toss_decision: str = Field("bat", example="bat", description="'bat' or 'field'")

class PreMatchResponse(BaseModel):
    team_1: str
    team_2: str
    venue: str
    team_1_win_probability: float
    team_2_win_probability: float
    model_name: str
    features_used: dict


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health():
    """Health check — confirms database and model are reachable."""
    try:
        con = get_db()
        match_count = con.execute("SELECT COUNT(*) FROM main_gold.fact_matches").fetchone()[0]
        champion = get_champion()
        return {
            "status": "ok",
            "total_matches": match_count,
            "model_loaded": champion is not None,
            "model_name": champion["name"] if champion else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/prematch", response_model=PreMatchResponse, tags=["Prediction"])
def predict_prematch(req: PreMatchRequest):
    """
    Pre-match win probability prediction.
    
    Computes features from historical data and runs the champion model
    to return calibrated win probabilities for team_1 and team_2.
    """
    champion = get_champion()
    if not champion:
        raise HTTPException(status_code=503, detail="Champion model not loaded. Run training first.")

    con = get_db()

    # --- Feature Engineering (mirrors build_training_dataset.py) ---
    # 1. Venue avg 1st innings runs
    venue_res = con.execute(f"""
        SELECT AVG(i.total_runs) as avg_runs
        FROM main_gold.fact_innings i
        JOIN main_gold.fact_matches m ON i.match_id = m.match_id
        WHERE m.venue = ? AND i.innings_number = 1
    """, [req.venue]).fetchone()
    venue_avg = float(venue_res[0] or 150.0)

    # 2. H2H win rate for team_1 vs team_2
    h2h = con.execute(f"""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN winner = ? THEN 1 ELSE 0 END) as wins
        FROM main_gold.fact_matches m
        JOIN main_silver.slv_match_teams t1 ON m.match_id = t1.match_id
        JOIN main_silver.slv_match_teams t2 ON m.match_id = t2.match_id
        WHERE t1.team = ? AND t2.team = ?
    """, [req.team_1, req.team_1, req.team_2]).fetchone()
    h2h_rate = float(h2h[1]) / max(float(h2h[0]), 1)

    # 3. Rolling form (last 5 matches) per team
    def get_form(team: str) -> float:
        res = con.execute(f"""
            SELECT AVG(team_1_win::FLOAT) FROM (
                SELECT team_1_win FROM main_gold.fact_matches
                WHERE toss_winner = ?
                ORDER BY match_date DESC LIMIT 5
            )
        """, [team]).fetchone()
        return float(res[0] or 0.5)

    t1_form = get_form(req.team_1)
    t2_form = get_form(req.team_2)
    toss_bat = 1 if req.toss_decision == "bat" else 0

    features = {
        "toss_bat": toss_bat,
        "venue_avg_1st_inns_runs": round(venue_avg, 2),
        "team_1_h2h_win_rate": round(h2h_rate, 3),
        "team_1_form_last5": round(t1_form, 3),
        "team_2_form_last5": round(t2_form, 3),
    }

    X = pd.DataFrame([features])
    prob_t1 = float(champion["model"].predict_proba(X)[0][1])
    prob_t2 = round(1.0 - prob_t1, 4)
    prob_t1 = round(prob_t1, 4)

    return PreMatchResponse(
        team_1=req.team_1,
        team_2=req.team_2,
        venue=req.venue,
        team_1_win_probability=prob_t1,
        team_2_win_probability=prob_t2,
        model_name=champion["name"],
        features_used=features,
    )


@app.get("/predict/live", tags=["Prediction"])
def predict_live(
    match_id: int = Query(..., description="Match ID from fact_matches"),
    current_over: int = Query(0, description="Current over number (0-19)"),
    current_score: int = Query(0, description="Batting team's current score"),
    wickets_fallen: int = Query(0, description="Wickets fallen so far"),
):
    """
    Live win probability estimate (simplified Duckworth-Lewis approximation).
    
    Uses remaining resources (balls + wickets) vs target to estimate win probability.
    """
    con = get_db()
    res = con.execute(f"""
        SELECT i.total_runs as first_inns_target
        FROM main_gold.fact_innings i
        WHERE i.match_id = ? AND i.innings_number = 1
    """, [match_id]).fetchone()

    if not res:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found or 1st innings not complete.")

    target = int(res[0]) + 1
    balls_remaining = (20 - current_over) * 6
    runs_needed = target - current_score
    wickets_remaining = 10 - wickets_fallen

    if runs_needed <= 0:
        return {"chasing_win_probability": 1.0, "defending_win_probability": 0.0, "result": "Target achieved"}

    # Simple resource model: required run rate vs current run rate vs wickets in hand
    balls_used = current_over * 6
    crr = (current_score / max(balls_used, 1)) * 6
    rrr = (runs_needed / max(balls_remaining, 1)) * 6
    resource_factor = (wickets_remaining / 10) * (balls_remaining / 120)

    # Higher resource + lower RRR = higher chase probability
    chase_prob = round(min(max(resource_factor * (crr / max(rrr, 0.1)), 0.0), 1.0), 3)

    return {
        "match_id": match_id,
        "target": target,
        "current_score": current_score,
        "wickets_fallen": wickets_fallen,
        "runs_needed": runs_needed,
        "balls_remaining": balls_remaining,
        "required_run_rate": round(rrr, 2),
        "current_run_rate": round(crr, 2),
        "chasing_win_probability": chase_prob,
        "defending_win_probability": round(1 - chase_prob, 3),
    }


@app.get("/stats/team/{team_name}", tags=["Stats"])
def team_stats(
    team_name: str,
    last_n: int = Query(20, description="Last N matches to include"),
):
    """Historical win/loss record, toss stats, and venue breakdown for a team."""
    con = get_db()

    overall = con.execute(f"""
        SELECT COUNT(*) total_matches,
               SUM(CASE WHEN winner = ? THEN 1 ELSE 0 END) wins,
               ROUND(AVG(CASE WHEN winner = ? THEN 1.0 ELSE 0.0 END) * 100, 1) win_pct
        FROM main_gold.fact_matches m
        JOIN main_silver.slv_match_teams t ON m.match_id = t.match_id
        WHERE t.team = ?
    """, [team_name, team_name, team_name]).df()

    if overall.iloc[0]["total_matches"] == 0:
        raise HTTPException(status_code=404, detail=f"No data found for team: {team_name}")

    venues = con.execute(f"""
        SELECT m.venue,
               COUNT(*) as matches,
               SUM(CASE WHEN m.winner = ? THEN 1 ELSE 0 END) as wins,
               ROUND(SUM(CASE WHEN m.winner = ? THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 1) as win_pct
        FROM main_gold.fact_matches m
        JOIN main_silver.slv_match_teams t ON m.match_id = t.match_id
        WHERE t.team = ?
        GROUP BY m.venue HAVING matches >= 3
        ORDER BY win_pct DESC LIMIT 10
    """, [team_name, team_name, team_name]).df().to_dict("records")

    return {
        "team": team_name,
        "overall": overall.to_dict("records")[0],
        "top_venues": venues,
    }


@app.get("/stats/player/{player_name}", tags=["Stats"])
def player_stats(player_name: str):
    """Career batting and bowling statistics for a player."""
    con = get_db()

    batting = con.execute(f"""
        SELECT SUM(runs_batter) runs, SUM(is_legal_ball) balls,
               COUNT(DISTINCT match_id) innings,
               ROUND(SUM(runs_batter)::FLOAT / NULLIF(SUM(is_legal_ball),0) * 100, 2) sr,
               MAX(innings_runs) best
        FROM (
            SELECT match_id, innings_number, SUM(runs_batter) as innings_runs,
                   SUM(runs_batter) as runs_batter, SUM(is_legal_ball) as is_legal_ball
            FROM main_gold.fact_deliveries WHERE batter = ?
            GROUP BY match_id, innings_number
        )
    """, [player_name]).df().to_dict("records")

    bowling = con.execute(f"""
        SELECT SUM(is_wicket) wickets,
               ROUND(SUM(runs_batter + runs_extras)::FLOAT / NULLIF(SUM(is_legal_ball),0) * 6, 2) economy,
               ROUND(SUM(is_legal_ball)::FLOAT / NULLIF(SUM(is_wicket),0), 1) bowling_sr,
               COUNT(DISTINCT match_id) innings_bowled
        FROM main_gold.fact_deliveries WHERE bowler = ?
    """, [player_name]).df().to_dict("records")

    if not batting or batting[0]["innings"] == 0:
        raise HTTPException(status_code=404, detail=f"No data found for player: {player_name}")

    return {
        "player": player_name,
        "batting": batting[0],
        "bowling": bowling[0],
    }


@app.get("/stats/venue/{venue_name}", tags=["Stats"])
def venue_stats(venue_name: str):
    """Venue scoring patterns, toss impact, and chase success rates."""
    con = get_db()

    summary = con.execute(f"""
        SELECT COUNT(*) total_matches,
               ROUND(AVG(i.total_runs) FILTER (WHERE i.innings_number=1), 1) avg_1st_inns,
               ROUND(AVG(i.total_runs) FILTER (WHERE i.innings_number=2), 1) avg_2nd_inns,
               SUM(CASE WHEN m.toss_decision='field' AND m.toss_winner=m.winner 
                        OR m.toss_decision='bat' AND m.toss_winner!=m.winner THEN 1 ELSE 0 END) chases_won
        FROM main_gold.fact_matches m
        JOIN main_gold.fact_innings i ON m.match_id = i.match_id
        WHERE m.venue = ?
    """, [venue_name]).df()

    if summary.iloc[0]["total_matches"] == 0:
        raise HTTPException(status_code=404, detail=f"No data found for venue: {venue_name}")

    total = int(summary.iloc[0]["total_matches"])
    chases = int(summary.iloc[0]["chases_won"] or 0)

    return {
        "venue": venue_name,
        "total_matches": total,
        "avg_1st_innings_score": summary.iloc[0]["avg_1st_inns"],
        "avg_2nd_innings_score": summary.iloc[0]["avg_2nd_inns"],
        "chase_success_rate": round(chases / max(total, 1) * 100, 1),
    }
