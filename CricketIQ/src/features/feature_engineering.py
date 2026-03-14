"""
src/features/feature_engineering.py
───────────────────────────────────
Advanced feature engineering for CricketIQ match outcome prediction.
Generates comprehensive features including rolling team form, head-to-head rates,
venue advantages, batting/bowling strength aggregates, and momentum indexes.
"""

import os
import duckdb
import pandas as pd
import numpy as np
from typing import Tuple
from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

def get_base_matches(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extract base match records excluding ties/no results."""
    # Resolve team_2 by joining with slv_match_teams
    df_matches = con.execute("""
        WITH teams AS (
            SELECT match_id, LIST(team) as teams
            FROM main_silver.slv_match_teams
            GROUP BY match_id
        )
        SELECT 
            m.match_id, 
            CAST(m.match_date AS DATE) as match_date, 
            m.venue, 
            m.toss_decision,
            m.toss_winner,
            m.team_1, 
            t.teams,
            m.team_1_win, 
            m.result_type,
            m.event_name
        FROM main_gold.fact_matches m
        JOIN teams t ON m.match_id = t.match_id
        WHERE m.result_type NOT IN ('no result', 'tie')
          AND m.team_1_win IS NOT NULL
        ORDER BY m.match_date ASC
    """).df()

    # Determine team_2
    def extract_team_2(row):
        teams = row["teams"]
        t1 = row["team_1"]
        others = [t for t in teams if t != t1]
        return others[0] if others else None
        
    df_matches["team_2"] = df_matches.apply(extract_team_2, axis=1)
    return df_matches.dropna(subset=["team_2"]).copy()

def compute_rolling_form(df: pd.DataFrame, window_sizes=[5, 10]) -> pd.DataFrame:
    """Compute rolling win rates and momentum for each team."""
    log.info("Computing rolling team form...")
    form_rows = []
    for _, row in df.iterrows():
        dt = row["match_date"]
        form_rows.append({"match_date": dt, "team": row["team_1"], "won": int(row["team_1_win"])})
        form_rows.append({"match_date": dt, "team": row["team_2"], "won": 1 - int(row["team_1_win"])})

    df_form = pd.DataFrame(form_rows).sort_values("match_date").reset_index(drop=True)
    
    # Pre-calculate for speed
    team_histories = {}
    for team, grp in df_form.groupby("team"):
        # Shift to avoid lookahead bias (don't include current match)
        won_shifted = grp["won"].shift(1)
        
        hist = pd.DataFrame({"match_date": grp["match_date"]})
        hist["form_last5"] = won_shifted.rolling(5, min_periods=1).mean().fillna(0.5)
        hist["form_last10"] = won_shifted.rolling(10, min_periods=1).mean().fillna(0.5)
        
        # Weighted momentum (more weight to recent matches)
        # e.g., sum of (won * weight) / sum(weights)
        weights = np.arange(1, 6)
        def weighted_avg(x):
            if len(x) < 5: return x.mean()
            return np.average(x, weights=weights)
        hist["momentum_idx"] = won_shifted.rolling(5, min_periods=1).apply(weighted_avg, raw=True).fillna(0.5)
        team_histories[team] = hist

    # Map back to main matches dataframe
    def get_hist_val(team, dt, col):
        if team not in team_histories: return 0.5
        th = team_histories[team]
        val = th[th["match_date"] == dt][col]
        return val.iloc[0] if not val.empty else 0.5

    for w in window_sizes:
        df[f"team_1_form_last{w}"] = df.apply(lambda r: get_hist_val(r["team_1"], r["match_date"], f"form_last{w}"), axis=1)
        df[f"team_2_form_last{w}"] = df.apply(lambda r: get_hist_val(r["team_2"], r["match_date"], f"form_last{w}"), axis=1)
    
    df["team_1_momentum"] = df.apply(lambda r: get_hist_val(r["team_1"], r["match_date"], "momentum_idx"), axis=1)
    df["team_2_momentum"] = df.apply(lambda r: get_hist_val(r["team_2"], r["match_date"], "momentum_idx"), axis=1)

    return df

def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head win rate between team 1 and team 2."""
    log.info("Computing Head-to-Head advantages...")
    
    h2h_records = []
    for idx, row in df.iterrows():
        t1, t2, dt = row["team_1"], row["team_2"], row["match_date"]
        
        # Past matches between these two
        past = df[(df["match_date"] < dt) & 
                  (((df["team_1"] == t1) & (df["team_2"] == t2)) | 
                   ((df["team_1"] == t2) & (df["team_2"] == t1)))]
        
        if len(past) == 0:
            h2h_records.append(0.5)
        else:
            wins_t1 = ((past["team_1"] == t1) & (past["team_1_win"] == 1)).sum() + \
                      ((past["team_1"] == t2) & (past["team_1_win"] == 0)).sum()
            h2h_records.append(wins_t1 / len(past))
            
    df["team_1_h2h_win_rate"] = np.round(h2h_records, 3)
    return df

def compute_venue_features(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute venue-specific features: avg 1st innings score, team win rate at venue."""
    log.info("Computing venue features...")
    
    # 1. First innings score
    df_innings = con.execute("SELECT match_id, innings_number, total_runs FROM main_gold.fact_innings").df()
    first_inns = df_innings[df_innings["innings_number"] == 1][["match_id", "total_runs"]]
    first_inns.columns = ["match_id", "first_inns_runs"]
    
    df = df.merge(first_inns, on="match_id", how="left")
    venue_means = df.groupby("venue")["first_inns_runs"].apply(lambda x: x.expanding().mean().shift(1)).reset_index(level=0, drop=True)
    df["venue_avg_1st_inns_runs"] = venue_means.fillna(150.0).round(2)
    
    # Calculate chasing success rate at venue up to this match
    chase_success = []
    for idx, row in df.iterrows():
        venue, dt = row["venue"], row["match_date"]
        past_venue = df[(df["venue"] == venue) & (df["match_date"] < dt)]
        if len(past_venue) == 0:
            chase_success.append(0.5)
        else:
            # Chasing team wins if toss_decision=field and they win, or toss_decision=bat and they lose
            # Or simpler: generally innings 2 winner
            # Since team_1 is toss_winner:
            # If team_1 bats first (toss_bat=1): chasing team wins if team_1_win=0
            # If team_1 fields first (toss_bat=0): chasing team wins if team_1_win=1
            chasing_wins = past_venue.apply(
                lambda r: 1 if (r["toss_decision"] == "bat" and r["team_1_win"] == 0) or 
                              (r["toss_decision"] == "field" and r["team_1_win"] == 1) else 0, axis=1
            ).sum()
            chase_success.append(chasing_wins / len(past_venue))
            
    df["venue_chase_success_rate"] = np.round(chase_success, 3)
    
    # Team 1 Venue Win Rate
    def team_venue_win_rate(team, venue, dt):
        past = df[(df["venue"] == venue) & (df["match_date"] < dt) & 
                  ((df["team_1"] == team) | (df["team_2"] == team))]
        if len(past) == 0: return 0.5
        wins = ((past["team_1"] == team) & (past["team_1_win"] == 1)).sum() + \
               ((past["team_2"] == team) & (past["team_1_win"] == 0)).sum()
        return wins / len(past)
    
    df["team_1_venue_win_rate"] = df.apply(lambda r: team_venue_win_rate(r["team_1"], r["venue"], r["match_date"]), axis=1)
    df["team_2_venue_win_rate"] = df.apply(lambda r: team_venue_win_rate(r["team_2"], r["venue"], r["match_date"]), axis=1)

    return df

def build_advanced_features(duckdb_path: str, output_parquet: str) -> pd.DataFrame:
    log.info(f"Connecting to Data Warehouse: {duckdb_path}")
    con = duckdb.connect(duckdb_path, read_only=True)
    
    df = get_base_matches(con)
    df = compute_rolling_form(df, window_sizes=[5, 10])
    df = compute_h2h_features(df)
    df = compute_venue_features(df, con)
    
    # Toss context
    df["toss_bat"] = (df["toss_decision"] == "bat").astype(int)
    # Phase 22: Removed toss_winner_is_team_1 (always 1, useless)
    
    # Phase 22: Add relative strength features
    df["form_diff"] = df["team_1_form_last5"] - df["team_2_form_last5"]
    df["momentum_diff"] = df["team_1_momentum"] - df["team_2_momentum"]
    
    con.close()
    
    # Clean up and select features
    feature_cols = [
        "match_id", "match_date", "venue", "team_1", "team_2", 
        "toss_bat",
        "venue_avg_1st_inns_runs", "venue_chase_success_rate",
        "team_1_h2h_win_rate", 
        "team_1_form_last5", "team_1_form_last10", "team_1_momentum",
        "team_2_form_last5", "team_2_form_last10", "team_2_momentum",
        "team_1_venue_win_rate", "team_2_venue_win_rate",
        "form_diff", "momentum_diff",
        "team_1_win"
    ]
    
    df_final = df[feature_cols].dropna(subset=["team_1_win"])
    
    log.info(f"Final Advanced Features Shape: {df_final.shape}")
    
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df_final.to_parquet(output_parquet, index=False)
    log.info(f"New ML-ready dataset written successfully to: {output_parquet}")
    
    return df_final

if __name__ == "__main__":
    cfg = get_config()
    db = str(resolve_path(cfg["paths"]["duckdb_path"]))
    out = str(resolve_path(cfg["paths"]["training_dataset"]))
    build_advanced_features(db, out)
