"""
src/features/build_training_dataset.py
───────────────────────────────────────
Builds the final ML-ready `training_dataset.parquet` by querying the DuckDB Gold layer.

Key ML features engineering:
  1. Venue historical first-innings average (before each match)
  2. Team Head-to-Head win rate (before each match)
  3. Rolling 5-match form for team_1 and team_2

Output Schema (1 row per Match, target = team_1_win):
 - match_id, match_date, venue, toss_decision, team_1, team_2
 - venue_avg_1st_inns_runs, team_1_h2h_win_rate
 - team_1_form_last5, team_2_form_last5
 - team_1_win  (classification target label)
"""

import os
import duckdb
import pandas as pd
from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)


def build_features(duckdb_path: str, output_parquet: str) -> pd.DataFrame:
    log.info(f"Extracting ML features from Gold Layer: {duckdb_path}")
    con = duckdb.connect(duckdb_path, read_only=True)

    # ── STEP 1: Canonical Base Match Table ────────────────────────────────────
    # Resolve team_2 via slv_match_teams (every team in match_id)
    log.info("Building base match table ...")
    df_matches = con.execute("""
        SELECT m.match_id, m.match_date, m.venue, m.toss_decision,
               m.team_1, m.team_1_win, m.result_type
        FROM main_gold.fact_matches m
        WHERE m.result_type NOT IN ('no result', 'tie')
          AND m.team_1_win IS NOT NULL
        ORDER BY m.match_date ASC
    """).df()

    # Resolve team_2 (the other team that isn't team_1)
    df_teams = con.execute("""
        SELECT t.match_id, t.team
        FROM main_silver.slv_match_teams t
    """).df()

    # Group by match_id and pick the team that isn't team_1
    team2_lookup = {}
    for mid, grp in df_teams.groupby("match_id"):
        teams = list(grp["team"])
        team2_lookup[mid] = teams  # will resolve later

    def resolve_team2(row):
        teams = team2_lookup.get(row["match_id"], [])
        others = [t for t in teams if t != row["team_1"]]
        return others[0] if others else None

    df_matches["team_2"] = df_matches.apply(resolve_team2, axis=1)
    df_matches = df_matches.dropna(subset=["team_2"])
    log.info(f"Base matches: {len(df_matches):,}")

    # ── STEP 2: Venue Feature — Running Average of 1st Innings Score ─────────
    log.info("Computing venue historical averages ...")
    df_innings = con.execute("""
        SELECT match_id, innings_number, total_runs
        FROM main_gold.fact_innings
    """).df()

    first_innings = df_innings[df_innings["innings_number"] == 1][["match_id", "total_runs"]]
    first_innings.columns = ["match_id", "first_inns_runs"]

    df_matches = df_matches.merge(first_innings, on="match_id", how="left")

    # Compute expanding (historical) mean per venue
    df_matches = df_matches.sort_values("match_date").reset_index(drop=True)
    venue_means = df_matches.groupby("venue")["first_inns_runs"].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=0, drop=True)
    df_matches["venue_avg_1st_inns_runs"] = venue_means.fillna(150.0).round(2)

    # ── STEP 3: Head-to-Head Win Rate ─────────────────────────────────────────
    log.info("Computing head-to-head win rates ...")

    h2h_records = []
    for idx, row in df_matches.iterrows():
        t1, t2 = row["team_1"], row["team_2"]
        dt = row["match_date"]

        # Historical matches between these two teams before this date
        past = df_matches[
            (df_matches["match_date"] < dt)
            & (
                ((df_matches["team_1"] == t1) & (df_matches["team_2"] == t2))
                | ((df_matches["team_1"] == t2) & (df_matches["team_2"] == t1))
            )
        ]

        if len(past) == 0:
            h2h_win_rate = 0.50
        else:
            # Count wins for team_1 in 'our perspective' (where team_1 is focal, accounting for reversal)
            wins_for_t1 = (
                ((past["team_1"] == t1) & (past["team_1_win"] == 1)).sum()
                + ((past["team_1"] == t2) & (past["team_1_win"] == 0)).sum()
            )
            h2h_win_rate = round(wins_for_t1 / len(past), 3)

        h2h_records.append(h2h_win_rate)

    df_matches["team_1_h2h_win_rate"] = h2h_records

    # ── STEP 4: Rolling Form — Last 5 Matches Win Rate ────────────────────────
    log.info("Computing rolling team form ...")

    # Build a long-form record of (date, team, win)
    form_rows = []
    for _, row in df_matches.iterrows():
        form_rows.append({"match_date": row["match_date"], "team": row["team_1"], "won": int(row["team_1_win"])})
        form_rows.append({"match_date": row["match_date"], "team": row["team_2"], "won": 1 - int(row["team_1_win"])})

    df_form = pd.DataFrame(form_rows).sort_values("match_date").reset_index(drop=True)

    def rolling_form(team: str, before_date, window: int = 5) -> float:
        past = df_form[(df_form["team"] == team) & (df_form["match_date"] < before_date)].tail(window)
        if len(past) == 0:
            return 0.50
        return round(past["won"].mean(), 3)

    df_matches["team_1_form_last5"] = [
        rolling_form(r["team_1"], r["match_date"]) for _, r in df_matches.iterrows()
    ]
    df_matches["team_2_form_last5"] = [
        rolling_form(r["team_2"], r["match_date"]) for _, r in df_matches.iterrows()
    ]

    # ── STEP 5: Toss Encoding ─────────────────────────────────────────────────
    df_matches["toss_bat"] = (df_matches["toss_decision"] == "bat").astype(int)

    # ── STEP 6: Final Feature Table ────────────────────────────────────────────
    feature_cols = [
        "match_id", "match_date", "venue", "team_1", "team_2",
        "toss_bat",
        "venue_avg_1st_inns_runs",
        "team_1_h2h_win_rate",
        "team_1_form_last5", "team_2_form_last5",
        "team_1_win"
    ]
    df_final = df_matches[feature_cols].dropna(subset=["team_1_win"])
    con.close()

    log.info(f"Training dataset shape: {df_final.shape}")
    log.info(f"Class balance - team_1 wins: {df_final['team_1_win'].mean():.2%}")

    # ── STEP 7: Save to Parquet ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df_final.to_parquet(output_parquet, index=False)
    log.info(f"Training dataset written to: {output_parquet}")
    return df_final


if __name__ == "__main__":
    cfg = get_config()
    db = str(resolve_path(cfg["paths"]["duckdb_path"]))
    out = str(resolve_path(cfg["paths"]["training_dataset"]))
    build_features(db, out)
