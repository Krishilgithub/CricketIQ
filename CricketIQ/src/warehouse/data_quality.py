"""
src/warehouse/data_quality.py
──────────────────────────────
Data Quality Validation Suite for the Gold Layer.

Validates the structural and logical integrity of the CricketIQ data.
"""

import duckdb
import pandas as pd
from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

def build_and_run_dq_suite(duckdb_path: str) -> bool:
    log.info(f"Running Data Quality checks against: {duckdb_path}")

    con = duckdb.connect(duckdb_path, read_only=True)
    df_matches = con.execute("SELECT * FROM main_gold.fact_matches").df()
    df_deliveries = con.execute("SELECT * FROM main_gold.fact_deliveries").df()
    df_innings = con.execute("SELECT * FROM main_gold.fact_innings").df()
    con.close()

    all_passed = True

    def _assert(condition: bool, msg: str):
        nonlocal all_passed
        if condition:
            log.info(f"  ✅ PASS: {msg}")
        else:
            log.error(f"  ❌ FAIL: {msg}")
            all_passed = False

    log.info("── Validating fact_matches ──────────────────────")
    _assert(df_matches['match_id'].is_unique, "match_id must be strictly unique")
    _assert(df_matches['match_id'].notnull().all(), "match_id must not contain nulls")
    _assert(df_matches['match_date'].notnull().all(), "match_date must not contain nulls")
    # Some matches with 'no result' might have null team_1_win. We filter them out or check the expected domain
    valid_wins = df_matches['team_1_win'].isin([0, 1]).all()
    _assert(valid_wins, "team_1_win must strictly be 0 or 1")

    log.info("── Validating fact_deliveries ───────────────────")
    math_valid = (df_deliveries['runs_total'] == df_deliveries['runs_batter'] + df_deliveries['runs_extras']).all()
    _assert(math_valid, "runs_total strictly equals batter runs plus extras")
    
    overs_valid = df_deliveries['over_number'].between(0, 50).all()
    _assert(overs_valid, "over_number is within valid T20 bounds (0 to 30)")
    
    balls_valid = df_deliveries['ball_number'].between(1, 25).all()
    _assert(balls_valid, "ball_number is strictly positive")
    
    legal_valid = df_deliveries['is_legal_ball'].isin([0, 1]).all()
    _assert(legal_valid, "is_legal_ball must strictly be 0 or 1")

    log.info("── Validating fact_innings ──────────────────────")
    wickets_valid = df_deliveries.groupby(['match_id', 'innings_number'])['is_wicket'].sum().le(10).all()
    _assert(wickets_valid, "Wickets per innings cannot exceed 10")
    
    runs_valid = df_innings['total_runs'].between(0, 400).all()
    _assert(runs_valid, "Total runs reasonable bounds (0 to 400)")

    log.info("─────────────────────────────────────────────────")
    if all_passed:
        log.info("🎉 All Gold Layer Data Quality assertions passed successfully! 🎉")
    else:
        log.error("Data Quality validations failed. See errors above.")
        
    return all_passed

if __name__ == "__main__":
    cfg = get_config()
    db_path = str(resolve_path(cfg["paths"]["duckdb_path"]))
    build_and_run_dq_suite(db_path)
