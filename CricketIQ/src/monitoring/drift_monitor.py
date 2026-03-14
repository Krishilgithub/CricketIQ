"""
src/monitoring/drift_monitor.py
───────────────────────────────
Calculates data drift and target drift using Evidently AI.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import duckdb
import json
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

def run_full_monitoring():
    cfg = get_config()
    db_path = str(resolve_path(cfg["paths"]["duckdb_path"]))
    report_dir = resolve_path(cfg["paths"]["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    
    con = duckdb.connect(db_path, read_only=True)
    
    # Load dataset
    query = """
    SELECT 
        team_1_win, 
        toss_decision,
        venue,
        team_1,
        team_2
    FROM main_gold.fact_matches
    """
    df = con.execute(query).df()
    
    if len(df) < 500:
        log.warning("Not enough data to run drift monitoring.")
        return
        
    # Split into reference (older 80%) and current (newer 20%)
    split_idx = int(len(df) * 0.8)
    ref_df = df.iloc[:split_idx]
    curr_df = df.iloc[split_idx:]
    
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    report.run(reference_data=ref_df, current_data=curr_df)
    
    report_path = report_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report.save_html(str(report_path))
    
    summary = report.as_dict()
    dataset_drift = summary["metrics"][0]["result"]["dataset_drift"]
    
    payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset_drift": dataset_drift,
        "report_path": str(report_path)
    }
    
    with open(report_dir / "latest_drift.json", "w") as f:
        json.dump(payload, f)
        
    log.info(f"Drift report generated. Dataset drift: {dataset_drift}")
    
if __name__ == "__main__":
    run_full_monitoring()
