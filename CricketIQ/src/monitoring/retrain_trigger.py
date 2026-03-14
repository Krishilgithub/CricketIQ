"""
src/monitoring/retrain_trigger.py
─────────────────────────────────
Evaluates conditions (time, drift) to trigger ML model retraining.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
from pathlib import Path
from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

def check_retrain_conditions():
    cfg = get_config()
    report_dir = resolve_path(cfg["paths"]["reports_dir"])
    drift_file = report_dir / "latest_drift.json"
    
    status = {
        "should_retrain": False,
        "reason": "OK"
    }
    
    if drift_file.exists():
        with open(drift_file, "r") as f:
            data = json.load(f)
            if data.get("dataset_drift", False):
                status["should_retrain"] = True
                status["reason"] = "Data drift detected in latest report."
                return status
                
    return status

if __name__ == "__main__":
    status = check_retrain_conditions()
    log.info(f"Retrain check: {status}")
