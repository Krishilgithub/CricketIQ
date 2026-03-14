"""
src/monitoring/retrain_trigger.py
──────────────────────────────────
Retraining trigger logic for CricketIQ.

Evaluates three conditions (configurable via config.yaml):
  1. Time-based   — model is older than N days
  2. Volume-based — N new matches since last training
  3. Drift-based  — drift alert from drift_monitor

If ANY condition triggers, initiates the retraining pipeline automatically.
"""

import os
import json
import pickle
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from glob import glob

import pandas as pd

from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)


def get_model_age_days(models_dir: str) -> float:
    """Return how many days ago the champion model was last saved."""
    champion_path = os.path.join(models_dir, "champion_model.pkl")
    if not os.path.exists(champion_path):
        return float("inf")  # No model → always retrain
    mtime = os.path.getmtime(champion_path)
    age = (datetime.now().timestamp() - mtime) / 86400
    return round(age, 2)


def count_new_matches_since_training(parquet_path: str, models_dir: str) -> int:
    """Count matches added to the dataset since the champion model was saved."""
    champion_path = os.path.join(models_dir, "champion_model.pkl")
    if not os.path.exists(champion_path):
        return 9999

    model_mtime = datetime.fromtimestamp(os.path.getmtime(champion_path))
    df = pd.read_parquet(parquet_path)
    df["match_date"] = pd.to_datetime(df["match_date"])
    new_count = int((df["match_date"] > model_mtime).sum())
    return new_count


def get_latest_drift_alert(reports_dir: str) -> bool:
    """Read the most recent drift_summary JSON to check alert status."""
    pattern = os.path.join(reports_dir, "drift_summary_*.json")
    summaries = sorted(glob(pattern), reverse=True)
    if not summaries:
        return False
    with open(summaries[0]) as f:
        summary = json.load(f)
    return bool(summary.get("alert", False))


def check_retrain_conditions(cfg: dict) -> dict:
    """
    Evaluate all three retraining conditions.
    Returns a dict with each condition and whether retraining is needed.
    """
    parquet = str(resolve_path(cfg["paths"]["training_dataset"]))
    models_dir = str(resolve_path(cfg["paths"]["models_dir"]))
    reports_dir = str(resolve_path(cfg["paths"]["reports_dir"]))
    trigger_cfg = cfg.get("monitoring", {}).get("retrain_trigger", {})

    max_age_days = trigger_cfg.get("days_since_last_train", 7)
    min_new_matches = trigger_cfg.get("new_matches_threshold", 50)

    model_age = get_model_age_days(models_dir)
    new_matches = count_new_matches_since_training(parquet, models_dir)
    drift_alert = get_latest_drift_alert(reports_dir)

    conditions = {
        "time_based": {
            "triggered": model_age >= max_age_days,
            "value": f"{model_age:.1f} days (threshold: {max_age_days})",
        },
        "volume_based": {
            "triggered": new_matches >= min_new_matches,
            "value": f"{new_matches} new matches (threshold: {min_new_matches})",
        },
        "drift_based": {
            "triggered": drift_alert,
            "value": f"Drift alert: {drift_alert}",
        },
    }

    should_retrain = any(c["triggered"] for c in conditions.values())

    return {
        "should_retrain": should_retrain,
        "conditions": conditions,
        "checked_at": datetime.now().isoformat(),
    }


def trigger_retraining_if_needed(cfg: dict = None) -> bool:
    """
    Main entry point. Checks conditions and runs the training pipeline
    if retraining is warranted.

    Returns True if retraining was triggered, False otherwise.
    """
    if cfg is None:
        cfg = get_config()

    log.info("Checking retraining conditions...")
    status = check_retrain_conditions(cfg)

    for name, cond in status["conditions"].items():
        icon = "🔴" if cond["triggered"] else "✅"
        log.info(f"  {icon} [{name}]: {cond['value']}")

    if not status["should_retrain"]:
        log.info("✅ No retraining needed at this time.")
        return False

    log.warning("⚡ Retraining triggered! Starting training pipeline...")

    # Run feature engineering first, then retrain
    parquet = str(resolve_path(cfg["paths"]["training_dataset"]))
    models_dir = str(resolve_path(cfg["paths"]["models_dir"]))

    try:
        from src.models.train_model import train_and_select
        champion_path = train_and_select(parquet, models_dir)
        log.info(f"✅ Retraining complete. New champion saved: {champion_path}")

        # Log retraining event
        reports_dir = str(resolve_path(cfg["paths"]["reports_dir"]))
        os.makedirs(reports_dir, exist_ok=True)
        event = {
            "event": "retrain_triggered",
            "timestamp": datetime.now().isoformat(),
            "conditions": status["conditions"],
            "champion_path": champion_path,
        }
        with open(os.path.join(reports_dir, f"retrain_event_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(event, f, indent=2)

        return True

    except Exception as e:
        log.error(f"Retraining failed: {e}")
        return False


if __name__ == "__main__":
    cfg = get_config()
    status = check_retrain_conditions(cfg)
    print("\n=== Retraining Trigger Status ===")
    for name, cond in status["conditions"].items():
        icon = "🔴 TRIGGERED" if cond["triggered"] else "✅ OK"
        print(f"  {name:15s}: {icon} | {cond['value']}")
    print(f"\n  Should retrain: {'YES ⚡' if status['should_retrain'] else 'NO'}")

    if status["should_retrain"]:
        trigger_retraining_if_needed(cfg)
