"""
src/monitoring/drift_monitor.py
────────────────────────────────
Data & Model Drift Monitoring using Evidently AI.

Detects:
  1. Feature Distribution Drift  — training vs recent production data
  2. Model Prediction Drift      — probability distribution shift over time
  3. Target Label Drift          — win rate shift (concept drift proxy)

Reports saved as HTML to artifacts/reports/drift_*.html
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

FEATURE_COLS = [
    "toss_bat",
    "venue_avg_1st_inns_runs",
    "team_1_h2h_win_rate",
    "team_1_form_last5",
    "team_2_form_last5",
]

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
    from evidently.metrics import (
        DatasetDriftMetric,
        ColumnDriftMetric,
    )
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False
    log.warning("Evidently not installed. Install with: pip install evidently")


def load_training_reference(parquet_path: str) -> pd.DataFrame:
    """Load training dataset as reference (first 80% time-sorted rows)."""
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("match_date").reset_index(drop=True)
    cutoff = int(len(df) * 0.8)
    return df.iloc[:cutoff][FEATURE_COLS + ["team_1_win"]]


def load_recent_data(parquet_path: str, days: int = 180) -> pd.DataFrame:
    """Load recent data (last N days) as current window for drift check."""
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("match_date").reset_index(drop=True)
    cutoff_date = pd.to_datetime(df["match_date"].max()) - timedelta(days=days)
    recent = df[pd.to_datetime(df["match_date"]) > cutoff_date]
    if len(recent) < 30:
        # Not enough data — use last 20% instead
        recent = df.iloc[int(len(df) * 0.8):]
    return recent[FEATURE_COLS + ["team_1_win"]]


def run_data_drift_report(reference: pd.DataFrame, current: pd.DataFrame, report_path: str) -> dict:
    """Run Evidently DataDriftPreset and save HTML report."""
    if not HAS_EVIDENTLY:
        log.warning("Evidently not available — skipping drift report generation.")
        return {"drift_detected": False, "error": "Evidently not installed"}

    col_mapping = ColumnMapping(
        target="team_1_win",
        numerical_features=FEATURE_COLS,
    )

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
        TargetDriftPreset(),
    ])
    report.run(reference_data=reference, current_data=current, column_mapping=col_mapping)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report.save_html(report_path)
    log.info(f"Drift report saved: {report_path}")

    # Extract structured results
    result_dict = report.as_dict()
    drift_detected = False
    drifted_cols = []

    for metric in result_dict.get("metrics", []):
        name = metric.get("metric", "")
        result = metric.get("result", {})
        if "dataset_drift" in name.lower() or "drift_share" in str(result):
            drift_score = result.get("drift_share", result.get("drift_by_columns", {}))
            if isinstance(drift_score, float) and drift_score > 0.5:
                drift_detected = True

    return {
        "drift_detected": drift_detected,
        "drifted_columns": drifted_cols,
        "report_path": report_path,
        "reference_size": len(reference),
        "current_size": len(current),
    }


def run_prediction_drift(reference_probs: np.ndarray, current_probs: np.ndarray) -> dict:
    """
    KL-Divergence based prediction distribution drift check.
    No Evidently required for this lightweight check.
    """
    # Bin probabilities into 10 buckets
    ref_hist, _ = np.histogram(reference_probs, bins=10, range=(0, 1), density=True)
    cur_hist, _ = np.histogram(current_probs, bins=10, range=(0, 1), density=True)

    # Smooth zeros
    eps = 1e-10
    ref_hist = ref_hist + eps
    cur_hist = cur_hist + eps
    ref_hist /= ref_hist.sum()
    cur_hist /= cur_hist.sum()

    kl_div = float(np.sum(ref_hist * np.log(ref_hist / cur_hist)))
    drift_detected = kl_div > 0.1  # threshold: 0.1 nats

    return {
        "kl_divergence": round(kl_div, 4),
        "drift_detected": drift_detected,
        "ref_mean_prob": round(float(reference_probs.mean()), 4),
        "cur_mean_prob": round(float(current_probs.mean()), 4),
    }


def check_target_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Check for concept drift via win rate shift in target label."""
    ref_rate = float(reference["team_1_win"].mean())
    cur_rate = float(current["team_1_win"].mean())
    drift = abs(ref_rate - cur_rate) > 0.05  # >5% shift = drift

    log.info(f"Target drift check: ref_win_rate={ref_rate:.3f}, cur_win_rate={cur_rate:.3f}, drift={drift}")
    return {
        "reference_win_rate": round(ref_rate, 4),
        "current_win_rate": round(cur_rate, 4),
        "shift": round(cur_rate - ref_rate, 4),
        "drift_detected": drift,
    }


def run_full_monitoring(parquet_path: str, reports_dir: str) -> dict:
    """
    Full monitoring run: data drift + target drift.
    Returns a summary dict with drift signals and report paths.
    """
    log.info("=" * 55)
    log.info("  CricketIQ Drift Monitoring Run")
    log.info("=" * 55)

    reference = load_training_reference(parquet_path)
    current = load_recent_data(parquet_path)

    log.info(f"Reference size: {len(reference):,} | Current window: {len(current):,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(reports_dir, f"drift_report_{timestamp}.html")

    drift_result = run_data_drift_report(reference, current, report_path)
    target_result = check_target_drift(reference, current)

    summary = {
        "timestamp": timestamp,
        "data_drift": drift_result,
        "target_drift": target_result,
        "alert": drift_result.get("drift_detected", False) or target_result["drift_detected"],
    }

    # Save JSON summary for CI/CD and monitoring dashboards
    json_path = os.path.join(reports_dir, f"drift_summary_{timestamp}.json")
    os.makedirs(reports_dir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Monitoring summary saved: {json_path}")

    if summary["alert"]:
        log.warning("⚠️  DRIFT ALERT — Consider retraining the model!")
    else:
        log.info("✅ No significant drift detected.")

    return summary


if __name__ == "__main__":
    cfg = get_config()
    parquet = str(resolve_path(cfg["paths"]["training_dataset"]))
    reports = str(resolve_path(cfg["paths"]["reports_dir"]))
    result = run_full_monitoring(parquet, reports)
    print("\n=== Drift Monitoring Summary ===")
    print(f"  Data  Drift : {result['data_drift'].get('drift_detected', False)}")
    print(f"  Target Drift: {result['target_drift']['drift_detected']} "
          f"(shift={result['target_drift']['shift']:+.3f})")
    print(f"  ALERT       : {result['alert']}")
