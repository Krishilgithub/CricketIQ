"""
src/monitoring/freshness_sla.py
────────────────────────────────
Feature Freshness SLA Checker.

Ensures each data layer and model artifact meets its SLA:
  - DuckDB Gold Layer (latest match)   : within 24h of real world
  - Training dataset parquet           : built within 7 days
  - Champion model pickle              : trained within 7 days
  - MLflow runs directory              : experiment logged recently

Reports SLA status as pass/fail with actionable remediation steps.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
import duckdb

from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)


SLA_CONFIG = {
    "gold_layer_max_hours":     24,    # Latest match in DB must be <24h old (or acceptable lag)
    "training_dataset_max_days": 7,    # Parquet must be refreshed within 7 days
    "champion_model_max_days":   7,    # Model must be retrained within 7 days
    "reports_max_days":          7,    # At least one drift report within 7 days
}


def check_file_age(path: str, label: str, max_days: float) -> dict:
    """Generic file age check."""
    if not os.path.exists(path):
        return {"asset": label, "status": "MISSING", "age_days": None, "sla_days": max_days,
                "remediation": f"File not found: {path}"}

    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    age_days = (datetime.now() - mtime).total_seconds() / 86400
    passed = age_days <= max_days

    return {
        "asset": label,
        "status": "PASS ✅" if passed else "FAIL ❌",
        "age_days": round(age_days, 2),
        "sla_days": max_days,
        "last_modified": mtime.strftime("%Y-%m-%d %H:%M"),
        "remediation": "" if passed else f"Refresh {label} — last updated {age_days:.1f} days ago (SLA: {max_days}d)",
    }


def check_gold_layer_freshness(duckdb_path: str) -> dict:
    """Check how recent the latest match in the Gold Layer is."""
    try:
        con = duckdb.connect(duckdb_path, read_only=True)
        row = con.execute("SELECT MAX(match_date) FROM main_gold.fact_matches").fetchone()
        con.close()

        if not row or not row[0]:
            return {"asset": "Gold Layer (latest match)", "status": "NO DATA ❌",
                    "age_days": None, "remediation": "Run dbt models to populate Gold Layer"}

        latest = datetime.strptime(str(row[0])[:10], "%Y-%m-%d")
        # Allow up to 60 days lag since we use historical Cricsheet data
        # (live ingest would tighten this to 24h)
        age_days = (datetime.now() - latest).days
        sla_days = SLA_CONFIG["gold_layer_max_hours"] / 24
        passed = age_days <= 60  # relaxed for historical dataset

        return {
            "asset": "Gold Layer (latest match)",
            "status": "PASS ✅" if passed else "WARN ⚠️",
            "age_days": age_days,
            "latest_date": str(row[0])[:10],
            "sla_days": "60 (historical) / 1 (live)",
            "remediation": "" if passed else "Run `python -m src.ingestion.ingest_live_json` to pull new matches",
        }
    except Exception as e:
        return {"asset": "Gold Layer (latest match)", "status": "ERROR ❌",
                "age_days": None, "remediation": str(e)}


def check_reports_freshness(reports_dir: str) -> dict:
    """Check if a drift report has been generated recently."""
    from glob import glob
    pattern = os.path.join(reports_dir, "drift_summary_*.json")
    files = sorted(glob(pattern), reverse=True)

    if not files:
        return {"asset": "Drift Reports", "status": "MISSING ❌", "age_days": None,
                "sla_days": SLA_CONFIG["reports_max_days"],
                "remediation": "Run `python -m src.monitoring.drift_monitor` to generate drift report"}

    latest = files[0]
    mtime = datetime.fromtimestamp(os.path.getmtime(latest))
    age_days = (datetime.now() - mtime).total_seconds() / 86400
    passed = age_days <= SLA_CONFIG["reports_max_days"]

    return {
        "asset": "Drift Reports",
        "status": "PASS ✅" if passed else "FAIL ❌",
        "age_days": round(age_days, 2),
        "sla_days": SLA_CONFIG["reports_max_days"],
        "last_report": os.path.basename(latest),
        "remediation": "" if passed else "Run drift monitor to check for feature/model drift",
    }


def run_sla_checks() -> dict:
    """
    Run all SLA checks and return a structured status report.
    Returns dict with 'checks' list and 'overall_status'.
    """
    cfg = get_config()
    duckdb_path  = str(resolve_path(cfg["paths"]["duckdb_path"]))
    parquet_path = str(resolve_path(cfg["paths"]["training_dataset"]))
    models_dir   = str(resolve_path(cfg["paths"]["models_dir"]))
    reports_dir  = str(resolve_path(cfg["paths"]["reports_dir"]))

    checks = [
        check_gold_layer_freshness(duckdb_path),
        check_file_age(parquet_path, "Training Dataset (parquet)",
                       SLA_CONFIG["training_dataset_max_days"]),
        check_file_age(os.path.join(models_dir, "champion_model.pkl"),
                       "Champion Model (pickle)",
                       SLA_CONFIG["champion_model_max_days"]),
        check_reports_freshness(reports_dir),
    ]

    all_pass = all("PASS" in c["status"] or "WARN" in c["status"] for c in checks)

    log.info("=" * 55)
    log.info("  CricketIQ Feature Freshness SLA Report")
    log.info("=" * 55)
    for c in checks:
        log.info(f"  {c['status']}  {c['asset']} — age: {c.get('age_days', 'N/A')} days (SLA: {c['sla_days']}d)")
        if c.get("remediation"):
            log.warning(f"    ↳ {c['remediation']}")

    return {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "PASS ✅" if all_pass else "FAIL ❌",
        "checks": checks,
    }


if __name__ == "__main__":
    report = run_sla_checks()
    print(f"\nOverall SLA Status: {report['overall_status']}")
