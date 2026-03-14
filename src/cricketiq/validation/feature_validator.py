from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def run_feature_validation(config_path: Path) -> Dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    feature_file = Path(cfg["feature_file"])
    quality_file = Path(cfg["quality_file"])
    output_report = Path(cfg["output_report"])
    required_columns = cfg.get("required_columns", [])

    df = pd.read_csv(feature_file)
    quality = json.loads(quality_file.read_text(encoding="utf-8"))

    missing_required = [c for c in required_columns if c not in df.columns]

    null_ratio = (df.isna().sum() / max(len(df), 1)).to_dict()
    high_null_columns = {k: float(v) for k, v in null_ratio.items() if float(v) > 0.5}

    duplicate_match_rows = int(df.duplicated(subset=["match_id"]).sum()) if "match_id" in df.columns else -1

    checks = {
        "has_required_columns": len(missing_required) == 0,
        "binary_label_presence": "label_is_binary" in df.columns,
        "target_presence": "team_1_win" in df.columns,
        "duplicate_match_rows_is_zero": duplicate_match_rows == 0,
        "quality_duplicate_delivery_keys_is_zero": int(quality.get("duplicate_delivery_keys", 0)) == 0,
        "quality_duplicate_wicket_keys_is_zero": int(quality.get("duplicate_wicket_keys", 0)) == 0,
        "quality_innings_delivery_mismatch_is_zero": int(quality.get("innings_delivery_mismatch_count", 0)) == 0,
    }

    passed = all(checks.values())

    report: Dict[str, Any] = {
        "passed": passed,
        "feature_rows": int(len(df)),
        "feature_columns": int(len(df.columns)),
        "missing_required_columns": missing_required,
        "high_null_columns_gt_50pct": high_null_columns,
        "duplicate_match_rows": duplicate_match_rows,
        "quality_checks": checks,
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
