from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def _safe_metric(func, *args, **kwargs):
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return None


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    eps = 1e-6
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.quantile(expected, quantiles)
    cuts[0] = -np.inf
    cuts[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=cuts)
    actual_counts, _ = np.histogram(actual, bins=cuts)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def run_monitoring(config_path: Path) -> Dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    model_bundle_path = Path(cfg["model_bundle"])
    feature_file = Path(cfg["feature_file"])
    output_report = Path(cfg["output_report"])
    target_col = cfg.get("target_column", "team_1_win")
    binary_flag_col = cfg.get("binary_label_flag_column", "label_is_binary")

    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    features = bundle["features"]

    df = pd.read_csv(feature_file)
    if "outcome_class" in df.columns:
        df = df[df["outcome_class"].isin(["win", "loss"])].copy()
        df[target_col] = df["outcome_class"].map({"win": 1, "loss": 0}).astype(int)
    else:
        if binary_flag_col in df.columns:
            df = df[df[binary_flag_col] == 1].copy()
        parsed_target = pd.to_numeric(df[target_col], errors="coerce")
        df = df[parsed_target.isin([0, 1])].copy()
        df[target_col] = parsed_target.loc[df.index].astype(int)

    if df.empty:
        raise ValueError("No binary-labeled rows available for monitoring")

    df = df.sort_values(by=["match_date", "match_id"]).reset_index(drop=True)

    probs = model.predict_proba(df[features])[:, 1]
    preds = model.predict(df[features])

    y = df[target_col].values

    metrics = {
        "accuracy": float((preds == y).mean()),
        "roc_auc": _safe_metric(roc_auc_score, y, probs),
        "log_loss": _safe_metric(log_loss, y, probs, labels=[0, 1]),
        "brier": _safe_metric(brier_score_loss, y, probs),
    }

    split = max(1, int(len(probs) * 0.7))
    ref = probs[:split]
    cur = probs[split:]
    psi = _psi(ref, cur) if len(cur) > 0 else 0.0

    report = {
        "rows_scored": int(len(df)),
        "metrics": metrics,
        "prediction_prob_mean": float(np.mean(probs)) if len(probs) else None,
        "prediction_prob_std": float(np.std(probs)) if len(probs) else None,
        "probability_psi_ref_vs_recent": float(psi),
        "drift_flag": bool(psi >= 0.2),
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
