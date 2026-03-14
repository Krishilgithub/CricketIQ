"""
src/models/train_model.py
──────────────────────────
Trains match outcome prediction models on the engineered training_dataset.parquet.

Three candidate models:
  1. Logistic Regression (baseline)
  2. XGBoost Classifier
  3. LightGBM Classifier

Training Protocol:
  - TimeSeriesSplit (no data leakage — future matches never inform past training)
  - Probability calibration via isotonic regression
  - Primary metric: Log-Loss + Brier Score
  - MLflow tracking for reproducibility
  - Champion model is saved as `artifacts/models/champion_model.pkl`
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

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
TARGET_COL = "team_1_win"


def load_dataset(parquet_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the training dataset and return X, y split."""
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("match_date").reset_index(drop=True)
    X = df[FEATURE_COLS].astype(float)
    y = df[TARGET_COL].astype(int)
    log.info(f"Loaded {len(df):,} rows | Features: {FEATURE_COLS}")
    log.info(f"Class balance: {y.mean():.2%} positive")
    return X, y


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """Evaluate via TimeSeriesSplit; return mean log-loss and brier score."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    ll_scores, bs_scores = [], []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]

        ll_scores.append(log_loss(y_val, probs))
        bs_scores.append(brier_score_loss(y_val, probs))

    return {
        "log_loss_mean": float(np.mean(ll_scores)),
        "log_loss_std": float(np.std(ll_scores)),
        "brier_score_mean": float(np.mean(bs_scores)),
        "brier_score_std": float(np.std(bs_scores)),
    }


def build_candidates() -> list[tuple[str, object]]:
    """Return list of (name, unfitted model) pairs."""
    candidates = [
        (
            "logistic_regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(
                    LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
                    method="isotonic", cv=5
                )),
            ]),
        ),
    ]

    if HAS_XGB:
        candidates.append((
            "xgboost",
            CalibratedClassifierCV(
                xgb.XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", verbosity=0, random_state=42
                ),
                method="isotonic", cv=5
            ),
        ))

    if HAS_LGB:
        candidates.append((
            "lightgbm",
            CalibratedClassifierCV(
                lgb.LGBMClassifier(
                    n_estimators=200, num_leaves=31, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    verbose=-1, random_state=42
                ),
                method="isotonic", cv=5
            ),
        ))

    return candidates


def train_and_select(parquet_path: str, models_dir: str) -> str:
    """Full training loop — evaluates all candidates and saves the champion."""
    X, y = load_dataset(parquet_path)
    os.makedirs(models_dir, exist_ok=True)

    candidates = build_candidates()
    results = {}

    if HAS_MLFLOW:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("cricketiq_match_prediction")

    log.info("=" * 60)
    log.info(f"  Training {len(candidates)} model(s) with TimeSeriesSplit")
    log.info("=" * 60)

    for name, model in candidates:
        log.info(f"[{name}] Evaluating...")
        metrics = evaluate_model(model, X.copy(), y.copy())
        results[name] = {"model": model, "metrics": metrics}

        log.info(f"  Log-Loss : {metrics['log_loss_mean']:.4f} ± {metrics['log_loss_std']:.4f}")
        log.info(f"  Brier    : {metrics['brier_score_mean']:.4f} ± {metrics['brier_score_std']:.4f}")

        if HAS_MLFLOW:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"model": name, "features": FEATURE_COLS})
                mlflow.log_metrics({
                    "log_loss_mean": metrics["log_loss_mean"],
                    "log_loss_std": metrics["log_loss_std"],
                    "brier_score_mean": metrics["brier_score_mean"],
                    "brier_score_std": metrics["brier_score_std"],
                })
                # Note: model artifact stored in pickle, not MLflow registry
                # to avoid MLflow 3.x meta.yaml compatibility issues

    # Select champion: lowest log-loss mean
    champion_name = min(results, key=lambda n: results[n]["metrics"]["log_loss_mean"])
    champion_model = results[champion_name]["model"]
    champion_metrics = results[champion_name]["metrics"]

    log.info("=" * 60)
    log.info(f"🏆 Champion: [{champion_name}]")
    log.info(f"   Log-Loss : {champion_metrics['log_loss_mean']:.4f}")
    log.info(f"   Brier    : {champion_metrics['brier_score_mean']:.4f}")
    log.info("=" * 60)

    # Final fit on complete data
    champion_model.fit(X, y)
    champion_path = os.path.join(models_dir, "champion_model.pkl")
    with open(champion_path, "wb") as f:
        pickle.dump({"model": champion_model, "features": FEATURE_COLS, "name": champion_name, "metrics": champion_metrics}, f)

    log.info(f"Champion model saved to: {champion_path}")
    return champion_path


if __name__ == "__main__":
    cfg = get_config()
    parquet = str(resolve_path(cfg["paths"]["training_dataset"]))
    models_dir = str(resolve_path(cfg["paths"]["models_dir"]))
    train_and_select(parquet, models_dir)
