"""
src/models/train_models.py
──────────────────────────
Production-grade ML training pipeline for CricketIQ T20 match prediction.
Trains multiple candidate models, performs TimeSeriesSplit cross-validation,
and logs EVERYTHING to MLflow:
  - Dataset metadata
  - Preprocessing parameters
  - All model hyperparameters
  - Full set of CV metrics (mean + std) for every fold
  - All diagnostic plots: Confusion Matrix, ROC, PR, Feature Importance, Learning Curve, SHAP
  - Feature analysis artifacts: correlation heatmap, distributions, data quality report
  - Champion model to MLflow Model Registry with Production alias
  - Final experiment_summary.json
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.base import clone

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

import mlflow
import mlflow.sklearn

from src.config import get_config, resolve_path
from src.logger import get_logger
from src.features.data_preprocessing import (
    get_preprocessor, get_training_columns,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)
from src.models.mlflow_tracking import (
    setup_mlflow,
    log_dataset_metadata,
    log_preprocessing_params,
    log_model_params,
    log_feature_list_artifact,
    log_data_quality_artifact,
    log_feature_correlation_artifact,
    log_feature_distribution_artifact,
    log_champion_to_registry,
)
from src.models.evaluate_models import (
    evaluate_predictions,
    log_all_artifacts,
)

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(parquet_path: str):
    """Load pre-engineered features and targets from Parquet, sorted by date."""
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("match_date").reset_index(drop=True)

    X_cols, y_col = get_training_columns()
    X = df[X_cols]
    y = df[y_col].astype(int)

    log.info(f"Loaded {len(df)} rows | Features: {len(X_cols)} | Class balance: {y.mean():.2%}")
    return X, y, df


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_candidate_models() -> Dict[str, Any]:
    """Define all candidate classifiers with tuned hyperparameters."""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=0.5, solver="lbfgs",
            class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=10,
            min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }

    if xgb:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.75, colsample_bytree=0.75,
            min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=1, eval_metric="logloss",
            random_state=42, use_label_encoder=False
        )

    if lgb:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, num_leaves=24, learning_rate=0.05,
            subsample=0.75, colsample_bytree=0.75,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            class_weight="balanced", random_state=42, verbose=-1
        )

    return models


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(model_pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    TimeSeriesSplit cross-validation.
    Returns:
      - agg_metrics: dict of mean + std for all metrics
      - oof_y_true / oof_y_probs: full out-of-fold arrays for final plots
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    oof_y_true, oof_y_probs = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        fold_pipeline = clone(model_pipeline)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_pipeline.fit(X_tr, y_tr)

        if hasattr(fold_pipeline, "predict_proba"):
            probs = fold_pipeline.predict_proba(X_val)[:, 1]
        else:
            probs = fold_pipeline.decision_function(X_val)

        fold_metrics = evaluate_predictions(y_val.values, probs)
        metrics_list.append(fold_metrics)
        oof_y_true.extend(y_val.values)
        oof_y_probs.extend(probs)

        log.info(
            f"  Fold {fold_idx+1}/{n_splits}: "
            f"AUC={fold_metrics['roc_auc']:.4f}  "
            f"LogLoss={fold_metrics['log_loss']:.4f}  "
            f"F1={fold_metrics['f1']:.4f}"
        )

    # Aggregate: mean + std for each metric
    agg_metrics = {}
    for key in metrics_list[0].keys():
        vals = [m[key] for m in metrics_list]
        agg_metrics[f"{key}"] = float(np.mean(vals))
        agg_metrics[f"{key}_std"] = float(np.std(vals))

    return agg_metrics, np.array(oof_y_true), np.array(oof_y_probs)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train_and_register_champion(parquet_path: str, models_dir: str):
    """
    Full training pipeline with comprehensive MLflow logging:
    1. Load data
    2. Log shared feature analysis artifacts (run once for the experiment)
    3. Train each candidate model with CV
    4. Log params, full metrics (mean + std), and all diagnostic plots
    5. Register the champion model with Production alias
    6. Write experiment_summary.json
    """
    os.makedirs(models_dir, exist_ok=True)
    setup_mlflow()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    X, y, full_df = load_data(parquet_path)
    candidate_models = get_candidate_models()
    preprocessor = get_preprocessor()

    best_log_loss = float("inf")
    best_model_name = None
    best_pipeline = None
    best_metrics = {}
    all_results = {}

    # ── Log shared feature analysis artifacts (once, outside model runs) ──
    log.info("Logging feature analysis artifacts...")
    feature_artifact_dir = os.path.join(models_dir, "feature_analysis")
    os.makedirs(feature_artifact_dir, exist_ok=True)

    with mlflow.start_run(run_name=f"FeatureAnalysis_{run_timestamp}"):
        log_feature_list_artifact(NUMERICAL_FEATURES, CATEGORICAL_FEATURES, feature_artifact_dir)
        log_data_quality_artifact(X, feature_artifact_dir)
        log_feature_correlation_artifact(X, feature_artifact_dir)
        log_feature_distribution_artifact(X, feature_artifact_dir)
        mlflow.set_tag("run_type", "feature_analysis")
        log.info("Feature analysis run complete.")

    # ── Train each candidate model ─────────────────────────────────────────
    log.info(f"Starting training runs for {len(candidate_models)} models.")

    for name, clf in candidate_models.items():
        log.info("=" * 60)
        log.info(f"Training [{name}] with TimeSeriesSplit CV (n_splits=5)")

        run_artifact_dir = os.path.join(models_dir, name)
        os.makedirs(run_artifact_dir, exist_ok=True)

        with mlflow.start_run(run_name=f"{name}_{run_timestamp}"):
            mlflow.set_tag("run_type", "model_training")
            mlflow.set_tag("run_timestamp", run_timestamp)

            # 1. Log dataset + preprocessing metadata
            log_dataset_metadata(full_df, parquet_path, target_col="team_1_win")
            log_preprocessing_params(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
            mlflow.log_param("cv_n_splits", 5)

            # 2. Log model hyperparams
            log_model_params(name, clf)

            # 3. Build pipeline
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf)
            ])

            # 4. Cross-validate
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                agg_metrics, oof_true, oof_probs = cross_validate_model(
                    pipeline, X, y, n_splits=5
                )

            # 5. Log all CV metrics (mean + std)
            for metric_key, metric_val in agg_metrics.items():
                mlflow.log_metric(f"cv_{metric_key}", metric_val)

            log.info(
                f"  [{name}] CV Summary → "
                f"LogLoss={agg_metrics['log_loss']:.4f}±{agg_metrics['log_loss_std']:.4f}  "
                f"AUC={agg_metrics['roc_auc']:.4f}±{agg_metrics['roc_auc_std']:.4f}  "
                f"F1={agg_metrics['f1']:.4f}"
            )

            # 6. Fit final model on ALL data
            log.info(f"  [{name}] Fitting final model on full dataset...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X, y)

            # 7. Log all diagnostic plots and SHAP artifacts
            log_all_artifacts(
                pipeline, X, oof_true, oof_probs,
                model_name=name, artifact_dir=run_artifact_dir
            )

            # 8. Log the sklearn pipeline to MLflow (per-model registry)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=f"CricketIQ_{name}",
                metadata={"model_type": name, "cv_log_loss": agg_metrics["log_loss"]}
            )

            # 9. Track champion
            if agg_metrics["log_loss"] < best_log_loss:
                best_log_loss = agg_metrics["log_loss"]
                best_model_name = name
                best_pipeline = pipeline
                best_metrics = agg_metrics

            all_results[name] = {
                "cv_log_loss": agg_metrics["log_loss"],
                "cv_roc_auc": agg_metrics["roc_auc"],
                "cv_accuracy": agg_metrics["accuracy"],
                "cv_f1": agg_metrics["f1"],
                "cv_precision": agg_metrics["precision"],
                "cv_recall": agg_metrics["recall"],
            }

    # ── Register Champion Model ────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"🏆 Champion: {best_model_name}  (Log-Loss: {best_log_loss:.4f})")

    with mlflow.start_run(run_name=f"Champion_{best_model_name}_{run_timestamp}"):
        mlflow.set_tag("run_type", "champion_registration")

        # Log all champion metrics
        for metric_key, metric_val in best_metrics.items():
            mlflow.log_metric(f"champion_cv_{metric_key}", metric_val)

        # Register to Model Registry with Production alias
        log_champion_to_registry(best_model_name, best_pipeline, best_metrics)

        # Save locally too
        champion_path = os.path.join(models_dir, "champion_model.pkl")
        with open(champion_path, "wb") as f:
            pickle.dump({
                "name": best_model_name,
                "pipeline": best_pipeline,
                "features": list(X.columns),
                "cv_metrics": best_metrics,
                "trained_at": run_timestamp
            }, f)
        mlflow.log_artifact(champion_path, "model")
        log.info(f"Champion saved to {champion_path}")

        # ── Write experiment_summary.json ──────────────────────────────────
        summary = {
            "run_timestamp": run_timestamp,
            "champion_model": best_model_name,
            "champion_cv_log_loss": best_log_loss,
            "champion_cv_roc_auc": best_metrics.get("roc_auc"),
            "champion_cv_f1": best_metrics.get("f1"),
            "feature_count": len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES),
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "training_dataset": parquet_path,
            "all_model_results": all_results
        }
        summary_path = os.path.join(models_dir, "experiment_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path, "experiment")
        log.info(f"Experiment summary: {summary_path}")

    # ── Final console summary ──────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("📊 EXPERIMENT RESULTS SUMMARY")
    log.info("=" * 60)
    header = f"{'Model':<22} {'LogLoss':>10} {'AUC':>8} {'F1':>8} {'Accuracy':>10}"
    log.info(header)
    log.info("-" * 60)
    for model_name_r, r in all_results.items():
        champion_tag = " 🏆" if model_name_r == best_model_name else ""
        row = (
            f"{model_name_r:<22} "
            f"{r['cv_log_loss']:>10.4f} "
            f"{r['cv_roc_auc']:>8.4f} "
            f"{r['cv_f1']:>8.4f} "
            f"{r['cv_accuracy']:>10.4f}"
            f"{champion_tag}"
        )
        log.info(row)
    log.info("=" * 60)
    log.info("View results at: http://127.0.0.1:5000")

    return best_pipeline, best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_config()
    pq_path = str(resolve_path(cfg["paths"]["training_dataset"]))
    models_dir = str(resolve_path(cfg["paths"]["models_dir"]))
    train_and_register_champion(pq_path, models_dir)
