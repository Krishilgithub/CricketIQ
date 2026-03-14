"""
src/models/train_models.py
──────────────────────────
Trains multiple ML models combining preprocessing pipelines and raw features.
Evaluates using TimeSeriesSplit and logs all metrics, parameters, and models 
to MLflow Model Registry.
"""

import os
import pickle
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

import mlflow

from src.config import get_config, resolve_path
from src.logger import get_logger
from src.features.data_preprocessing import get_preprocessor, get_training_columns
from src.models.mlflow_tracking import setup_mlflow
from src.models.evaluate_models import evaluate_predictions, generate_confusion_matrix_plot

log = get_logger(__name__)

def load_data(parquet_path: str):
    """Load pre-engineered features and targets from Parquet."""
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("match_date").reset_index(drop=True)
    
    X_cols, y_col = get_training_columns()
    X = df[X_cols]
    y = df[y_col].astype(int)
    log.info(f"Loaded {len(df)} rows. Features: {len(X_cols)}")
    log.info(f"Class balance (label 1): {y.mean():.2%}")
    return X, y

def get_candidate_models() -> Dict[str, Any]:
    """Define base classifier definitions."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=0.5, solver="lbfgs", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
    }
    
    if xgb:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05, 
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, use_label_encoder=False
        )
        
    if lgb:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=150, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        
    return models

from sklearn.base import clone

def cross_validate_model(model_pipeline, X: pd.DataFrame, y: pd.Series, n_splits=5):
    """Evaluate pipeline using Time Series Split to prevent data leakage."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    
    # Store complete OOF (Out-of-Fold) predictions for overall plot
    oof_y_true, oof_y_probs = [], []

    for train_idx, val_idx in tscv.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Clone pipeline to avoid feature size mismatch when OneHotEncoder expands
        fold_pipeline = clone(model_pipeline)
        fold_pipeline.fit(X_tr, y_tr)
        
        # Predict on validation set
        if hasattr(fold_pipeline, "predict_proba"):
            probs = fold_pipeline.predict_proba(X_val)[:, 1]
        else:
            probs = fold_pipeline.predict(X_val)
            
        metrics = evaluate_predictions(y_val, probs)
        metrics_list.append(metrics)
        
        oof_y_true.extend(y_val)
        oof_y_probs.extend(probs)
        
    # Aggregate metrics
    agg_metrics = {
        k: float(np.mean([m[k] for m in metrics_list])) 
        for k in metrics_list[0].keys()
    }
    return agg_metrics, np.array(oof_y_true), np.array(oof_y_probs)

def train_and_register_champion(parquet_path: str, models_dir: str):
    X, y = load_data(parquet_path)
    os.makedirs(models_dir, exist_ok=True)
    
    setup_mlflow()
    
    candidate_models = get_candidate_models()
    preprocessor = get_preprocessor()
    
    best_log_loss = float('inf')
    best_model_name = None
    best_pipeline = None
    
    log.info(f"Starting MLflow runs for {len(candidate_models)} candidate models.")
    
    for name, clf in candidate_models.items():
        with mlflow.start_run(run_name=f"CV_{name}"):
            log.info(f"Training and Evaluating [{name}]...")
            
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf)
            ])
            
            # 1. Perform TS Cross-Validation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                agg_metrics, oof_true, oof_probs = cross_validate_model(pipeline, X, y, n_splits=5)
            
            # 2. Log Metrics
            mlflow.log_metrics({f"cv_mean_{k}": v for k, v in agg_metrics.items()})
            log.info(f"  [{name}] CV Log Loss: {agg_metrics['log_loss']:.4f} | AUC: {agg_metrics['roc_auc']:.4f}")
            
            # 3. Fit on ALL data for final artifacts and model logging
            pipeline.fit(X, y)
            
            # 4. Generate & log artifacts
            cm_path = os.path.join(models_dir, f"{name}_cm.png")
            generate_confusion_matrix_plot(oof_true, oof_probs, cm_path)
            mlflow.log_artifact(cm_path, "evaluation_plots")
            
            # 5. Log Model Definition
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=f"CricketIQ_{name}"
            )
            
            # Track champion
            if agg_metrics["log_loss"] < best_log_loss:
                best_log_loss = agg_metrics["log_loss"]
                best_model_name = name
                best_pipeline = pipeline

    log.info("="*50)
    log.info(f"🏆 Champion Model Selected: {best_model_name}")
    log.info(f"   Log-Loss: {best_log_loss:.4f}")
    
    # Save champion locally to artifacts
    champion_path = os.path.join(models_dir, "champion_model.pkl")
    with open(champion_path, "wb") as f:
        pickle.dump({
            "name": best_model_name,
            "pipeline": best_pipeline,
            "features": list(X.columns)
        }, f)
        
    log.info(f"Champion Model fully trained and saved to {champion_path}")
    
    # Optional explicitly tag best model in registry
    # (By tracking standard names, we can load standard name later)

if __name__ == "__main__":
    cfg = get_config()
    db_path = str(resolve_path(cfg["paths"]["duckdb_path"]))
    pq_path = str(resolve_path(cfg["paths"]["training_dataset"]))
    models_dir = str(resolve_path(cfg["paths"]["models_dir"]))
    train_and_register_champion(pq_path, models_dir)
