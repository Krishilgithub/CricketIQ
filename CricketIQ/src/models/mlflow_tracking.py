"""
src/models/mlflow_tracking.py
─────────────────────────────
Utility to initialize and manage MLflow experiments and model registry logic.
"""

import os
import mlflow
from src.logger import get_logger

log = get_logger(__name__)

def setup_mlflow(tracking_uri: str = "file:./mlruns", experiment_name: str = "cricketiq_advanced_ml"):
    """
    Configure MLflow tracking URI and ensure the experiment exists.
    Returns the experiment ID.
    """
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        try:
            exp_id = mlflow.create_experiment(experiment_name)
            log.info(f"Created new MLflow experiment: {experiment_name} (ID: {exp_id})")
        except Exception as e:
            log.warning(f"Could not create MLflow experiment: {e}")
            exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    else:
        exp_id = exp.experiment_id
        log.info(f"Using existing MLflow experiment: {experiment_name} (ID: {exp_id})")
        
    mlflow.set_experiment(experiment_name)
    return exp_id
