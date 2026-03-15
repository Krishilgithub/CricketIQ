"""
src/models/mlflow_tracking.py
─────────────────────────────
Production-grade MLflow tracking helpers for the CricketIQ match prediction pipeline.
Logs: experiment config, dataset metadata, preprocessing params, model hyperparams,
      feature analysis artifacts, data quality reports, and model registry management.
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from src.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_mlflow(
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "cricketiq_match_prediction"
) -> str:
    """
    Configure MLflow tracking URI and set/create the experiment.
    Returns the experiment ID.
    """
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(
            experiment_name,
            tags={"project": "CricketIQ", "task": "T20_win_prediction"}
        )
        log.info(f"Created MLflow experiment: {experiment_name} (ID: {exp_id})")
    else:
        exp_id = exp.experiment_id
        log.info(f"Using existing MLflow experiment: {experiment_name} (ID: {exp_id})")

    mlflow.set_experiment(experiment_name)
    return exp_id


# ─────────────────────────────────────────────────────────────────────────────
# DATASET METADATA
# ─────────────────────────────────────────────────────────────────────────────

def log_dataset_metadata(df: pd.DataFrame, dataset_path: str, target_col: str):
    """
    Log all dataset metadata as MLflow params:
    - Row count, feature count, date range, class balance, dataset path
    """
    n_rows, n_cols = df.shape
    n_features = n_cols - 1  # exclude target

    mlflow.log_param("dataset_path", dataset_path)
    mlflow.log_param("dataset_n_rows", n_rows)
    mlflow.log_param("dataset_n_features", n_features)

    # Date range (if available)
    if "match_date" in df.columns:
        try:
            dates = pd.to_datetime(df["match_date"])
            mlflow.log_param("dataset_date_range_start", str(dates.min().date()))
            mlflow.log_param("dataset_date_range_end", str(dates.max().date()))
        except Exception:
            pass

    # Class balance
    if target_col in df.columns:
        class_balance = float(df[target_col].mean())
        mlflow.log_param("dataset_class_balance_team1_wins", round(class_balance, 4))
        mlflow.log_metric("dataset_class_balance", round(class_balance, 4))

    log.info(f"Logged dataset metadata: {n_rows} rows, {n_features} features")


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def log_preprocessing_params(numerical_features: list, categorical_features: list):
    """Log all preprocessing configuration as MLflow params."""
    mlflow.log_param("preprocessor_numeric_imputer", "median")
    mlflow.log_param("preprocessor_numeric_scaler", "StandardScaler")
    mlflow.log_param("preprocessor_categorical_imputer", "most_frequent")
    mlflow.log_param("preprocessor_categorical_encoder", "OneHotEncoder_handle_unknown_ignore")
    mlflow.log_param("n_numerical_features", len(numerical_features))
    mlflow.log_param("n_categorical_features", len(categorical_features))
    mlflow.log_param("n_features_total", len(numerical_features) + len(categorical_features))
    mlflow.log_param("cv_strategy", "TimeSeriesSplit")
    log.info("Logged preprocessing params")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def log_model_params(model_name: str, clf):
    """
    Extract and log all hyperparameters for each candidate model.
    Handles: LogisticRegression, RandomForest, XGBoost, LightGBM.
    """
    mlflow.log_param("model_type", model_name)
    mlflow.log_param("random_state", 42)

    params = clf.get_params() if hasattr(clf, "get_params") else {}

    # Common params across all models
    common_keys = [
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "num_leaves", "min_samples_split",
        "C", "solver", "max_iter", "eval_metric",
        "reg_alpha", "reg_lambda", "min_child_weight"
    ]
    for key in common_keys:
        if key in params:
            mlflow.log_param(key, params[key])

    log.info(f"Logged hyperparams for {model_name}: {len(params)} params found")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE LIST ARTIFACT
# ─────────────────────────────────────────────────────────────────────────────

def log_feature_list_artifact(
    numerical_features: list,
    categorical_features: list,
    artifact_dir: str
):
    """Save the feature list as a JSON artifact for reproducibility."""
    os.makedirs(artifact_dir, exist_ok=True)
    feature_info = {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "all_features": numerical_features + categorical_features,
        "n_total": len(numerical_features) + len(categorical_features),
        "n_numerical": len(numerical_features),
        "n_categorical": len(categorical_features)
    }
    path = os.path.join(artifact_dir, "feature_list.json")
    with open(path, "w") as f:
        json.dump(feature_info, f, indent=2)
    mlflow.log_artifact(path, "feature_analysis")
    log.info(f"Logged feature list artifact: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY ARTIFACT
# ─────────────────────────────────────────────────────────────────────────────

def log_data_quality_artifact(X: pd.DataFrame, artifact_dir: str):
    """
    Generate and log a data quality report covering:
    - Missing value % per feature
    - Zero-variance features
    - Outlier counts (IQR method) for numerical features
    """
    os.makedirs(artifact_dir, exist_ok=True)

    rows = []
    for col in X.columns:
        missing_pct = round(X[col].isna().mean() * 100, 2)
        n_unique = X[col].nunique()
        zero_variance = (n_unique <= 1)

        outlier_count = 0
        if pd.api.types.is_numeric_dtype(X[col]):
            q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            iqr = q3 - q1
            outlier_count = int(((X[col] < q1 - 1.5 * iqr) | (X[col] > q3 + 1.5 * iqr)).sum())

        rows.append({
            "feature": col,
            "missing_pct": missing_pct,
            "n_unique": n_unique,
            "zero_variance": zero_variance,
            "outlier_count_iqr": outlier_count,
            "dtype": str(X[col].dtype)
        })

    dq_df = pd.DataFrame(rows)
    path = os.path.join(artifact_dir, "data_quality_report.csv")
    dq_df.to_csv(path, index=False)
    mlflow.log_artifact(path, "feature_analysis")

    # Log summary metrics
    mlflow.log_metric("dq_features_with_missing", int((dq_df["missing_pct"] > 0).sum()))
    mlflow.log_metric("dq_zero_variance_features", int(dq_df["zero_variance"].sum()))
    mlflow.log_metric("dq_total_outliers_iqr", int(dq_df["outlier_count_iqr"].sum()))

    log.info(f"Logged data quality report: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def log_feature_correlation_artifact(X: pd.DataFrame, artifact_dir: str):
    """Generate and log a Pearson correlation heatmap for numerical features."""
    os.makedirs(artifact_dir, exist_ok=True)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return

    corr_matrix = X[num_cols].corr()

    fig, ax = plt.subplots(figsize=(max(8, len(num_cols)), max(6, len(num_cols) - 2)))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5,
        annot_kws={"size": 8}, ax=ax
    )
    ax.set_title("Feature Correlation Heatmap (Numerical Features)", fontsize=13, pad=12)
    plt.tight_layout()

    path = os.path.join(artifact_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(path, "feature_analysis")
    log.info(f"Logged correlation heatmap: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DISTRIBUTION HISTOGRAMS
# ─────────────────────────────────────────────────────────────────────────────

def log_feature_distribution_artifact(X: pd.DataFrame, artifact_dir: str):
    """Generate and log per-feature distribution histograms for numerical features."""
    os.makedirs(artifact_dir, exist_ok=True)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return

    n_cols_grid = 3
    n_rows_grid = (len(num_cols) + n_cols_grid - 1) // n_cols_grid
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(18, n_rows_grid * 3.5))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        data = X[col].dropna()
        ax.hist(data, bins=30, color="#2563eb", edgecolor="white", alpha=0.8)
        ax.axvline(data.mean(), color="#ef4444", linestyle="--", linewidth=1.5, label=f"μ={data.mean():.2f}")
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    # Hide extra axes
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions — Numerical Features", fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(artifact_dir, "feature_distributions.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(path, "feature_analysis")
    log.info(f"Logged feature distribution plots: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY — CHAMPION PROMOTION
# ─────────────────────────────────────────────────────────────────────────────

def log_champion_to_registry(
    model_name: str,
    pipeline,
    run_metrics: dict,
    artifact_path: str = "model"
):
    """
    Log the champion model to MLflow Model Registry under 'CricketIQ_Match_Predictor'
    and set the Production alias with full metric tags.
    """
    registered_name = "CricketIQ_Match_Predictor"

    # Log the sklearn pipeline
    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path=artifact_path,
        registered_model_name=registered_name,
        metadata={
            "champion_model_type": model_name,
            "cv_log_loss": round(run_metrics.get("log_loss", -1), 4),
            "cv_roc_auc": round(run_metrics.get("roc_auc", -1), 4),
        }
    )

    # Tag the run with champion info
    mlflow.set_tag("champion", "true")
    mlflow.set_tag("champion_model_type", model_name)
    mlflow.set_tag("cv_log_loss", round(run_metrics.get("log_loss", -1), 4))
    mlflow.set_tag("cv_roc_auc", round(run_metrics.get("roc_auc", -1), 4))
    mlflow.set_tag("cricket_iq_version", "v1.0")

    # Set Production alias on the newly registered version
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{registered_name}'")
        if versions:
            latest_version = sorted(versions, key=lambda v: int(v.version))[-1].version
            client.set_registered_model_alias(registered_name, "Production", latest_version)
            log.info(f"Set 'Production' alias on {registered_name} v{latest_version}")
    except Exception as e:
        log.warning(f"Could not set Production alias: {e}")

    log.info(f"Champion '{model_name}' registered as '{registered_name}'")
    return model_info
