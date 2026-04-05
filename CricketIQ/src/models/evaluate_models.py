"""
src/models/evaluate_models.py
─────────────────────────────
Production-grade model evaluation module for the CricketIQ match prediction pipeline.
Generates and logs all diagnostic plots and metrics to MLflow:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
  - Feature Importance (tree-based models)
  - Learning Curve
  - SHAP Summary + Beeswarm + Top Dependence plots
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    log_loss, brier_score_loss, roc_auc_score, accuracy_score,
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve

import mlflow

from src.logger import get_logger
log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STYLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
STYLE = {
    "bg": "#0f172a",
    "fg": "#e2e8f0",
    "primary": "#3b82f6",
    "secondary": "#f59e0b",
    "success": "#22c55e",
    "danger": "#ef4444",
    "grid": "#334155",
}

def _set_dark_style():
    plt.rcParams.update({
        "figure.facecolor": STYLE["bg"],
        "axes.facecolor": STYLE["bg"],
        "text.color": STYLE["fg"],
        "axes.labelcolor": STYLE["fg"],
        "axes.edgecolor": STYLE["grid"],
        "xtick.color": STYLE["fg"],
        "ytick.color": STYLE["fg"],
        "grid.color": STYLE["grid"],
        "font.size": 10,
    })

# ─────────────────────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(y_true, y_probs, threshold: float = 0.5) -> dict:
    """
    Compute full classification metrics for a set of predictions.
    Returns: log_loss, brier_score, roc_auc, accuracy, precision, recall, f1, ap_score,
             and raw confusion matrix values (tp, tn, fp, fn).
    """
    y_pred = (np.array(y_probs) >= threshold).astype(int)
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "log_loss":      round(log_loss(y_true, y_probs), 5),
        "brier_score":   round(brier_score_loss(y_true, y_probs), 5),
        "roc_auc":       round(roc_auc_score(y_true, y_probs), 5),
        "accuracy":      round(accuracy_score(y_true, y_pred), 5),
        "precision":     round(precision_score(y_true, y_pred, zero_division=0), 5),
        "recall":        round(recall_score(y_true, y_pred, zero_division=0), 5),
        "f1":            round(f1_score(y_true, y_pred, zero_division=0), 5),
        "ap_score":      round(average_precision_score(y_true, y_probs), 5),
        # Raw confusion matrix values — visible as metrics in MLflow UI
        "tp":  int(tp),
        "tn":  int(tn),
        "fp":  int(fp),
        "fn":  int(fn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def generate_confusion_matrix_plot(y_true, y_probs, save_path: str) -> str:
    """Generates, saves, and logs a styled confusion matrix to MLflow."""
    _set_dark_style()
    y_pred = (np.array(y_probs) >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=["Team 2 Wins", "Team 1 Wins"],
        yticklabels=["Team 2 Wins", "Team 1 Wins"],
        annot_kws={"size": 16, "weight": "bold"}, ax=ax
    )
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(
        f"Confusion Matrix — Accuracy: {acc:.2%}\n"
        f"TP={tp}  TN={tn}  FP={fp}  FN={fn}",
        color=STYLE["fg"], fontsize=11, pad=10
    )
    ax.set_ylabel("Actual", color=STYLE["fg"])
    ax.set_xlabel("Predicted", color=STYLE["fg"])
    plt.tight_layout()

    # Save to disk
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])

    # Also log natively to MLflow so it's visible as an inline image in the UI
    try:
        mlflow.log_figure(fig, "evaluation_plots/confusion_matrix.png")
    except Exception:
        pass  # mlflow.log_figure needs an active run; _safe_log handles the artifact path anyway

    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROC CURVE
# ─────────────────────────────────────────────────────────────────────────────

def generate_roc_curve_plot(y_true, y_probs, model_name: str, save_path: str) -> str:
    """Generates and saves a ROC curve with AUC annotation."""
    _set_dark_style()
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(fpr, tpr, color=STYLE["primary"], lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color=STYLE["grid"], lw=1.5, linestyle="--", label="Random Baseline")
    ax.fill_between(fpr, tpr, alpha=0.08, color=STYLE["primary"])
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", color=STYLE["fg"])
    ax.set_ylabel("True Positive Rate", color=STYLE["fg"])
    ax.set_title(f"ROC Curve — {model_name}", color=STYLE["fg"], fontsize=13)
    ax.legend(loc="lower right", framealpha=0.2, labelcolor=STYLE["fg"])
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRECISION-RECALL CURVE
# ─────────────────────────────────────────────────────────────────────────────

def generate_pr_curve_plot(y_true, y_probs, model_name: str, save_path: str) -> str:
    """Generates and saves a Precision-Recall curve with AP score annotation."""
    _set_dark_style()
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    baseline = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(recall, precision, color=STYLE["secondary"], lw=2.5,
            label=f"PR Curve (AP = {ap:.4f})")
    ax.axhline(y=baseline, color=STYLE["grid"], linestyle="--", lw=1.5,
               label=f"Baseline (prevalence={baseline:.2f})")
    ax.fill_between(recall, precision, baseline, alpha=0.08, color=STYLE["secondary"])
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", color=STYLE["fg"])
    ax.set_ylabel("Precision", color=STYLE["fg"])
    ax.set_title(f"Precision-Recall Curve — {model_name}", color=STYLE["fg"], fontsize=13)
    ax.legend(framealpha=0.2, labelcolor=STYLE["fg"])
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def generate_feature_importance_plot(
    pipeline, feature_names: list, model_name: str, save_path: str, top_n: int = 20
) -> str:
    """
    Extract feature importances from tree-based models (RF, XGBoost, LightGBM).
    After OHE, reconstructs expanded feature names from the pipeline's transformer.
    """
    _set_dark_style()
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        clf = pipeline.named_steps.get("classifier")

        if preprocessor is None or clf is None:
            log.warning("Could not find preprocessor/classifier in pipeline.")
            return None

        # Get expanded column names from ColumnTransformer
        if hasattr(preprocessor, "get_feature_names_out"):
            expanded_names = list(preprocessor.get_feature_names_out())
        else:
            expanded_names = feature_names  # Fallback

        # Extract importances
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            log.warning(f"{model_name} does not expose feature_importances_ or coef_.")
            return None

        n = min(len(importances), len(expanded_names))
        fi_df = pd.DataFrame({
            "feature": expanded_names[:n],
            "importance": importances[:n]
        }).sort_values("importance", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(9, max(5, len(fi_df) * 0.4 + 1)))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_df)))[::-1]
        ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color=colors)
        ax.set_xlabel("Importance Score", color=STYLE["fg"])
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}", color=STYLE["fg"], fontsize=13)
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
        plt.close()
        log.info(f"Feature importance plot saved: {save_path}")
        return save_path

    except Exception as e:
        log.warning(f"Could not generate feature importance for {model_name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. LEARNING CURVE
# ─────────────────────────────────────────────────────────────────────────────

def generate_learning_curve_plot(pipeline, X, y, model_name: str, save_path: str) -> str:
    """
    Generate a sklearn learning curve plot showing training vs validation accuracy
    as training set size increases — useful for diagnosing over/underfitting.
    """
    _set_dark_style()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_sizes, train_scores, val_scores = learning_curve(
                pipeline, X, y,
                cv=3,  # Fast CV for the curve
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 8),
                scoring="roc_auc"
            )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, train_mean, "o-", color=STYLE["primary"], lw=2, label="Training AUC")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color=STYLE["primary"])
        ax.plot(train_sizes, val_mean, "s-", color=STYLE["secondary"], lw=2, label="Validation AUC")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color=STYLE["secondary"])
        ax.set_xlabel("Training Set Size", color=STYLE["fg"])
        ax.set_ylabel("ROC-AUC", color=STYLE["fg"])
        ax.set_title(f"Learning Curve — {model_name}", color=STYLE["fg"], fontsize=13)
        ax.legend(framealpha=0.2, labelcolor=STYLE["fg"])
        ax.grid(True, alpha=0.2)
        ax.set_ylim([0.4, 1.02])
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
        plt.close()
        log.info(f"Learning curve plot saved: {save_path}")
        return save_path

    except Exception as e:
        log.warning(f"Could not generate learning curve for {model_name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. SHAP EXPLAINABILITY PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_shap_plots(pipeline, X: pd.DataFrame, model_name: str, artifact_dir: str) -> list:
    """
    Generate SHAP summary, beeswarm, and top-2 dependence plots.
    Returns list of saved paths (empty list if shap not available or model unsupported).
    """
    # Lazy import — avoids llvmlite/numba crash at module import time
    try:
        import shap
    except Exception as e:
        log.warning(f"shap not available: {e}. Skipping SHAP plots.")
        return []

    saved_paths = []
    try:
        # Transform X through preprocessor to get the model's feature space
        preprocessor = pipeline.named_steps.get("preprocessor")
        clf = pipeline.named_steps.get("classifier")
        if preprocessor is None or clf is None:
            return []

        X_transformed = preprocessor.transform(X)
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = list(preprocessor.get_feature_names_out())
        else:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        # Convert to DataFrame for SHAP
        X_df = pd.DataFrame(X_transformed, columns=feature_names)

        # Sample for speed (SHAP is slow on large datasets)
        sample_size = min(500, len(X_df))
        X_sample = X_df.sample(sample_size, random_state=42)

        # Choose SHAP explainer based on model type
        clf_name = type(clf).__name__
        if clf_name in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
                         "GradientBoostingClassifier", "ExtraTreesClassifier"):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_sample)
            # For binary classifiers that return list (RF), take class 1
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # Use KernelExplainer for LogisticRegression etc.
            background = shap.kmeans(X_sample, min(50, len(X_sample)))
            explainer = shap.KernelExplainer(
                lambda x: clf.predict_proba(x)[:, 1], background
            )
            shap_values = explainer.shap_values(X_sample[:100], nsamples=100)

        os.makedirs(artifact_dir, exist_ok=True)

        # ── SHAP Summary Bar Plot ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar",
                          feature_names=feature_names, show=False,
                          max_display=15, color=STYLE["primary"])
        plt.title(f"SHAP Feature Importance (Mean |SHAP|) — {model_name}",
                  color="#1a1a2e", fontsize=12)
        plt.tight_layout()
        path_bar = os.path.join(artifact_dir, "shap_summary_bar.png")
        plt.savefig(path_bar, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths.append(path_bar)

        # ── SHAP Beeswarm Plot ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample,
                          feature_names=feature_names, show=False,
                          max_display=15, color_bar=True)
        plt.title(f"SHAP Beeswarm Plot — {model_name}", color="#1a1a2e", fontsize=12)
        plt.tight_layout()
        path_bees = os.path.join(artifact_dir, "shap_beeswarm.png")
        plt.savefig(path_bees, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths.append(path_bees)

        # ── SHAP Dependence Plot for top 2 features ────────────────────────
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top2_idx = np.argsort(mean_abs_shap)[-2:][::-1]
        for idx in top2_idx:
            feat_name = feature_names[idx]
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(idx, shap_values, X_sample,
                                 feature_names=feature_names, show=False, ax=ax)
            plt.title(f"SHAP Dependence: {feat_name} — {model_name}",
                      color="#1a1a2e", fontsize=11)
            plt.tight_layout()
            safe_name = feat_name.replace("/", "_").replace(" ", "_")
            dep_path = os.path.join(artifact_dir, f"shap_dependence_{safe_name}.png")
            plt.savefig(dep_path, dpi=130, bbox_inches="tight")
            plt.close()
            saved_paths.append(dep_path)

        log.info(f"Generated {len(saved_paths)} SHAP plots for {model_name}")

    except Exception as e:
        log.warning(f"SHAP plot generation failed for {model_name}: {e}")

    return saved_paths


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR — LOG ALL ARTIFACTS FOR ONE MODEL RUN
# ─────────────────────────────────────────────────────────────────────────────

def log_all_artifacts(
    pipeline,
    X: pd.DataFrame,
    oof_y_true: np.ndarray,
    oof_y_probs: np.ndarray,
    model_name: str,
    artifact_dir: str
):
    """
    Orchestrate generation and MLflow logging of ALL diagnostic artifacts:
      1. Confusion Matrix
      2. ROC Curve
      3. PR Curve
      4. Feature Importance
      5. Learning Curve
      6. SHAP Summary + Beeswarm + Dependence plots

    All PNG files are logged to MLflow under 'evaluation_plots/' and 'shap_plots/'.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    feature_names = list(X.columns)

    def _safe_log(func, *args, mlflow_folder="evaluation_plots", **kwargs):
        try:
            path = func(*args, **kwargs)
            if path and os.path.exists(path):
                mlflow.log_artifact(path, mlflow_folder)
                log.info(f"  ✓ Logged artifact: {os.path.basename(path)}")
        except Exception as e:
            log.warning(f"  ✗ Failed to log artifact from {func.__name__}: {e}")

    log.info(f"Generating all evaluation artifacts for [{model_name}]...")

    # 1. Confusion Matrix
    _safe_log(
        generate_confusion_matrix_plot,
        oof_y_true, oof_y_probs,
        os.path.join(artifact_dir, "confusion_matrix.png")
    )

    # 2. ROC Curve
    _safe_log(
        generate_roc_curve_plot,
        oof_y_true, oof_y_probs, model_name,
        os.path.join(artifact_dir, "roc_curve.png")
    )

    # 3. PR Curve
    _safe_log(
        generate_pr_curve_plot,
        oof_y_true, oof_y_probs, model_name,
        os.path.join(artifact_dir, "pr_curve.png")
    )

    # 4. Feature Importance
    _safe_log(
        generate_feature_importance_plot,
        pipeline, feature_names, model_name,
        os.path.join(artifact_dir, "feature_importance.png")
    )

    # 5. Learning Curve
    _safe_log(
        generate_learning_curve_plot,
        pipeline, X, oof_y_true, model_name,
        os.path.join(artifact_dir, "learning_curve.png")
    )

    # 6. SHAP Plots (logged separately under shap_plots/)
    try:
        shap_dir = os.path.join(artifact_dir, "shap")
        shap_paths = generate_shap_plots(pipeline, X, model_name, shap_dir)
        for path in shap_paths:
            if os.path.exists(path):
                mlflow.log_artifact(path, "shap_plots")
    except Exception as e:
        log.warning(f"SHAP logging failed: {e}")

    log.info(f"All artifacts logged for [{model_name}]")
