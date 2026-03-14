"""
src/models/evaluate_models.py
─────────────────────────────
Functions to evaluate model performance and generate artifact plots 
(confusion matrix, feature importance) for MLflow logging.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score, confusion_matrix

def evaluate_predictions(y_true, y_probs) -> dict:
    """Calculates core classification metrics."""
    y_pred = (y_probs >= 0.5).astype(int)
    return {
        "log_loss": log_loss(y_true, y_probs),
        "brier_score": brier_score_loss(y_true, y_probs),
        "roc_auc": roc_auc_score(y_true, y_probs),
        "accuracy": accuracy_score(y_true, y_pred)
    }

def generate_confusion_matrix_plot(y_true, y_probs, save_path: str) -> str:
    """Generates and saves a confusion matrix plot."""
    y_pred = (y_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix (T20 Match Prediction)")
    plt.ylabel('Actual: Team 1 Win')
    plt.xlabel('Predicted: Team 1 Win')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return save_path
