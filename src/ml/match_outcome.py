"""
Match Outcome Classification — Predict match winner (Team1 vs Team2).

Models: Logistic Regression, Random Forest, XGBoost, LightGBM.
"""

import sys
import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import ML_MODEL_DIR, RANDOM_STATE, TEST_SIZE


FEATURE_COLS = [
    "toss_winner_is_team1", "toss_elected_bat", "is_world_cup",
    "team1_recent_win_rate", "team2_recent_win_rate",
    "team1_avg_score", "team2_avg_score",
    "team1_avg_rr", "team2_avg_rr",
    "team1_avg_pp_rr", "team2_avg_pp_rr",
    "team1_avg_death_rr", "team2_avg_death_rr",
    "team1_avg_boundaries", "team2_avg_boundaries",
    "h2h_team1_win_rate", "h2h_total_matches",
    "venue_avg_1st_score", "venue_avg_2nd_score", "venue_matches",
    "win_rate_diff", "avg_score_diff", "avg_rr_diff",
]


def train_match_outcome_models(features_df: pd.DataFrame) -> dict:
    """Train and evaluate multiple classifiers."""
    df = features_df.dropna(subset=FEATURE_COLS + ["target"]).copy()

    if len(df) < 50:
        print("⚠️ Not enough data for training.")
        return {}

    X = df[FEATURE_COLS].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE,
        ),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss",
            use_label_encoder=False,
        )
    except ImportError:
        print("⚠️ XGBoost not installed, skipping.")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=-1,
        )
    except ImportError:
        print("⚠️ LightGBM not installed, skipping.")

    results = {}
    best_score = 0
    best_model_name = None

    for name, model in models.items():
        print(f"\n🔄 Training {name}...")

        if name in ["Logistic Regression"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Cross-validation
        if name in ["Logistic Regression"]:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": roc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": cm,
            "classification_report": report,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        print(f"  ✅ {name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={roc:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

        if acc > best_score:
            best_score = acc
            best_model_name = name

    # Save best model
    ML_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best = results[best_model_name]
    with open(ML_MODEL_DIR / "match_outcome_model.pkl", "wb") as f:
        pickle.dump({
            "model": best["model"],
            "scaler": scaler,
            "feature_cols": FEATURE_COLS,
            "model_name": best_model_name,
        }, f)

    results["_scaler"] = scaler
    results["_best_model"] = best_model_name
    results["_feature_cols"] = FEATURE_COLS

    print(f"\n🏆 Best model: {best_model_name} (Acc={best_score:.4f})")
    return results


def get_feature_importance(results: dict) -> pd.DataFrame:
    """Extract feature importance from tree-based models."""
    importances = []
    for name, res in results.items():
        if name.startswith("_"):
            continue
        model = res["model"]
        if hasattr(model, "feature_importances_"):
            for feat, imp in zip(FEATURE_COLS, model.feature_importances_):
                importances.append({"Model": name, "Feature": feat, "Importance": imp})

    return pd.DataFrame(importances)


if __name__ == "__main__":
    from src.warehouse.schema import get_connection
    from src.ml.feature_engineering import build_match_features

    conn = get_connection()
    print("Building features...")
    features = build_match_features(conn, source="t20i")
    conn.close()

    if not features.empty:
        print(f"Built {len(features)} match feature vectors.")
        results = train_match_outcome_models(features)
    else:
        print("❌ No features built.")
