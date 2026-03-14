"""
Score Prediction — Predict 1st innings total score.

Models: Linear Regression, Ridge, XGBoost Regressor.
"""

import sys
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import ML_MODEL_DIR, RANDOM_STATE, TEST_SIZE


def train_score_models(innings_df: pd.DataFrame) -> dict:
    """Train regression models to predict 1st innings total."""
    df = innings_df.copy()

    if len(df) < 50:
        print("⚠️ Not enough data for training.")
        return {}

    # Features for score prediction
    feature_cols = []

    # Venue average as a feature
    if "venue_avg" in df.columns:
        df["venue_avg"] = df["venue_avg"].fillna(df["total_runs"].mean())
        feature_cols.append("venue_avg")

    if "venue_matches" in df.columns:
        df["venue_matches"] = df["venue_matches"].fillna(0)
        feature_cols.append("venue_matches")

    if "toss_decision" in df.columns:
        df["elected_bat"] = (df["toss_decision"] == "bat").astype(int)
        feature_cols.append("elected_bat")

    # Use powerplay stats as early-innings predictor (simulating partial info)
    for col in ["pp_runs", "pp_run_rate", "pp_wickets"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            feature_cols.append(col)

    if not feature_cols:
        print("❌ No usable features found.")
        return {}

    X = df[feature_cols].values
    y = df["total_runs"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost Regressor"] = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE,
        )
    except ImportError:
        pass

    results = {}
    best_r2 = -999
    best_name = None

    for name, model in models.items():
        print(f"\n🔄 Training {name}...")

        if "XGBoost" in name:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "y_test": y_test,
            "y_pred": y_pred,
        }

        print(f"  ✅ {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name

    # Save best model
    ML_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(ML_MODEL_DIR / "score_prediction_model.pkl", "wb") as f:
        pickle.dump({
            "model": results[best_name]["model"],
            "scaler": scaler,
            "feature_cols": feature_cols,
            "model_name": best_name,
        }, f)

    results["_scaler"] = scaler
    results["_best_model"] = best_name
    results["_feature_cols"] = feature_cols

    print(f"\n🏆 Best model: {best_name} (R²={best_r2:.4f})")
    return results


if __name__ == "__main__":
    from src.warehouse.schema import get_connection
    from src.ml.feature_engineering import build_innings_features

    conn = get_connection()
    innings = build_innings_features(conn, source="t20i")
    conn.close()

    if not innings.empty:
        results = train_score_models(innings)
