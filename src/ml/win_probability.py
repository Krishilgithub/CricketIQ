"""
Live Win Probability — Ball-by-ball win probability prediction.
"""

import sys
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import ML_MODEL_DIR, RANDOM_STATE, TEST_SIZE


def build_ball_level_features(conn, source: str = "t20i") -> pd.DataFrame:
    """Build ball-by-ball features for win probability model."""
    df = conn.execute(f"""
        SELECT
            d.match_id, d.innings, d.over, d.ball,
            d.cumulative_runs, d.cumulative_wickets,
            d.total_runs AS ball_runs, d.run_rate,
            d.phase, d.is_wicket, d.is_boundary_four, d.is_boundary_six,
            m.winner, d.batting_team, d.bowling_team
        FROM silver.deliveries d
        JOIN silver.matches m ON d.match_id = m.match_id
        WHERE d.source = '{source}'
          AND m.winner IS NOT NULL
          AND d.innings IN (1, 2)
    """).df()

    if df.empty:
        return pd.DataFrame()

    # Get innings totals for target calculation
    innings_totals = conn.execute(f"""
        SELECT match_id, innings, SUM(total_runs) AS innings_total
        FROM silver.deliveries
        WHERE source = '{source}'
        GROUP BY match_id, innings
    """).df()

    totals_map = {}
    for _, row in innings_totals.iterrows():
        totals_map[(row["match_id"], row["innings"])] = row["innings_total"]

    # Build features
    records = []
    for _, row in df.iterrows():
        match_id = row["match_id"]
        innings = row["innings"]

        target_total = totals_map.get((match_id, 1), 150)

        feat = {
            "match_id": match_id,
            "innings": innings,
            "over": row["over"],
            "ball": row["ball"],
            "runs_scored": row["cumulative_runs"],
            "wickets_lost": row["cumulative_wickets"],
            "current_rr": row["run_rate"] if row["run_rate"] else 0,
            "overs_remaining": 20 - row["over"] - 1,
            "balls_remaining": max(0, (20 - row["over"] - 1) * 6 + (6 - row["ball"])),
            "wickets_remaining": 10 - row["cumulative_wickets"],
        }

        if innings == 2:
            feat["target"] = target_total + 1
            feat["runs_needed"] = max(0, target_total + 1 - row["cumulative_runs"])
            feat["required_rate"] = (
                feat["runs_needed"] / max(feat["overs_remaining"] + 1, 0.1) * 6
                if feat["balls_remaining"] > 0 else 999
            )
        else:
            feat["target"] = 0
            feat["runs_needed"] = 0
            feat["required_rate"] = 0

        # Phase encoding
        feat["is_powerplay"] = 1 if row["phase"] == "powerplay" else 0
        feat["is_middle"] = 1 if row["phase"] == "middle" else 0
        feat["is_death"] = 1 if row["phase"] == "death" else 0

        # Target: batting team won
        feat["batting_team_won"] = 1 if row["batting_team"] == row["winner"] else 0

        records.append(feat)

    return pd.DataFrame(records)


def train_win_probability_model(ball_df: pd.DataFrame) -> dict:
    """Train ball-level win probability model."""
    feature_cols = [
        "innings", "over", "runs_scored", "wickets_lost",
        "current_rr", "overs_remaining", "balls_remaining",
        "wickets_remaining", "runs_needed", "required_rate",
        "is_powerplay", "is_middle", "is_death",
    ]

    df = ball_df.dropna(subset=feature_cols + ["batting_team_won"]).copy()

    # Sample to manage memory (ball-level data is huge)
    if len(df) > 200000:
        df = df.sample(200000, random_state=RANDOM_STATE)

    X = df[feature_cols].values
    y = df["batting_team_won"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression (fast, interpretable)
    print("🔄 Training Win Probability model (Logistic Regression)...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    y_prob = lr.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    print(f"  ✅ Win Prob LR: Acc={acc:.4f}, AUC={auc:.4f}, LogLoss={ll:.4f}")

    # Try XGBoost
    xgb_results = None
    try:
        from xgboost import XGBClassifier
        print("🔄 Training Win Probability model (XGBoost)...")
        xgb = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss",
            use_label_encoder=False,
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_prob = xgb.predict_proba(X_test)[:, 1]
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_auc = roc_auc_score(y_test, xgb_prob)
        print(f"  ✅ Win Prob XGB: Acc={xgb_acc:.4f}, AUC={xgb_auc:.4f}")
        xgb_results = {"model": xgb, "accuracy": xgb_acc, "auc": xgb_auc}
    except ImportError:
        pass

    # Save best
    ML_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model = lr
    best_name = "Logistic Regression"
    if xgb_results and xgb_results["auc"] > auc:
        best_model = xgb_results["model"]
        best_name = "XGBoost"

    with open(ML_MODEL_DIR / "win_probability_model.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "model_name": best_name,
        }, f)

    return {
        "lr": {"model": lr, "accuracy": acc, "auc": auc, "log_loss": ll},
        "xgb": xgb_results,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "best_model": best_name,
        "y_test": y_test,
        "y_prob": y_prob,
    }


if __name__ == "__main__":
    from src.warehouse.schema import get_connection

    conn = get_connection()
    print("Building ball-level features...")
    ball_df = build_ball_level_features(conn, source="t20i")
    conn.close()

    if not ball_df.empty:
        print(f"Built {len(ball_df)} ball-level feature vectors.")
        results = train_win_probability_model(ball_df)
