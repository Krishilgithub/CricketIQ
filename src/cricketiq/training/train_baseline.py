from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _safe_metric(func, *args, **kwargs):
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return None


def _split_by_time(df: pd.DataFrame, date_col: str, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(by=[date_col, "match_id"], ascending=[True, True]).reset_index(drop=True)
    n = len(df_sorted)
    split_idx = int(n * (1.0 - test_size))
    split_idx = max(1, min(split_idx, n - 1))
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    return train_df, test_df


def _feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    drop_cols = {
        "match_id",
        "match_date",
        target_col,
        "outcome_class",
        "label_is_binary",
    }
    return [c for c in df.columns if c not in drop_cols]


def train_baseline_model(config_path: Path) -> Dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    feature_file = Path(cfg["feature_file"])
    output_dir = Path(cfg["output_dir"])
    target_col = cfg.get("target_column", "team_1_win")
    binary_flag_col = cfg.get("binary_label_flag_column", "label_is_binary")
    date_col = cfg.get("date_column", "match_date")
    test_size = float(cfg.get("test_size", 0.2))
    random_state = int(cfg.get("random_state", 42))
    model_name = cfg.get("model_name", "logistic_calibrated_v1")

    df = pd.read_csv(feature_file)

    # Build robust binary labels from outcome class first, then fall back to target column.
    if "outcome_class" in df.columns:
        df = df[df["outcome_class"].isin(["win", "loss"])].copy()
        df[target_col] = df["outcome_class"].map({"win": 1, "loss": 0}).astype(int)
    else:
        if binary_flag_col in df.columns:
            df = df[df[binary_flag_col] == 1].copy()
        parsed_target = pd.to_numeric(df[target_col], errors="coerce")
        df = df[parsed_target.isin([0, 1])].copy()
        df[target_col] = parsed_target.loc[df.index].astype(int)

    if len(df) < 10:
        raise ValueError("Not enough labeled rows after filtering to train model")

    train_df, test_df = _split_by_time(df, date_col, test_size)

    features = _feature_columns(df, target_col)
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_cols = [c for c in features if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    base_model = LogisticRegression(max_iter=1000, random_state=random_state)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=5)),
        ]
    )

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": _safe_metric(accuracy_score, y_test, pred),
        "precision": _safe_metric(precision_score, y_test, pred, zero_division=0),
        "recall": _safe_metric(recall_score, y_test, pred, zero_division=0),
        "f1": _safe_metric(f1_score, y_test, pred, zero_division=0),
        "roc_auc": _safe_metric(roc_auc_score, y_test, prob),
        "log_loss": _safe_metric(log_loss, y_test, prob, labels=[0, 1]),
        "brier": _safe_metric(brier_score_loss, y_test, prob),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.joblib"
    metrics_path = output_dir / f"{model_name}_metrics.json"
    preds_path = output_dir / f"{model_name}_test_predictions.csv"

    bundle = {
        "model": pipeline,
        "features": features,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "target": target_col,
        "model_name": model_name,
    }
    joblib.dump(bundle, model_path)

    metrics_payload = {
        "model_name": model_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features_count": int(len(features)),
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    preds_df = test_df[["match_id", "match_date", "team_1", "team_2", target_col]].copy()
    preds_df["pred_label"] = pred
    preds_df["pred_prob_team_1_win"] = prob
    preds_df.to_csv(preds_path, index=False)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(preds_path),
        "metrics": metrics,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
