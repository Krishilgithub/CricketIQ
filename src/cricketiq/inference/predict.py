from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd


def load_model_bundle(model_path: Path) -> Dict[str, Any]:
    bundle = joblib.load(model_path)
    return bundle


def predict_from_records(model_path: Path, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    features = bundle["features"]

    df = pd.DataFrame(records)
    for col in features:
        if col not in df.columns:
            df[col] = None

    X = df[features]
    prob = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    out = []
    for i in range(len(df)):
        item = records[i].copy()
        item["pred_label"] = int(pred[i])
        item["pred_prob_team_1_win"] = float(prob[i])
        out.append(item)
    return out
