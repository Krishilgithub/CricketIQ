# Runbook

## 1) Data Preparation

- Convert raw Cricsheet JSON to normalized CSV:
  - `python scripts/data/convert_cricsheet_json_to_csv.py`
- Build men-only processed dataset:
  - `python scripts/data/filter_mens_dataset.py`

## 2) Feature Engineering

- Run feature pipeline:
  - `python scripts/run_feature_engineering.py`

Outputs in `artifacts/features/`:
- `match_level_features.csv`
- `match_team_features.csv`
- `live_state_features.csv`
- `feature_quality_report.json`
- `feature_engineering_summary.json`

## 3) Validation

- Validate feature integrity:
  - `python scripts/run_validation.py`

Output:
- `artifacts/reports/feature_validation_report.json`

## 4) Training

- Train calibrated baseline model:
  - `python scripts/run_training.py`

Outputs in `artifacts/models/`:
- `logistic_calibrated_v1.joblib`
- `logistic_calibrated_v1_metrics.json`
- `logistic_calibrated_v1_test_predictions.csv`

## 5) Monitoring

- Run monitoring/drift report:
  - `python scripts/run_monitoring.py`

Output:
- `artifacts/reports/monitoring_report.json`
