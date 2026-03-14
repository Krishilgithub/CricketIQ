# KenexAI - ICC Men's T20 Outcome Intelligence

Production-oriented codebase for ICC Men's T20 outcome prediction using Cricsheet data.

## Repository Layout

- `src/cricketiq/`: core package (feature engineering, training, inference, monitoring)
- `scripts/data/`: data ingestion and conversion scripts
- `scripts/dev/`: utility and analysis scripts
- `configs/`: runtime configuration files
- `data/raw/`: immutable source datasets
- `data/processed/`: curated modeling-ready datasets
- `artifacts/`: generated outputs (features, reports, models)
- `docs/`: planning, research, and architecture documents
- `tests/`: test scaffolding

## Quick Start

1. Create or activate virtual environment.
2. Install dependencies:
   - `python -m pip install -r requirements.txt`
3. Build men-only processed dataset:
   - `python scripts/data/filter_mens_dataset.py`
4. Run feature engineering:
   - `python scripts/run_feature_engineering.py`
5. Validate engineered features:
   - `python scripts/run_validation.py`
6. Train baseline calibrated model:
   - `python scripts/run_training.py`
7. Run monitoring report:
   - `python scripts/run_monitoring.py`

Generated artifacts:
- `artifacts/features/`
- `artifacts/models/`
- `artifacts/reports/`
