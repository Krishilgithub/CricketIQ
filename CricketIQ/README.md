# 🏏 CricketIQ — ICC Men's T20 World Cup 2026 Prediction Platform

> An end-to-end intelligent cricket analytics platform that predicts match outcomes and delivers persona-based dashboards using historical ball-by-ball data and live match feeds.

---

## 🚀 Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/your-org/cricketiq.git
cd cricketiq
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys and paths

# 3. Install pre-commit hooks
pre-commit install

# 4. Run the full stack (Docker)
docker-compose up --build

# 5. Or run services individually
uvicorn src.inference.api:app --reload          # API at http://localhost:8000
streamlit run src/dashboard/main_app.py         # Dashboard at http://localhost:8501
mlflow ui --port 5050                           # MLflow at http://localhost:5050
```

---

## 📦 Project Structure

```
CricketIQ/
├── data/
│   ├── raw/
│   │   ├── cricsheet_csv_all/       # Full Cricsheet historical CSVs
│   │   └── given_data_csv/          # Hackathon tournament data (2026)
│   └── processed/
│       └── cricsheet_csv_men/       # Pre-filtered Men's International T20
├── src/
│   ├── ingestion/                   # Bronze data loaders
│   ├── simulation/                  # Live match event simulator
│   ├── warehouse/                   # DuckDB setup + ETL + data quality
│   ├── features/                    # Feature engineering pipeline
│   ├── training/                    # Model training, evaluation, registry
│   ├── inference/                   # FastAPI prediction service
│   ├── monitoring/                  # Drift + quality monitoring
│   ├── dashboard/                   # Streamlit persona dashboards
│   └── genai/                       # RAG assistant (optional)
├── dbt/                             # dbt Silver + Gold transform models
├── configs/                         # config.yaml + schema definitions
├── notebooks/                       # EDA and experiment notebooks
├── tests/                           # pytest unit + integration tests
├── docker/                          # Dockerfiles per service
├── scripts/                         # One-off utility scripts
└── artifacts/
    ├── models/                      # Saved model files
    └── reports/                     # Evaluation + DQ reports
```

---

## 🏗️ Architecture Overview

```
Raw CSVs (Cricsheet + Given)
        │
        ▼
┌───────────────────┐
│   Bronze Layer    │  Raw ingestion into DuckDB
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Silver Layer    │  Clean, standardize, filter (dbt)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    Gold Layer     │  Fact/Dim + KPI Marts (dbt)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Feature Pipeline  │  Rolling form, phase, venue, H2H features
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   ML Models       │  XGBoost + Calibration + SHAP + MLflow
└───────────────────┘
        │
        ▼
┌─────────┐   ┌────────────────────┐   ┌─────────────────┐
│ FastAPI │   │ Streamlit Dashboard│   │  GenAI Chatbot  │
└─────────┘   └────────────────────┘   └─────────────────┘
```

---

## 📊 Data Sources

| Dataset | Location | Description |
|---|---|---|
| Cricsheet CSV (All) | `data/raw/cricsheet_csv_all/` | 1.1M+ ball-by-ball events, 5,071 T20 matches |
| Given Hackathon Data | `data/raw/given_data_csv/` | 2026 ICC T20 WC tournament context (squads, venues, stage) |
| Processed (Men's) | `data/processed/cricsheet_csv_men/` | Pre-filtered Men's International subset |

---

## 🤖 ML Models

| Model | Type | Target | Metrics |
|---|---|---|---|
| Pre-match predictor | Binary classification | `team_1_win` | ROC-AUC, Log Loss, Brier Score |
| Live win probability | Sequential classification | `team_1_win` (per ball) | Log Loss, calibration |
| Expected score | Regression | Innings total | MAE, RMSE |

---

## 🧑‍💻 Tech Stack

| Layer | Technology |
|---|---|
| Data processing | Pandas, Polars, DuckDB |
| Transforms | dbt-core + dbt-duckdb |
| Data quality | Great Expectations |
| Orchestration | Prefect |
| ML | XGBoost, LightGBM, CatBoost, scikit-learn |
| Explainability | SHAP |
| Tracking | MLflow |
| Monitoring | Evidently |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| GenAI | LangChain + OpenAI + ChromaDB |
| Containers | Docker + Docker Compose |
| CI/CD | GitHub Actions |

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src
```

---

## 📖 Documentation

| Document | Location |
|---|---|
| Project Plan | `docs/planning/project_plan.md` |
| Dataset Comparison | `docs/research/dataset_comparison.md` |
| Feature Engineering | `docs/research/feature_engineering_notes.md` |
| Accuracy Brainstorm | `docs/research/accuracy_improvement_brainstorm.md` |

---

## 🗂️ Implementation Phases

| Phase | Status | Description |
|---|---|---|
| 0 | ✅ Done | Repository scaffold, configs, CI/CD |
| 1 | 🔄 Next | Data ingestion + live simulator |
| 2 | ⬜ Pending | Data warehouse (Bronze → Silver → Gold) |
| 3 | ⬜ Pending | ETL + Data quality pipeline |
| 4 | ⬜ Pending | EDA Dashboard |
| 5 | ⬜ Pending | Feature engineering |
| 6 | ⬜ Pending | ML modeling + MLflow |
| 7 | ⬜ Pending | Persona dashboards |
| 8 | ⬜ Pending | FastAPI service |
| 9 | ⬜ Pending | GenAI assistant |
| 10 | ⬜ Pending | MLOps + monitoring |
| 11 | ⬜ Pending | Docker + CI/CD |

---

*CricketIQ is designed to be both hackathon-winnable and production-scalable.*
