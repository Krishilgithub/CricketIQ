# ICC Men's T20 World Cup 2026 Outcome Prediction - Detailed Project Plan

## 1) Problem Statement (Reframed)
Predict the outcome of ICC Men's T20 World Cup 2026 matches using historical and near-real-time cricket data, and operationalize the solution as an end-to-end intelligent analytics platform.

This project must not only build a strong prediction model, but also provide:
- robust data engineering
- decision-support dashboards for multiple personas
- live data ingestion + continuous model retraining pipeline
- optional GenAI assistant capabilities
- containerized deployment

## 2) What Success Looks Like
A successful project should produce all of the following:
1. Reliable match outcome probability predictions before and during matches.
2. Reproducible data pipeline from raw ingestion to curated feature tables.
3. Data quality checks with automated alerts.
4. Persona-based dashboards (team analyst, coach, management, fan/media view).
5. MLOps workflow: experiment tracking, model registry, retraining, deployment.
6. Drift/performance monitoring with fallback strategy.
7. Dockerized deployable system that can run locally or in cloud.

## 3) Existing Dataset Understanding (Current Workspace)
Current tables:
- `matches.csv`: match-level outcomes, toss, margin, venue, winner
- `batting_stats.csv`: player batting aggregates
- `bowling_stats.csv`: player bowling aggregates
- `key_scorecards.csv`: key innings/player-level score entries
- `squads.csv`: player-team-role mappings
- `points_table.csv`: standings-level performance context
- `venues.csv`: venue metadata and capacity
- `awards.csv`: tournament highlight entities
- `tournament_summary.csv`: tournament-level meta facts

This is enough to build a robust baseline model and analytics layer.

## 4) Recommended Architecture (Hackathon-Strong + Production-Friendly)
Use a modern but practical architecture:

- **Ingestion Layer (Batch + Near Real-Time)**
  - Batch historical loads from CSV/Kaggle/API snapshots.
  - Near real-time periodic ingestion (every 5-15 mins) for live score/toss/team updates.

- **Data Lakehouse Layers (Medallion)**
  - **Bronze**: raw immutable data (exact source copy)
  - **Silver**: cleaned, standardized, deduplicated tables
  - **Gold**: analytics-ready feature marts + KPI marts

- **Warehouse / Query Layer**
  - DuckDB for local dev + fast analytics.
  - PostgreSQL (or BigQuery/Snowflake if cloud) for serving and BI.

- **ML Layer**
  - Feature engineering pipeline
  - Model training/evaluation
  - Model registry + versioning
  - Batch and online inference interfaces

- **Serving Layer**
  - FastAPI prediction service
  - Dashboard app (Streamlit or Power BI)

- **Monitoring Layer**
  - Data quality monitoring
  - Model performance + drift monitoring
  - Job orchestration and observability

## 5) Tooling and Tech Stack (Recommended)

### Core Language and Compute
- Python 3.11+
- Pandas + Polars (Polars for speed, Pandas for compatibility)
- NumPy

### Data Engineering
- DuckDB (local analytical warehouse)
- PostgreSQL (serving store)
- dbt-core (transform modeling, testing, docs)
- Great Expectations (data quality validation)
- Prefect (easy orchestration) or Airflow (if team prefers)

### ML and MLOps
- scikit-learn (baseline + interpretable models)
- XGBoost / LightGBM / CatBoost (strong tabular performance)
- Optuna (hyperparameter tuning)
- MLflow (experiment tracking + model registry)
- Evidently (data drift/performance drift monitoring)

### API and App
- FastAPI (prediction and stats APIs)
- Streamlit (rapid analytics + prediction UI)
- Plotly (interactive charts)

### CI/CD and Infra
- Docker + Docker Compose
- GitHub Actions (tests, lint, build, deployment)
- pre-commit (code quality hooks)

### Optional GenAI Layer
- Azure OpenAI/OpenAI API with RAG using:
  - LangChain or LlamaIndex
  - Chroma/FAISS for vector storage

## 6) Project Directory Blueprint

```text
KenexAI/
  data/
  src/
    ingestion/
    simulation/
    warehouse/
    features/
    training/
    inference/
    monitoring/
    dashboard/
    genai/
  dbt/
  configs/
  notebooks/
  tests/
  docker/
  scripts/
  artifacts/
    models/
    reports/
  plan.md
  README.md
```

## 7) Detailed Implementation Plan (Mapped to Expected Outcomes)

### Phase 1: Data Acquisition and Simulation
Objective: Prepare historical base + live-like stream.

Tasks:
1. Consolidate provided CSVs into Bronze tables.
2. Add external historical T20 datasets (if allowed) from Kaggle/API.
3. Build a **data simulator** for live match events:
   - emits toss update, playing XI, innings progression, score snapshots
   - configurable interval (e.g., every 30 sec or 2 min)
4. Write ingestion connectors:
   - `ingest_historical.py` (batch)
   - `ingest_live.py` (polling/event ingestion)

Outputs:
- Reproducible ingested raw datasets
- Simulated live event stream

### Phase 2: Data Warehouse Design
Objective: Model cricket data using dimensional best practices.

Design (Star schema in Gold):
- Fact tables:
  - `fact_matches`
  - `fact_innings_snapshots`
  - `fact_player_match_performance`
- Dimensions:
  - `dim_team`
  - `dim_player`
  - `dim_venue`
  - `dim_date`
  - `dim_tournament_stage`

Tasks:
1. Build Bronze -> Silver cleaning transforms.
2. Build Silver -> Gold marts with dbt models.
3. Add dbt tests (unique, not null, accepted values, relationship tests).

Outputs:
- Documented warehouse schema
- Lineage graph and data dictionary

### Phase 3: ETL + Data Quality + Profiling
Objective: Production-grade preprocessing pipeline.

Tasks:
1. Missing value strategy per table/column.
2. Type standardization (dates, numeric stats, categories).
3. Outlier and anomaly checks (margin, strike rate, economy).
4. Feature-safe joins across match/team/player entities.
5. Great Expectations validation suite:
   - schema checks
   - value range checks
   - freshness checks
   - duplicate checks
6. Automated quality report generation.

Outputs:
- Stable ETL jobs
- Data quality dashboard and failed-check alerting

### Phase 4: EDA and Data Quality Dashboard
Objective: Interactive exploratory and quality insights.

Dashboard modules:
1. Team performance trends by stage/venue.
2. Toss impact on win probability.
3. Batting-bowling matchup patterns.
4. Venue behavior (average first innings score, chasing success).
5. Data quality status panel (pass/fail trend).

Outputs:
- Stakeholder-friendly EDA + trust metrics

### Phase 5: Persona and KPI Framework
Objective: Align analytics with user decisions.

Personas and KPIs:
1. Team Analyst
   - pre-match win probability
   - best playing XI confidence score
   - venue-adjusted expected score
2. Coach/Captain
   - powerplay performance index
   - death overs risk index
   - toss decision recommendation confidence
3. Management/Strategy
   - player consistency index
   - form momentum score
   - opposition weakness heatmap
4. Fan/Media
   - match excitement index
   - key player impact tracker

Outputs:
- Persona-specific KPI pages and summaries

### Phase 6: ML Use Cases and Modeling
Objective: Build predictive models with explainability.

Primary use case:
- Match outcome prediction (classification): Team1 win / Team2 win

Secondary use cases:
- Expected total prediction (regression)
- Player impact clustering (unsupervised)
- Association rules for winning patterns (optional)

Feature set (examples):
- team form (last N matches)
- venue-adjusted batting/bowling strength
- toss + toss decision effect
- head-to-head record
- squad role balance (batsman/bowler/all-rounder mix)
- pressure/stage context

Modeling strategy:
1. Baseline: Logistic Regression
2. Advanced: XGBoost/LightGBM/CatBoost
3. Calibration: Platt/Isotonic for probability reliability
4. Explainability: SHAP global and local explanations

Evaluation metrics:
- ROC-AUC
- Log Loss (important for probabilities)
- Brier Score (calibration quality)
- Precision/Recall/F1 (secondary)

Outputs:
- Best model + feature importance + calibration report

### Phase 7: GenAI Use Cases
Objective: Add intelligent natural-language analytics.

Use cases:
1. Conversational cricket analyst bot:
   - "Why is Team A favored at this venue?"
2. Auto-generated pre-match brief:
   - strengths, risks, X-factors
3. Unstructured document extractor:
   - parse articles/reports and add contextual signals (optional)

Implementation notes:
- RAG over curated tables + generated insights
- prompt templates for persona-specific explanations
- guardrails to avoid unsupported claims

Outputs:
- GenAI assistant integrated into dashboard/API

### Phase 8: Optimization and Reliability
Objective: Improve speed, cost, and robustness.

Tasks:
1. Feature computation caching.
2. Incremental dbt models.
3. Parallelized ingestion and transformations.
4. Model serving latency optimization.
5. Champion-challenger model strategy.

Outputs:
- Faster pipelines and stable low-latency predictions

### Phase 9: Dockerized End-to-End Deployment
Objective: One-command spin-up of full stack.

Containers:
- API service
- Dashboard service
- Orchestrator/worker
- Database
- MLflow server (optional for demo)

Deliverable:
- `docker-compose up` launches complete platform

## 8) Live Data + Continuous Training Pipeline (Critical Requirement)

## Pipeline Design
1. **Live Ingestion Job (Every 5-15 mins)**
   - Pull data from API/simulator
   - Append to Bronze tables

2. **Validation + Standardization Job**
   - Run quality checks
   - If failed, route to quarantine and raise alert

3. **Feature Refresh Job**
   - Recompute incremental features in Gold
   - Update feature snapshots for inference

4. **Inference Update Job**
   - Score upcoming/live matches
   - Store probabilities with timestamp/model_version

5. **Retraining Trigger Logic**
   - Time-based trigger (e.g., daily)
   - Event-based trigger (N new matches)
   - Drift-based trigger (Evidently alert)

6. **Training + Evaluation Pipeline**
   - train candidate model
   - compare with champion model
   - register if threshold criteria are met

7. **Safe Promotion**
   - canary release or staged rollout
   - fallback to previous stable model on regressions

## Promotion Gate Criteria (Example)
Promote candidate only if all conditions pass:
- Log Loss improves by >= 2%
- Brier Score does not worsen
- Calibration error within threshold
- No critical data quality failures in past 24h

## Monitoring Signals
- Data freshness lag
- Feature null spikes
- Prediction confidence drift
- Realized accuracy by stage/venue/team cluster

## 9) Suggested Milestones (8-Week Hackathon Plan)

Week 1:
- finalize architecture, data contracts, repository scaffold

Week 2:
- Bronze/Silver/Gold pipeline + dbt tests + basic EDA

Week 3:
- baseline model + feature store tables + evaluation notebook

Week 4:
- advanced models + SHAP + calibration + model registry

Week 5:
- dashboards (EDA + persona KPIs + prediction page)

Week 6:
- live simulator + scheduled ingestion + incremental feature refresh

Week 7:
- retraining pipeline + drift monitoring + alert rules

Week 8:
- dockerization + CI/CD + final demo script + documentation

## 10) Risk Register and Mitigations
1. Data sparsity for certain team combinations
   - Mitigation: smoothing priors + hierarchical features
2. Leakage in match-level features
   - Mitigation: strict time-aware train/validation split
3. Poor probability calibration
   - Mitigation: post-hoc calibration and calibration monitoring
4. Live source interruptions
   - Mitigation: retry/backoff + cached fallback + simulation mode
5. Concept drift across tournaments
   - Mitigation: drift triggers + periodic retraining + champion fallback

## 11) Demo Day Storyboard
1. Show live ingestion and freshness panel.
2. Show data quality checks and passed validations.
3. Explain model prediction for an upcoming match.
4. Show SHAP explanation of top drivers.
5. Ask GenAI assistant a tactical question.
6. Trigger retrain workflow and show model registry update.
7. Show containerized launch and reproducibility.

## 12) Final Deliverables Checklist
- [ ] End-to-end ETL + warehouse pipeline
- [ ] Data quality framework and dashboard
- [ ] Match outcome prediction model with explainability
- [ ] Live data ingestion + retraining workflow
- [ ] Persona-based analytics dashboards
- [ ] GenAI assistant module (optional but high impact)
- [ ] Dockerized deployment
- [ ] README with setup/run instructions
- [ ] Architecture diagram and presentation deck

## 13) Immediate Next Steps (Execution Order)
1. Scaffold repository folders and config templates.
2. Build ingestion + Bronze/Silver transformations.
3. Define Gold feature marts and KPI marts.
4. Train baseline model and create MLflow experiment.
5. Build dashboard v1 and API v1.
6. Add live simulation + scheduled jobs + retraining triggers.
7. Containerize and finalize hackathon narrative.

---
This plan is intentionally designed to be both hackathon-winnable and production-scalable: strong engineering depth, measurable ML quality, stakeholder usability, and operational reliability.
