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

## 3) Existing Dataset Understanding (Updated to New Cricsheet CSV Dataset)

Current source is Cricsheet JSON converted to normalized CSV tables in `data/cricsheet_csv`.

Available tables (already generated):

- `matches.csv` (5071 rows): match-level metadata, toss, winner, result
- `match_teams.csv` (10142 rows): match to team mapping
- `innings.csv` (10153 rows): innings totals, wickets, extras
- `deliveries.csv` (1147474 rows): ball-by-ball events with runs, extras, reviews, replacements
- `wickets.csv` (63793 rows): wicket events and dismissal details
- `powerplays.csv` (10064 rows): powerplay intervals by innings
- `player_of_match.csv` (4593 rows): player of the match labels
- `officials.csv` (19883 rows): umpires/referees and officials

Implications for modeling and analytics:

- We now have high-granularity ball-by-ball data, enabling stronger feature engineering.
- `batting_stats`, `bowling_stats`, `points_table` and similar aggregates should now be treated as derived Gold marts, not base inputs.
- Team, player, and venue signals should be derived from historical rollups of `deliveries`, `wickets`, `innings`, and `matches`.

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

1. Ingest normalized Cricsheet CSVs from `data/cricsheet_csv` into Bronze tables.
2. Keep optional external datasets separate and only join after schema compatibility checks.
3. Build a **data simulator** for live match events:
   - emits toss update, playing XI, innings progression, score snapshots
   - configurable interval (e.g., every 30 sec or 2 min)
4. Write ingestion connectors:
   - `ingest_historical.py` for batch load from normalized CSVs
   - `ingest_live_json.py` for new Cricsheet JSON drops (incremental)
   - `convert_new_json_to_csv.py` for continuous conversion of newly landed JSON files

Outputs:

- Reproducible ingested raw datasets
- Simulated live event stream

### Phase 2: Data Warehouse Design

Objective: Model cricket data using dimensional best practices.

Design (Star schema in Gold):

- Fact tables:
  - `fact_matches` (from `matches` + `match_teams`)
  - `fact_innings` (from `innings`)
  - `fact_deliveries` (from `deliveries`)
  - `fact_wickets` (from `wickets`)
  - `fact_player_match_performance` (derived from ball-by-ball rollups)
- Dimensions:
  - `dim_team`
  - `dim_player`
  - `dim_venue`
  - `dim_date`
  - `dim_tournament_stage`
  - `dim_official`

Tasks:

1. Build Bronze -> Silver cleaning transforms.
2. Build Silver -> Gold marts with dbt models, including derived marts:
   - `mart_batting_stats`
   - `mart_bowling_stats`
   - `mart_team_form`
   - `mart_venue_behavior`
   - `mart_powerplay_death_metrics`
3. Add dbt tests (unique, not null, accepted values, relationship tests).

Outputs:

- Documented warehouse schema
- Lineage graph and data dictionary

### Phase 3: ETL + Data Quality + Profiling

Objective: Production-grade preprocessing pipeline.

Tasks:

1. Missing value strategy per table/column.
2. Type standardization (dates, numeric stats, categories).
3. Outlier and anomaly checks (invalid over-ball combinations, impossible run totals, extras inconsistencies, wicket-event mismatches).
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
3. Batting-bowling matchup patterns from ball-by-ball interactions.
4. Venue behavior (average first innings score, chasing success, phase-wise scoring).
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
- venue-adjusted batting/bowling strength from delivery-level rollups
- toss + toss decision effect
- head-to-head record
- phase-based features (powerplay, middle overs, death overs run rate and wicket loss)
- wicket type profile (pace vs spin dismissal behavior)
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
   - Pull new JSON from API/simulator/drop-folder
   - Convert incrementally to normalized CSV rows
   - Append only unseen `match_id` or unseen delivery keys to Bronze tables

2. **Validation + Standardization Job**
   - Run quality checks
   - If failed, route to quarantine and raise alert

3. **Feature Refresh Job**
   - Recompute incremental features in Gold from `deliveries`/`wickets` deltas
   - Update feature snapshots for inference

4. **Inference Update Job**
   - Score upcoming/live matches
   - Store probabilities with timestamp/model_version

5. **Retraining Trigger Logic**
   - Time-based trigger (e.g., daily)
   - Event-based trigger (N new matches or M new deliveries)
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
2. Finalize Bronze schema contracts around `matches`, `innings`, `deliveries`, `wickets`, and related tables.
3. Build ingestion + Bronze/Silver transformations for normalized Cricsheet CSVs.
4. Define Gold feature marts and KPI marts (including batting/bowling aggregate marts derived from deliveries).
5. Train baseline model and create MLflow experiment.
6. Build dashboard v1 and API v1.
7. Add live JSON drop ingestion + scheduled conversion + retraining triggers.
8. Containerize and finalize hackathon narrative.

---

This plan is intentionally designed to be both hackathon-winnable and production-scalable: strong engineering depth, measurable ML quality, stakeholder usability, and operational reliability.
