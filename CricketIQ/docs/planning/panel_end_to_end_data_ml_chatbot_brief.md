# CricketIQ End-to-End Data, ML, and Chatbot Brief

## 1. What This Document Covers
This document explains:
- Raw data tables and sources
- Preprocessing at each pipeline layer
- ETL implementation and data engineering flow
- Bronze, Silver, Gold architecture with real 5-row samples
- ML features and prediction logic
- Why DuckDB was used and the benefits
- Chatbot implementation, RAG strategy, prompt behavior, and output generation
- Why MLflow and LangSmith are used in this project
- File-by-file code map from ingestion to UI
- Panel questions with strong answers, especially for data engineering and layered pipeline design

All phase samples below come from the current local run and current DuckDB state.

---

## 2. End-to-End Pipeline Overview

### High-level flow
1. ETL raw ingestion from CSV files into DuckDB Bronze tables
2. dbt transformations to Silver (cleaning, canonicalization, filtering)
3. dbt transformations to Gold (facts and marts for analytics + ML)
4. Feature engineering to build training dataset
5. Model training with CV and champion selection
6. Champion model serving in Streamlit prediction workflow
7. LLM chatbot with Text-to-SQL + retrieval-assisted prompt context
8. ML/LLM observability through MLflow and LangSmith

### Pipeline implementation points
- Ingestion: idempotent CSV-to-DuckDB Bronze loads
- Transform: dbt models for Silver and Gold schemas
- Feature generation: historical-only features to avoid leakage
- Training: TimeSeriesSplit cross-validation with multiple model candidates
- Serving: precomputed SQL features + model inference for match prediction

---

## 3. Raw Data and Table Inventory

### Raw Cricsheet normalized tables (loaded to Bronze)
- matches
- match_teams
- innings
- deliveries
- wickets
- powerplays
- player_of_match
- officials

### Given tournament/context tables (loaded to Bronze)
- given_matches
- given_batting_stats
- given_bowling_stats
- given_key_scorecards
- given_squads
- given_points_table
- given_venues
- given_awards
- given_tournament_summary

### Current warehouse table inventory
- Bronze schema tables: deliveries, given_awards, given_batting_stats, given_bowling_stats, given_key_scorecards, given_matches, given_points_table, given_squads, given_tournament_summary, given_venues, innings, match_teams, matches, officials, player_of_match, powerplays, wickets
- Silver schema tables: slv_deliveries, slv_innings, slv_match_teams, slv_matches, slv_wickets
- Gold schema tables: fact_deliveries, fact_innings, fact_matches, fact_wickets, mart_batting_stats, mart_bowling_stats, mart_team_form

---

## 4. What Preprocessing Was Done

### ETL to Bronze
- Raw CSVs are loaded as-is into Bronze tables
- Idempotency is enforced by primary-key based insert-if-not-exists logic
- No business filtering or canonical remapping at this stage

### Silver preprocessing
- Filter scope applied at match level:
  - gender = male
  - match_type = T20
  - team_type = international
- Canonical alias normalization using seed maps:
  - team_aliases
  - venue_aliases
- Type casting and useful derivations:
  - integer casts for over/ball/runs fields
  - legal-ball indicator in deliveries
  - stable over/ball sequence

### Gold preprocessing/modeling
- Gold facts are built for direct analytical and ML consumption
- Gold fact_matches adds:
  - team_1 (defined as toss_winner)
  - team_1_win target label
- Gold fact_deliveries enriches deliveries with wicket flags and dismissal details
- Gold mart_team_form builds rolling historical form using window functions

### ML preprocessing
- Numerical-only feature pipeline
- Median imputation + standard scaling
- Team identity categorical variables removed to reduce leakage/bias

---

## 5. Phase Samples (5 Rows Each)

### 5.1 ETL Phase Sample (raw CSV before Bronze)
Representative source: data/raw/cricsheet_csv_all/matches.csv

| match_id | match_date | gender | match_type | team_type | toss_winner | winner | venue |
|---:|---|---|---|---|---|---|---|
| 1001349 | 2017-02-17 00:00:00 | male | T20 | international | Sri Lanka | Sri Lanka | Melbourne Cricket Ground |
| 1001351 | 2017-02-19 00:00:00 | male | T20 | international | Sri Lanka | Sri Lanka | Simonds Stadium, South Geelong |
| 1001353 | 2017-02-22 00:00:00 | male | T20 | international | Sri Lanka | Australia | Adelaide Oval |
| 1004729 | 2016-09-05 00:00:00 | male | T20 | international | Hong Kong | Hong Kong | Bready Cricket Club, Magheramason |
| 1007655 | 2016-06-18 00:00:00 | male | T20 | international | India | Zimbabwe | Harare Sports Club |

### 5.2 Bronze Phase Sample (bronze.matches)

| match_id | match_date | gender | match_type | team_type | toss_winner | winner | venue |
|---:|---|---|---|---|---|---|---|
| 1001349 | 2017-02-17 00:00:00 | male | T20 | international | Sri Lanka | Sri Lanka | Melbourne Cricket Ground |
| 1001351 | 2017-02-19 00:00:00 | male | T20 | international | Sri Lanka | Sri Lanka | Simonds Stadium, South Geelong |
| 1001353 | 2017-02-22 00:00:00 | male | T20 | international | Sri Lanka | Australia | Adelaide Oval |
| 1004729 | 2016-09-05 00:00:00 | male | T20 | international | Hong Kong | Hong Kong | Bready Cricket Club, Magheramason |
| 1007655 | 2016-06-18 00:00:00 | male | T20 | international | India | Zimbabwe | Harare Sports Club |

### 5.3 Silver Phase Sample (main_silver.slv_matches)

| match_id | match_date | season | event_name | venue | toss_winner | winner | result_type |
|---:|---|---|---|---|---|---|---|
| 1001349 | 2017-02-17 00:00:00 | 2016/17 | Sri Lanka in Australia T20I Series | Melbourne Cricket Ground | Sri Lanka | Sri Lanka | wickets |
| 1001353 | 2017-02-22 00:00:00 | 2016/17 | Sri Lanka in Australia T20I Series | Adelaide Oval | Sri Lanka | Australia | runs |
| 1050217 | 2016-09-23 00:00:00 | 2016/17 | Pakistan v West Indies T20I Series | Dubai International Cricket Stadium | Pakistan | Pakistan | wickets |
| 1050219 | 2016-09-24 00:00:00 | 2016/17 | Pakistan v West Indies T20I Series | Dubai International Cricket Stadium | West Indies | Pakistan | runs |
| 1050221 | 2016-09-27 00:00:00 | 2016/17 | Pakistan v West Indies T20I Series | Sheikh Zayed Stadium | Pakistan | Pakistan | wickets |

### 5.4 Gold Phase Sample (main_gold.fact_matches)

| match_id | match_date | season | event_name | venue | team_1 | winner | team_1_win |
|---:|---|---|---|---|---|---|---:|
| 1001349 | 2017-02-17 00:00:00 | 2016/17 | Sri Lanka in Australia T20I Series | Melbourne Cricket Ground | Sri Lanka | Sri Lanka | 1 |
| 1001353 | 2017-02-22 00:00:00 | 2016/17 | Sri Lanka in Australia T20I Series | Adelaide Oval | Sri Lanka | Australia | 0 |
| 1050217 | 2016-09-23 00:00:00 | 2016/17 | Pakistan v West Indies T20I Series | Dubai International Cricket Stadium | Pakistan | Pakistan | 1 |
| 1050219 | 2016-09-24 00:00:00 | 2016/17 | Pakistan v West Indies T20I Series | Dubai International Cricket Stadium | West Indies | Pakistan | 0 |
| 1050221 | 2016-09-27 00:00:00 | 2016/17 | Pakistan v West Indies T20I Series | Sheikh Zayed Stadium | Pakistan | Pakistan | 1 |

---

## 6. Feature Set Used by the ML Model

### Final model input features
1. toss_bat
2. venue_avg_1st_inns_runs
3. venue_chase_success_rate
4. h2h_advantage
5. form_last5_diff
6. form_last10_diff
7. momentum_diff
8. venue_win_rate_diff

### How these are calculated
- toss_bat: 1 if toss decision is bat, else 0
- venue_avg_1st_inns_runs: rolling historical average first-innings total at the venue
- venue_chase_success_rate: historical chasing success at the venue
- h2h_advantage: team_1_h2h_win_rate - 0.5
- form_last5_diff: team_1_form_last5 - team_2_form_last5
- form_last10_diff: team_1_form_last10 - team_2_form_last10
- momentum_diff: team_1_momentum - team_2_momentum
- venue_win_rate_diff: team_1_venue_win_rate - team_2_venue_win_rate

### How output is calculated
- The champion model is loaded from champion_model.pkl
- Predictor computes all 8 features in real time from Gold-layer SQL utilities
- Model outputs probability P(team_1 wins) via predict_proba
- P(team_2 wins) = 1 - P(team_1 wins)
- Favourite = team with larger probability
- Confidence label from probability magnitude:
  - High if >= 62%
  - Moderate if >= 54%
  - Low otherwise

### Current champion model snapshot
- Champion: LogisticRegression
- CV Log Loss: 0.634434
- CV ROC-AUC: 0.689492

---

## 7. Why DuckDB Was Used and Benefits

### Why DuckDB here
- This is a local-first analytics platform with heavy SQL and columnar scans
- Need fast analytical reads for Streamlit dashboards + chatbot SQL
- Need simple deployability without running an external DB server

### Benefits in this project
- Very fast OLAP queries over large historical cricket datasets
- Embedded DB file is easy to version and move across environments
- Tight integration with pandas and dbt-duckdb
- Works well for layered medallion architecture in one local artifact
- Low operational overhead for hackathon and production-like demos

---

## 8. Chatbot and RAG Implementation

### Chatbot architecture
1. User input arrives in chatbot page
2. Intent classifier routes to:
   - Prediction path (ML model)
   - SQL analytics path (Text-to-SQL agent)
3. For SQL path:
   - Query rewrite for standalone context
   - RAG pipeline gathers detected entities from DB using fuzzy matching
   - Dynamic system prompt is built with schema + matched entities + strict instructions
   - LLM generates SQL in <SQL> tags
   - SQL executed on DuckDB (read-only)
   - DB result returned to LLM for final answer synthesis

### What RAG is used
- Retrieval is schema/entity retrieval, not vector embedding retrieval
- It retrieves likely teams, players, and venues from Gold tables
- It uses exact and fuzzy matching to enrich prompt context

### What the prompt gives the model
- Database schema details (fact and mart tables)
- Detected entities (team/player/venue hints)
- Strict generation rules:
  - Must query via SQL first
  - Read-only SQL only
  - Include proof-style result tables
  - Special handling for matchups and phase queries

### How output is generated
- Iterative loop up to fixed number of steps
- If SQL returned, result is injected back and model continues
- Final response is natural-language analysis backed by query outputs

---

## 9. Why MLflow and LangSmith Are Used

### MLflow use case here
- Experiment tracking for each candidate model run
- Logs:
  - hyperparameters
  - CV metrics (mean/std)
  - evaluation plots and feature artifacts
- Registers champion model in model registry
- Supports reproducibility and model governance

### LangSmith use case here
- Tracing and observability across LLM chains, tools, and agent loops
- Tracks:
  - intent classification
  - query rewrite
  - RAG retrieval
  - SQL generation and execution calls
  - prediction tool calls
- Helps debug prompt behavior, routing mistakes, and latency/reliability issues

---

## 10. File Map: Start-to-End Code Locations

### Configuration and app entry
- src/config.py
- configs/config.yaml
- src/app.py
- src/pages/shared.py

### ETL and ingestion
- src/ingestion/convert_new_json_to_csv.py
- src/ingestion/ingest_historical.py
- src/ingestion/ingest_live_json.py
- src/ingestion/canonical_mappings.py

### dbt data engineering layers
- dbt/dbt_project.yml
- dbt/profiles.yml
- dbt/seeds/team_aliases.csv
- dbt/seeds/venue_aliases.csv
- dbt/models/silver/slv_matches.sql
- dbt/models/silver/slv_match_teams.sql
- dbt/models/silver/slv_innings.sql
- dbt/models/silver/slv_deliveries.sql
- dbt/models/silver/slv_wickets.sql
- dbt/models/gold/fact_matches.sql
- dbt/models/gold/fact_innings.sql
- dbt/models/gold/fact_deliveries.sql
- dbt/models/gold/fact_wickets.sql
- dbt/models/gold/mart_team_form.sql
- dbt/models/gold/mart_batting_stats.sql
- dbt/models/gold/mart_bowling_stats.sql
- dbt/models/gold/schema.yml

### Feature engineering and preprocessing
- src/features/feature_engineering.py
- src/features/data_preprocessing.py

### Model training and registry
- src/models/train_models.py
- src/models/evaluate_models.py
- src/models/mlflow_tracking.py
- artifacts/models/experiment_summary.json
- artifacts/models/champion_model.pkl

### Online prediction path
- src/ml/predictor.py
- src/pages/prediction.py
- src/ui/prediction_display.py

### Chatbot, RAG, and SQL agent
- src/pages/chatbot.py
- src/chat/intent_classifier.py
- src/chat/entity_extractor.py
- src/agents/sql_agent.py
- src/rag/rag_pipeline.py
- src/rag/retriever.py
- src/rag/prompt_builder.py

### LLM observability
- src/observability/langsmith_tracing.py

---

## 11. Panel Q&A (with strong data-engineering focus)

### Q1. Why did you use Bronze, Silver, Gold instead of one cleaned table?
Answer: Separation of concerns. Bronze preserves source truth, Silver standardizes and filters with business rules, Gold serves analytics/ML-ready curated facts and marts. This gives traceability, easier debugging, and reproducible transformations.

### Q2. What exactly changes from Bronze to Silver?
Answer: Silver applies domain filters (male, T20, international), alias canonicalization for teams/venues, type casting, and standard structural cleanup like legal-ball flags and normalized over/ball numbering.

### Q3. What exactly changes from Silver to Gold?
Answer: Gold creates consumer-facing fact tables and marts. It also adds model target fields (team_1 and team_1_win) and enriched event-level fields (is_wicket and dismissal details).

### Q4. How did you prevent duplicate ingestion?
Answer: Ingestion uses primary-key aware insert-if-not-exists logic per table, so reruns remain idempotent.

### Q5. How do you ensure no leakage in features?
Answer: Rolling stats are shifted and computed only from prior matches (historical cutoff before current match), preventing future information from entering training features.

### Q6. Why is team identity removed from model features?
Answer: Team-name categorical leakage can cause memorization and unfair bias. We moved to relative strength/form features to improve generalization and symmetry.

### Q7. Why DuckDB over a server DB?
Answer: For this workload, DuckDB gives high analytical performance, zero server overhead, and smooth dbt/pandas integration in one file artifact.

### Q8. How do you support both analytics and prediction in one pipeline?
Answer: Gold layer serves both SQL analytics and ML feature extraction. Predictor queries shared Gold-derived statistics to build model input at runtime.

### Q9. How is model quality tracked?
Answer: MLflow records full CV metrics, plots, feature diagnostics, and model registry versions. Champion promotion is based on tracked metrics.

### Q10. Why add LangSmith if MLflow already exists?
Answer: MLflow tracks model experiments. LangSmith traces LLM agent behavior (prompt, routing, SQL generation, tool calls). They solve different observability problems.

### Q11. What is your RAG strategy?
Answer: Retrieval is schema/entity grounding. We retrieve candidate entities and schema context from DB and inject them into a controlled prompt for reliable Text-to-SQL generation.

### Q12. How do you keep chatbot SQL safe?
Answer: SQL agent blocks destructive statements and only allows read-oriented queries. Execution runs against DuckDB in read-only mode.

### Q13. If panel asks for proof that layers are real, what do you show?
Answer: Show dbt model files, schema-specific table lists, and side-by-side 5-row examples from ETL source, Bronze, Silver, and Gold exactly as in this document.

### Q14. If panel asks where business logic lives, what do you answer?
Answer: Business filtering and canonicalization live in Silver dbt models; analytics/target derivations live in Gold dbt models; ML-only transformations live in feature engineering Python.

### Q15. How do you explain end-to-end reproducibility?
Answer: Raw-to-Bronze ingestion scripts + dbt transforms + feature build + MLflow tracked training + versioned model artifact give deterministic reruns and auditability.

---

## 12. Practical Demo Sequence for Panel
1. Show raw and bronze sample rows
2. Show silver filtering/canonicalization SQL
3. Show gold target derivation and marts
4. Show feature table columns and historical-only logic
5. Show MLflow experiment summary and champion metrics
6. Run one prediction query in UI
7. Run one chatbot SQL question and show generated SQL + response

This sequence demonstrates complete ownership of data engineering, ML, and LLM integration.
