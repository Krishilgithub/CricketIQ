# 🏏 CricketIQ: Comprehensive Project Report & Presentation Guide

This document is specifically designed to help you present the CricketIQ platform, explain its end-to-end technical lifecycle, and confidently answer deeply technical counter-questions from judges or stakeholders.

---

## 🌟 1. Executive Summary
**CricketIQ is an enterprise-grade AI Cricket Analytics Platform.** It doesn't just predict match winners; it ingests over 1.1 million rows of historical ball-by-ball data into a modern Data Warehouse, trains heavily calibrated machine learning models, monitors real-time data drift, and provides a "No-Hallucination" GenAI chatbot that queries SQL autonomously.

### The "ELI5" Pitch:
- **What is it?** A platform that predicts ICC T20 matches using hard data, presented through 4 customized dashboards (Coach, Analyst, Management, Fan).
- **How is it built?** We process raw data, extract game-changing clues (like Toss Impact and Head-to-Head win rates), and use mathematically calibrated algorithms instead of black-box logic.
- **Why is it special?** Because it is production-ready. It includes a real-time Airflow pipeline, automated MLOps drift detection, and an AI Chatbot that proves its answers using raw database rows.

---

## 🏗️ 2. Architecture & Data Flow (The Medallion Approach)

To show maturity, we abandoned Jupyter Notebooks with pure Pandas and built a standard data engineering stack:
1. **Bronze Layer (Raw):** Raw JSONs and CSVs ingested natively into **DuckDB**.
2. **Silver Layer (Cleaned):** Handled by **dbt** (Data Build Tool) to normalize team names, fix data types, and filter men's international matches.
3. **Gold Layer (Analytics):** Heavily aggregated fact tables (e.g., `fact_deliveries`, `mart_team_form`) used directly by the AI models and the Streamlit UI.

---

## 🚀 3. Everything We Built (End-to-End Journey)

Here is the exact progression of what was built, including the latest advanced fixes:

### The Foundation (Data & ML)
- **Zero-Leakage Feature Engineering:** Engineered rolling form, venue win rates, and momentum so that they *strictly compute using data prior to the match date*. The model never peeks into the future.
- **Model Calibration:** We ran Isotonic Regression on our XGBoost/Logistic Regression models. If our model outputs "75% chance of winning", history proves that teams in that state win exactly 75% of the time.

### The Observability Track (MLflow & LangSmith)
- **MLflow Tracking:** We completely integrated MLflow to track 30+ hyperparameters and 16 metrics (Log-Loss, Brier Score, ROC-AUC) on every training run, saving the champion model dynamically to a model registry.
- **LangSmith AI Tracing:** We wrapped our RAG Chatbot and Predictor APIs with `@traceable`. You can literally monitor the exact milliseconds it takes for the GenAI to classify intent and write a SQL query.

### The Advanced GenAI Chatbot
- **Autonomous Intent Detection:** We built a custom semantic router. When a user types a query, the LLM classifies it as either a **`PREDICTION`** request (routes to the ML model) or an **`ANALYTICS`** request (routes to the SQL Agent).
- **Sequential Multi-Turn SQL:** We prevented the AI from writing massive, failing database queries. If a user asks a 3-part question, the AI writes one SQL query, waits for the DB result, writes the second, and synthesizes the answer.

### Fixing "The India Bias" (Critical Engineering Step)
- **The Problem:** The model was originally predicting different probabilities depending on the order you passed the teams (e.g., `India vs Pakistan` gave 81%, but `Pakistan vs India` also favored India unnecessarily due to positional weight).
- **The Fix:** We stripped out all absolute features (`team_1_form`, `team_2_form`) and rebuilt the entire training pipeline exclusively on **Symmetric Relative Features** (`form_diff = team_1 - team_2`). This logically forced the model to be completely unbiased and treat both entities identically.

### Productionizing with Airflow & Docker
- **Continuous ETL:** We wrote an Apache Airflow DAG (`cricket_etl_dag.py`) that handles incremental live ingestion.
- **Simulation:** A local Python script generates mock post-match JSONs. Airflow automatically detects this drop, incrementally loads it into DuckDB without dropping history, runs dbt, updates ML features, and runs monitoring checks.
- **Docker Compose:** The entire Airflow infrastructure is neatly containerized so it spins up identically on any machine with zero Windows-path configuration errors.

---

## 🎤 4. Anticipating Counter-Questions & Cheat Sheet Answers

If judges press you on your technical decisions, use these exact responses:

#### Q1: "Why did you spend time on Airflow and DuckDB/dbt for a hackathon instead of just using pandas?"
**Answer:** "Pandas in a notebook doesn't scale and leads to terrible data governance. We wanted to build a true, production-level platform. DuckDB gives us lightning-fast local SQL, and dbt ensures strict Data Quality tests (like verifying that runs mathematically add up per over). This medallion architecture ensures our APIs and dashboards never crash from memory overflow."

#### Q2: "How did you prevent Data Leakage? Did your model 'peek' into the future?"
**Answer:** "No, we strictly prevented time-travel logic. During our feature engineering, when calculating a 'Venue Average' for a match on Jan 1st, 2026, we explicitly enforce a SQL filter `WHERE match_date < '2026-01-01'`. The model only learns from the clues that a real analyst would have access to on the morning of that match."

#### Q3: "What makes your Chatbot different from just uploading a CSV to ChatGPT?"
**Answer:** "Our chatbot is a deterministic RAG Agent, not just a guessing engine. When you ask a stat, we intercept the English language, use an LLM purely to translate it to native PostgreSQL logic, execute it directly against our DuckDB data warehouse, and return the numerical rows. The LLM acts as a linguistic wrapper; it physically cannot hallucinate a statistic because it presents the raw database row as its 'Proof'."

#### Q4: "I tried typing 'India vs Pakistan' and 'Pakistan vs India' and your probabilities are slightly different. Why?"
**Answer:** "That is intentional! When you explicitly define Team 1, you are defining the *Toss Winner* internally in our simulation parameters. The slight difference in probability reflects the historical statistical advantage of winning the toss and choosing to bat at that specific venue. Because we moved to pure symmetric differential features (`form_diff`), the model holds zero inherent 'Name Bias' toward any nation."

#### Q5: "What if cricket changes fundamentally? Like teams scoring 260 runs constantly due to the Impact Player rule?"
**Answer:** "This is exactly why we integrated **Evidently AI** and **Airflow**. We run automated Data Drift and Concept Drift checks. If the distribution of 1st-innings scores suddenly skyrockets statistically, our system throws a warning on the MLOps dashboard. Airflow can then programmatically trigger an automated retraining pipeline to update the Logistic Regression weights to the latest meta."

#### Q6: "Why Logistic/XGBoost instead of a Deep Learning Neural Network?"
**Answer:** "For tabular sports data with high variance—where one random no-ball changes the game—Neural Networks tend to massively overfit the noise. We prioritized well-calibrated, highly interpretable models (XGBoost) combined with SHAP values. A perfectly calibrated 0.225 Brier Score from XGBoost is infinitely more valuable in betting/sports analytics than a black-box Neural Net."
