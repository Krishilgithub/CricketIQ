# CricketIQ Intelligence Hub: AI System & Chatbot Architecture

## Table of Contents
1. [Section 1 — Introduction](#section-1--introduction)
2. [Section 2 — System Architecture](#section-2--system-architecture)
3. [Section 3 — Query Routing System](#section-3--query-routing-system)
4. [Section 4 — RAG Model Explanation (Deterministic Retrieval)](#section-4--rag-model-explanation-deterministic-retrieval)
5. [Section 5 — The Vector Database Decision](#section-5--the-vector-database-decision)
6. [Section 6 — SQL Analytics Chatbot](#section-6--sql-analytics-chatbot)
7. [Section 7 — Machine Learning Prediction Model](#section-7--machine-learning-prediction-model)
8. [Section 8 — LangSmith Observability](#section-8--langsmith-observability)
9. [Section 9 — Data Flow](#section-9--data-flow)
10. [Section 10 — Example User Workflows](#section-10--example-user-workflows)
11. [Section 11 — Technology Stack](#section-11--technology-stack)
12. [Section 12 — Future Improvements](#section-12--future-improvements)

---

## Section 1 — Introduction

### What is CricketIQ?
**CricketIQ Intelligence Hub** is an advanced, end-to-end AI-powered cricket analytics platform. It replaces manual spreadsheet analysis with real-time predictive modeling, interactive dashboards, and an autonomous SQL-based conversational agent. 

### Purpose of the RAG System & Chatbot
The Chatbot serves as the primary gateway for users to interact with 1.1 million rows of historical ball-by-ball data without knowing SQL or Python. 
Instead of relying on the internal, potentially hallucinated memory of Large Language Models (LLMs), CricketIQ utilizes a highly specialized **Retrieval-Augmented Generation (RAG)** pipeline. This system intercepts a user's natural language question, retrieves hard statistical facts directly from the Data Warehouse, and formulates a mathematically guaranteed answer. 

### How it Helps Users
- **Explore Cricket Analytics:** Fans, coaches, and management can slice and dice player form without writing queries.
- **Query Hard Data:** "What is Virat Kohli's strike rate in the death overs?" yields an exact, calculated decibel rather than a semantic guess.
- **Generate Predictive Insights:** Users can ask the bot for outright predictions, which trigger complex Machine Learning inference graphs beneath the UI.

---

## Section 2 — System Architecture

The CricketIQ AI infrastructure is built on a loosely coupled microservice architecture to ensure speed, determinism, and extreme observability.

### High-Level Components
1. **Streamlit Dashboard:** The unified front-end interface housing the Interactive Chatbot, the Prediction engine, and visual EDA analysis.
2. **Query Router:** A semantic NLP intent classifier that parses incoming user strings and figures out *which* AI agent to trigger (ML Predictor vs Database Analyst).
3. **Deterministic RAG Pipeline:** Extracts exact textual entities (player names, venues) to fetch highly specific contextual subsets of the Data Warehouse.
4. **SQL Analytics Agent:** The autonomous subsystem that converts English constraints into executable DuckDB PostgreSQL dialects.
5. **Prediction Model:** An XGBoost/Logistic Regression engine calibrated with Isotonic Regression to provide percentage-based match winner outcomes.

### Architecture Diagram
```text
[ User Interface ]  ---> ( Streamlit Chatbot )
                                │
                                ▼
                       [ Query Router ]
                      /                \
          (Analytics Intent)       (Prediction Intent)
                /                        \
      [ SQL Agent RAG ]          [ ML Prediction Engine ]
             │                             │
             ▼                             ▼
   (Fuzzy Entity Matching)       (Relative Form Features)
             │                             │
             ▼                             ▼
     [ DuckDB Warehouse ]           [ MLflow Registry ]
```

---

## Section 3 — Query Routing System

One of the biggest flaws in generic ChatGPT wrappers is attempting to make a single prompt handle everything. CricketIQ employs a **Semantic Query Router** (`src/chat/intent_classifier.py`).

### Routing Logic
When the user types a prompt, an ultra-fast LLM call classifies the intent into one of two buckets:
1. **`ANALYTICS` Queries → SQL Agent Pipeline**
   - Example: *"Who hit the most sixes in 2024?"*
   - Example: *"What is the average 1st innings score at Wankhede?"*
2. **`PREDICTION` Queries → Machine Learning Pipeline**
   - Example: *"Who will win between India and Pakistan tomorrow?"*
   - Example: *"Predict the outcome of Australia vs England."*

By strictly routing the query, we prevent the SQL agent from trying to write a SQL query for a future event that hasn't happened yet, and we prevent the ML model from trying to predict a historical statistic.

---

## Section 4 — RAG Model Explanation (Deterministic Retrieval)

Standard RAG pipelines chunk text, generate embeddings arrays, and perform semantic cosine-similarity searches. **However, Vector RAG is mathematically fatal for strict tabular analytics.** 
Instead, CricketIQ uses an **Agentic SQL RAG pipeline**, guaranteeing zero hallucinations.

### 1. Document Loader & Data Storage
Instead of loading unstructured PDFs, our "documents" are strictly structured **Fact** and **KPI** tables living inside a high-performance **DuckDB** columnar database.

### 2. Entity Retrieval (Fuzzy Matching instead of Embeddings)
We bypass traditional text chunking and cosine similarity. When the user asks *"How does Shbman Gil perform?"*:
- We execute rapid string-matching (`thefuzz` library algorithm) directly against the `batter` column in our Database.
- We map the typo *"Shbman Gil"* to the canonical entity `["Shubman Gill"]`.

### 3. Prompt Construction
The exact entity is injected into a strict system prompt instruction set. 
- *Agent Memory:* "The user is asking about [Shubman Gill]. You are connected to `fact_deliveries`. Write a SQL query."

### 4. LLM Generation
The LLM writes the SQL, executes it natively, retrieves the numerical scalar (e.g., `runs: 1450, SR: 142.1`), and outputs a conversational response: *"Shubman Gill has scored 1,450 runs at a strike rate of 142.1."*

---

## Section 5 — The Vector Database Decision

### Why We Excluded Vector Embeddings (ChromaDB / FAISS)
A standard feature of generic RAG is a Vector Database (like Pinecone or ChromaDB). **CricketIQ explicitly removed this architectural component in favor of DuckDB.**

#### The Problem with Embeddings in Finance/Sports
If a user asks *"What is India's win rate batting first?"*, a vector database searches for the "closest semantic sentence". It might retrieve a paragraph stating *"India played a great match batting first in 2011"*. It has no ability to run `SUM()` or `COUNT()` over a million rows.

#### Our Solution: Database-as-Storage
By keeping the data in raw tabular formats (`dbt` Silver/Gold models), we retain the ability to run exact analytical aggregations. We rely on standard SQL matching and string similarity rather than dense vector embeddings arrays, ensuring 100% computational correctness. 

---

## Section 6 — SQL Analytics Chatbot

The core of the Analytics intent is the `src/agents/sql_agent.py`. It converts English context directly into deterministic PostgreSQL syntax.

### The Execution Flow:
1. **Natural Language Processing:** User asks, *"Who scored the most runs in the powerplay?"*
2. **SQL Query Generation:** The LLM observes the schema for `fact_deliveries` and generates:
   ```sql
   SELECT batter, SUM(runs_batter) as total_runs
   FROM main_gold.fact_deliveries
   WHERE over_number < 6
   GROUP BY batter
   ORDER BY total_runs DESC LIMIT 1;
   ```
3. **Database Execution:** The Python backend intercepts the SQL, connects to DuckDB, runs the query, and captures the raw dataframe.
4. **Result Interpretation:** The LLM is fed the raw string output: `| batter | total_runs | \n | Rohit Sharma | 850 |` and translates it back into English.

---

## Section 7 — Machine Learning Prediction Model

If the Query Router detects a **`PREDICTION`** intent, the Chatbot bypasses the SQL agent entirely and calls `src/ml/predictor.py`.

### Pipeline Breakdown:
1. **Feature Engineering:** The system extracts the specific teams mentioned, verifies who is batting first (Toss Winner), and looks up the active `mart_team_form` to calculate the mathematical momentum and **Relative Form Differential**.
2. **Model Loading (MLflow):** The ML pipeline queries the local MLflow Model Registry to dynamically pull the `"champion"` assigned **XGBoost / Logistic Regression** weights.
3. **Inference & Calibration:** The model outputs a raw float. This is passed through an **Isotonic Regression Calibrator** so that if the model predicts a 70% win rate, history proves the team mathematically wins 70% of the time.
4. **Response:** The system renders the percentage confidence in the Chatbot UI.

---

## Section 8 — LangSmith Observability

To bring this to a production-ready enterprise standard, the entire LLM pipeline is draped in **LangSmith** tracing decorators (`@traceable`).

### Why Observability is Critical
Without observability, an AI agent is a black box. If the SQL agent hallucinates a column name and fails, the developer has no idea why.

### What LangSmith Tracks in CricketIQ:
- **LLM Prompts:** Records the exact system prompt and user history passed to OpenAI/Gemini.
- **RAG Retrieval Steps:** Measures exactly how many milliseconds the `thefuzz` string-matching algorithm took to map the player's name.
- **SQL Agent Reasoning:** If the LLM writes a broken SQL query and retries, LangSmith logs the exact internal dialogue and error stack trace the LLM used to heal itself.
- **Tokens & Cost:** Measures exact tokenizer costs per interaction.

---

## Section 9 — Data Flow Architecture

Here is the exact millisecond-by-millisecond progression of a system interaction:

1. **User Query → Streamlit:** User types in the UI box.
2. **Streamlit → LangSmith Logger:** Session ID is generated for monitoring.
3. **Streamlit → Query Router:** LLM rapidly classifies as `ANALYTICS` or `PREDICTION`.
4. *(If Analytics)* **Query Router → SQL Agent (RAG):**
   - Agent pulls schema from DuckDB.
   - Agent writes SQL.
   - DuckDB executes SQL and returns memory to Agent.
5. *(If Prediction)* **Query Router → ML Predictor:**
   - Predictor loads `XGBoost.pkl` from MLflow.
   - Outputs float `0.84`.
6. **Agent/Predictor → Streamlit:** Formatted markdown and Plotly charts are generated for the end user.

---

## Section 10 — Example User Workflows

### Example 1: The Analytics Query
**User:** *"Which team has the highest win rate?"*
**Flow:** 
1. `Query Router` detects `ANALYTICS`.
2. `SQL Agent` looks at the Gold layer schemas.
3. Writes: `SELECT team, win_rate FROM main_gold.mart_team_form ORDER BY win_rate DESC LIMIT 1;`
4. DuckDB returns `"India | 0.72"`.
5. **AI output:** *"Historically, India holds the highest win rate in the dataset at 72.0%."*

### Example 2: The Prediction Query
**User:** *"Will India win against Australia tomorrow at Eden Gardens?"*
**Flow:**
1. `Query Router` detects `PREDICTION`.
2. `Entity Extractor` extracts `["India", "Australia", "Eden Gardens"]`.
3. `Predictor` fetches the rolling form for India and Australia, calculates the form differential `(India Form - Aus Form)`.
4. MLflow model predicts `0.65`.
5. **AI output:** *"Based on mathematical modeling and relative form differentials, India is mathematically favored to win with a 65% probability."*

---

## Section 11 — Technology Stack

| Component | Technology | Why Chosen? |
| :--- | :--- | :--- |
| **Front-End & UI** | `Streamlit` | Enables rapid dashboarding and interactive chat UIs in pure Python. |
| **Orchestration** | `LangChain` | Provides the agentic framework for looping SQL generation workflows. |
| **Data Warehouse** | `DuckDB` + `dbt` | Lightning-fast, in-memory analytical columnar processing without setting up bulky Postgres servers. |
| **Experiment Tracking** | `MLflow` | The industry standard for capturing Hyperparameters and Model Artifacts. |
| **Observability** | `LangSmith` | Crucial for debugging LLM token usage, latency, and reasoning loops. |
| **Programming** | `Python 3.11+` | Industry standard for Data Engineering and ML workloads. |

---

## Section 12 — Future Improvements

While CricketIQ is a highly robust V1, future extensions will focus on:
1. **Real-time Live Data Ingestion Socket:** Connecting to a live ESPNCricinfo WebSocket to provide ball-by-ball shifting win probabilities directly in the Chatbot.
2. **Advanced Player Analytics (Embeddings):** Integrating true Vector Embeddings on *unstructured cricket news articles* to provide qualitative context (e.g., "Is Virat Kohli currently injured?") alongside the quantitative SQL stats.
3. **Reinforcement Learning:** Training a dedicated smaller model specifically tuned on resolving DuckDB syntax errors faster than off-the-shelf Gemini/GPT.
