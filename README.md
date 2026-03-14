# 🏏 ICC Men's T20 World Cup 2026 — Outcome Prediction System

A comprehensive end-to-end Data & AI platform for predicting ICC Men's T20 Cricket World Cup 2026 match outcomes, built using Cricsheet ball-by-ball data enriched with franchise league intelligence.

## 🏗️ Architecture

**Multi-Source T20 Intelligence System** using Medallion Architecture:

```
Data Sources → Bronze (Raw) → Silver (Cleaned) → Gold (Analytics-Ready)
                                                        ↓
                                          ┌─────────────┼──────────────┐
                                          ↓             ↓              ↓
                                    Dashboards      ML Models     GenAI/RAG
```

## 📊 Data Sources (Cricsheet)

| Source | Description | Purpose |
|--------|-------------|---------|
| T20I Males | All ICC Men's T20 Internationals | Primary prediction target |
| IPL | Indian Premier League | Player form & feature enrichment |
| BBL | Big Bash League | Player profiling |
| CPL | Caribbean Premier League | Additional league intelligence |
| People Register | Player ID mappings | Cross-dataset player linking |

## 🚀 Features

1. **Data Warehouse** — Star schema with Medallion architecture (DuckDB)
2. **ETL Pipeline** — Automated ingestion, cleaning, transformation
3. **Data Simulation** — Ball-by-ball streaming simulator
4. **EDA Dashboard** — Data quality monitoring and exploratory analysis
5. **Persona Dashboards** — Coach, Analyst, Selector, Broadcaster, ICC views
6. **ML Models** — Match outcome, score prediction, player clustering, live win probability
7. **GenAI** — RAG chatbot for cricket stats, natural language SQL queries
8. **Optimization** — Batting order, squad selection optimization
9. **Docker Deployment** — Fully containerized end-to-end application

## 📁 Project Structure

```
icc-t20-wc-predictor/
├── config/              # Configuration files
├── data/
│   ├── raw/             # Raw Cricsheet JSON files
│   │   ├── t20i/        # Men's T20 Internationals
│   │   ├── ipl/         # IPL matches
│   │   ├── bbl/         # BBL matches
│   │   ├── cpl/         # CPL matches
│   │   └── register/    # People register
│   ├── bronze/          # Raw ingested data (parquet)
│   ├── silver/          # Cleaned & normalized (parquet)
│   └── gold/            # Aggregated & feature-ready (parquet)
├── src/
│   ├── ingestion/       # Data download & parsing
│   ├── etl/             # ETL pipeline
│   ├── simulation/      # Data streaming simulator
│   ├── warehouse/       # Schema definitions & migrations
│   ├── ml/              # Machine learning models
│   ├── genai/           # GenAI / RAG implementations
│   ├── optimization/    # Optimization use cases
│   ├── dashboards/      # Streamlit dashboards
│   └── utils/           # Shared utilities
├── notebooks/           # Jupyter notebooks for exploration
├── docker/              # Dockerfiles
├── tests/               # Unit & integration tests
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets from Cricsheet
python src/ingestion/download_data.py

# 3. Run ETL pipeline
python src/etl/run_pipeline.py

# 4. Launch dashboard
streamlit run src/dashboards/app.py

# 5. Docker deployment
docker-compose up --build
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Storage | DuckDB + Parquet files |
| ETL | Python + Polars + Prefect |
| Quality | Great Expectations + ydata-profiling |
| Dashboards | Streamlit |
| ML | scikit-learn, XGBoost, LightGBM |
| GenAI | LangChain + ChromaDB + Gemini API |
| Deployment | Docker + docker-compose |
