"""
Project configuration and settings.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ── Project Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
BRONZE_CRICSHEET_DIR = BRONZE_DIR / "cricsheet_csv"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# Raw data subdirectories
RAW_T20I_DIR = RAW_DIR / "t20i"
RAW_IPL_DIR = RAW_DIR / "ipl"
RAW_BBL_DIR = RAW_DIR / "bbl"
RAW_CPL_DIR = RAW_DIR / "cpl"
RAW_REGISTER_DIR = RAW_DIR / "register"

# ── Cricsheet Download URLs ───────────────────────────────────
CRICSHEET_URLS = {
    "t20i": "https://cricsheet.org/downloads/t20s_male_json.zip",
    "ipl": "https://cricsheet.org/downloads/ipl_male_json.zip",
    "bbl": "https://cricsheet.org/downloads/bbl_male_json.zip",
    "cpl": "https://cricsheet.org/downloads/cpl_male_json.zip",
    "people_register": "https://cricsheet.org/register/people.csv",
}

# Map dataset keys to raw directories
DATASET_DIRS = {
    "t20i": RAW_T20I_DIR,
    "ipl": RAW_IPL_DIR,
    "bbl": RAW_BBL_DIR,
    "cpl": RAW_CPL_DIR,
}

# ── Database ───────────────────────────────────────────────────
DUCKDB_PATH = DATA_DIR / "cricket_warehouse.duckdb"

# ── Streaming / Simulation ────────────────────────────────────
SIMULATION_INTERVAL_SECONDS = 5  # Ball-by-ball event interval
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_BALL_EVENTS = "cricket.ball_events"
KAFKA_TOPIC_MATCH_SUMMARY = "cricket.match_summary"

# ── ML ─────────────────────────────────────────────────────────
ML_MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── GenAI ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

# ── Match Type Constants ───────────────────────────────────────
T20_OVERS = 20
POWERPLAY_END = 6
MIDDLE_OVERS_END = 15
DEATH_OVERS_START = 16
