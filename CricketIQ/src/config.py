"""
CricketIQ — Shared configuration loader.

Usage:
    from src.config import get_config
    cfg = get_config()
    duckdb_path = cfg["paths"]["duckdb_path"]
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from project root (silently ignore if not present)
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    """Load and cache project config from configs/config.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Allow environment variable overrides for critical paths
    _env_overrides = {
        ("paths", "duckdb_path"):         "DUCKDB_PATH",
        ("paths", "data_raw_cricsheet"):  "CRICSHEET_DATA_PATH",
        ("paths", "data_raw_given"):      "GIVEN_DATA_PATH",
        ("paths", "data_processed"):      "PROCESSED_DATA_PATH",
        ("mlflow", "tracking_uri"):       "MLFLOW_TRACKING_URI",
    }
    for (section, key), env_var in _env_overrides.items():
        val = os.getenv(env_var)
        if val:
            cfg[section][key] = val

    return cfg


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[1]


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-relative path to an absolute Path."""
    return get_project_root() / relative_path
