"""
pytest configuration and shared fixtures for CricketIQ tests.
"""

import pytest
from pathlib import Path
import pandas as pd


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute project root directory."""
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def config():
    """Return the parsed project config."""
    from src.config import get_config
    return get_config()


@pytest.fixture
def sample_matches_df() -> pd.DataFrame:
    """Minimal synthetic matches DataFrame for use in unit tests."""
    return pd.DataFrame({
        "match_id": ["m001", "m002", "m003"],
        "match_date": pd.to_datetime(["2023-06-01", "2023-06-05", "2023-06-10"]),
        "venue": ["Wankhede Stadium", "Eden Gardens", "Wankhede Stadium"],
        "city": ["Mumbai", "Kolkata", "Mumbai"],
        "toss_winner": ["India", "Australia", "India"],
        "toss_decision": ["bat", "field", "field"],
        "winner": ["India", "Australia", "India"],
        "team_1": ["Australia", "Australia", "India"],
        "team_2": ["India", "India", "Pakistan"],
        "team_1_win": [0, 1, 1],
    })


@pytest.fixture
def sample_deliveries_df() -> pd.DataFrame:
    """Minimal synthetic deliveries DataFrame for unit tests."""
    return pd.DataFrame({
        "match_id":       ["m001"] * 12,
        "innings_number": [1] * 6 + [2] * 6,
        "over":           [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        "ball_in_over":   [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        "batting_team":   ["India"] * 6 + ["Australia"] * 6,
        "batter":         ["Rohit"] * 6 + ["Warner"] * 6,
        "bowler":         ["Starc"] * 6 + ["Bumrah"] * 6,
        "runs_batter":    [4, 1, 0, 6, 0, 2, 1, 4, 0, 6, 1, 0],
        "runs_extras":    [0] * 12,
        "runs_total":     [4, 1, 0, 6, 0, 2, 1, 4, 0, 6, 1, 0],
        "extras_wides":   [0] * 12,
        "extras_noballs": [0] * 12,
    })
