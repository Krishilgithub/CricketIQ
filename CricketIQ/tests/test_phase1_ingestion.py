"""Tests for Phase 1 ingestion modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# ── Canonical mappings tests ───────────────────────────────────────────────────

from src.ingestion.canonical_mappings import (
    standardize_team,
    standardize_venue,
    apply_canonical_teams,
    apply_canonical_venues,
    TEAM_ALIASES,
    VENUE_ALIASES,
)


def test_standardize_team_known_alias():
    assert standardize_team("U.A.E.") == "United Arab Emirates"
    assert standardize_team("UAE") == "United Arab Emirates"
    assert standardize_team("AUS") == "Australia"


def test_standardize_team_passthrough():
    assert standardize_team("India") == "India"
    assert standardize_team("SomeUnknownTeam") == "SomeUnknownTeam"


def test_standardize_team_none():
    assert standardize_team(None) is None


def test_standardize_venue_known_alias():
    assert standardize_venue("Dubai (DSC)") == "Dubai International Cricket Stadium"
    assert standardize_venue("Feroz Shah Kotla") == "Arun Jaitley Stadium"
    assert standardize_venue("MCG") == "Melbourne Cricket Ground"


def test_standardize_venue_passthrough():
    assert standardize_venue("Wankhede Stadium") == "Wankhede Stadium"


def test_apply_canonical_teams():
    df = pd.DataFrame({
        "team_1": ["U.A.E.", "India"],
        "team_2": ["AUS", "Pakistan"],
    })
    result = apply_canonical_teams(df, ["team_1", "team_2"])
    assert result["team_1"].tolist() == ["United Arab Emirates", "India"]
    assert result["team_2"].tolist() == ["Australia", "Pakistan"]


def test_apply_canonical_venues():
    df = pd.DataFrame({"venue": ["MCG", "Wankhede Stadium", "Dubai (DSC)"]})
    result = apply_canonical_venues(df, ["venue"])
    assert result["venue"].tolist() == [
        "Melbourne Cricket Ground",
        "Wankhede Stadium",
        "Dubai International Cricket Stadium",
    ]


# ── JSON converter tests ───────────────────────────────────────────────────────

from src.ingestion.convert_new_json_to_csv import convert_json_file


SAMPLE_JSON = {
    "meta": {"data_version": "1.1.0", "created": "2024-06-01", "revision": 1},
    "info": {
        "match_type": "T20",
        "gender": "male",
        "team_type": "international",
        "teams": ["India", "Australia"],
        "venue": "Wankhede Stadium",
        "city": "Mumbai",
        "dates": ["2024-06-01"],
        "season": "2024",
        "overs": 20,
        "balls_per_over": 6,
        "toss": {"winner": "India", "decision": "bat"},
        "outcome": {"winner": "India", "by": {"runs": 25}},
        "player_of_match": ["V Kohli"],
        "officials": {"umpires": ["S Ravi", "M Erasmus"]},
        "event": {"name": "ICC T20 WC 2024", "match_number": 1},
    },
    "innings": [
        {
            "team": "India",
            "overs": [
                {
                    "over": 0,
                    "deliveries": [
                        {
                            "batter": "RG Sharma",
                            "bowler": "M Starc",
                            "non_striker": "V Kohli",
                            "runs": {"batter": 4, "extras": 0, "total": 4},
                        },
                        {
                            "batter": "RG Sharma",
                            "bowler": "M Starc",
                            "non_striker": "V Kohli",
                            "runs": {"batter": 0, "extras": 1, "total": 1},
                            "extras": {"wides": 1},
                        },
                        {
                            "batter": "RG Sharma",
                            "bowler": "M Starc",
                            "non_striker": "V Kohli",
                            "runs": {"batter": 6, "extras": 0, "total": 6},
                            "wickets": [
                                {"player_out": "RG Sharma", "kind": "caught",
                                 "fielders": [{"name": "DA Warner"}]}
                            ],
                        },
                    ],
                }
            ],
            "powerplays": [{"type": "mandatory", "from": 1, "to": 6}],
        },
        {
            "team": "Australia",
            "overs": [
                {
                    "over": 0,
                    "deliveries": [
                        {
                            "batter": "DA Warner",
                            "bowler": "JJ Bumrah",
                            "non_striker": "DA Warner",
                            "runs": {"batter": 1, "extras": 0, "total": 1},
                        }
                    ],
                }
            ],
        },
    ],
}


@pytest.fixture
def sample_json_file(tmp_path: Path) -> Path:
    jf = tmp_path / "test_match_001.json"
    jf.write_text(json.dumps(SAMPLE_JSON))
    return jf


def test_convert_returns_all_tables(sample_json_file):
    result = convert_json_file(sample_json_file)
    for table in ["matches", "match_teams", "innings", "deliveries", "wickets",
                  "powerplays", "player_of_match", "officials"]:
        assert table in result, f"Missing table: {table}"


def test_match_row_fields(sample_json_file):
    result = convert_json_file(sample_json_file)
    match = result["matches"][0]
    assert match["match_id"] == "test_match_001"
    assert match["match_type"] == "T20"
    assert match["winner"] == "India"
    assert match["toss_winner"] == "India"
    assert match["venue"] == "Wankhede Stadium"


def test_match_teams_rows(sample_json_file):
    result = convert_json_file(sample_json_file)
    teams = {r["team"] for r in result["match_teams"]}
    assert teams == {"India", "Australia"}


def test_innings_aggregates(sample_json_file):
    result = convert_json_file(sample_json_file)
    innings = result["innings"]
    assert len(innings) == 2
    ind = next(i for i in innings if i["team"] == "India")
    # 4 + 1 (wide) + 6 = 11 total runs
    assert ind["total_runs"] == 11
    assert ind["total_wickets"] == 1
    assert ind["extras_wides"] == 1


def test_deliveries_parsed(sample_json_file):
    result = convert_json_file(sample_json_file)
    assert len(result["deliveries"]) == 4  # 3 India + 1 Australia


def test_wickets_parsed(sample_json_file):
    result = convert_json_file(sample_json_file)
    assert len(result["wickets"]) == 1
    w = result["wickets"][0]
    assert w["player_out"] == "RG Sharma"
    assert w["kind"] == "caught"
    assert "DA Warner" in w["fielders"]


def test_powerplays_parsed(sample_json_file):
    result = convert_json_file(sample_json_file)
    assert len(result["powerplays"]) == 1
    pp = result["powerplays"][0]
    assert pp["powerplay_type"] == "mandatory"


def test_player_of_match_parsed(sample_json_file):
    result = convert_json_file(sample_json_file)
    assert result["player_of_match"][0]["player"] == "V Kohli"


def test_officials_parsed(sample_json_file):
    result = convert_json_file(sample_json_file)
    names = {r["official_name"] for r in result["officials"]}
    assert "S Ravi" in names
    assert "M Erasmus" in names


# ── Historical ingestion smoke test ───────────────────────────────────────────

import duckdb
from src.ingestion.ingest_historical import ingest_cricsheet, print_bronze_row_counts


def test_ingest_single_match_to_duckdb(sample_json_file, tmp_path):
    """Convert one JSON → CSV folder → ingest into temp DuckDB."""
    from src.ingestion.convert_new_json_to_csv import convert_json_folder

    csv_dir = tmp_path / "csv_out"
    convert_json_folder(sample_json_file.parent, csv_dir, append=False)

    db_path = str(tmp_path / "test.duckdb")
    summary = ingest_cricsheet(db_path, csv_dir)

    assert summary.get("matches", 0) == 1
    assert summary.get("match_teams", 0) == 2
    assert summary.get("deliveries", 0) == 4
    assert summary.get("wickets", 0) == 1

    # Verify DB content
    con = duckdb.connect(db_path, read_only=True)
    count = con.execute("SELECT COUNT(*) FROM bronze.matches").fetchone()[0]
    con.close()
    assert count == 1


def test_ingest_idempotent(sample_json_file, tmp_path):
    """Running ingestion twice should not duplicate rows."""
    from src.ingestion.convert_new_json_to_csv import convert_json_folder

    csv_dir = tmp_path / "csv_out"
    convert_json_folder(sample_json_file.parent, csv_dir, append=False)

    db_path = str(tmp_path / "test.duckdb")
    ingest_cricsheet(db_path, csv_dir)
    ingest_cricsheet(db_path, csv_dir)   # run again

    con = duckdb.connect(db_path, read_only=True)
    count = con.execute("SELECT COUNT(*) FROM bronze.matches").fetchone()[0]
    con.close()
    assert count == 1, "Second run should not insert duplicate rows"
