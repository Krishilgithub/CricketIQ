"""
src/ingestion/convert_new_json_to_csv.py
─────────────────────────────────────────
Converts Cricsheet JSON match files (from a drop folder) into normalized
CSV rows across 8 tables:
  matches, match_teams, innings, deliveries, wickets,
  powerplays, player_of_match, officials

This mirrors what Cricsheet provides in their pre-built CSVs, so new JSON
drops can be incrementally appended without waiting for Cricsheet releases.

Usage:
    python -m src.ingestion.convert_new_json_to_csv --json-dir data/raw/new_json_drops
    python -m src.ingestion.convert_new_json_to_csv --json-dir data/raw/new_json_drops --out-dir data/raw/cricsheet_csv_all
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from src.logger import get_logger

log = get_logger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(d: dict, *keys, default=None):
    """Safely traverse nested dict by successive keys."""
    v = d
    for k in keys:
        if not isinstance(v, dict):
            return default
        v = v.get(k, default)
    return v


def _flatten_match(match_id: str, raw: dict) -> dict:
    """Extract match-level fields from Cricsheet JSON info block."""
    info = raw.get("info", {})
    dates = info.get("dates", [])
    match_date = dates[0] if dates else None

    event = info.get("event", {})
    event_name = event.get("name") if isinstance(event, dict) else None
    event_match_number = event.get("match_number") if isinstance(event, dict) else None

    outcome = info.get("outcome", {})
    winner = _safe(outcome, "winner")
    result_type = _safe(outcome, "result")
    result_margin = None
    result_text = None
    method = _safe(outcome, "method")

    by = outcome.get("by", {})
    if isinstance(by, dict) and by:
        result_type_key = list(by.keys())[0]
        result_margin = list(by.values())[0]
        result_text = f"{winner} won by {result_margin} {result_type_key}"

    toss = info.get("toss", {})

    return {
        "match_id": match_id,
        "data_version": _safe(raw, "meta", "data_version"),
        "created": _safe(raw, "meta", "created"),
        "revision": _safe(raw, "meta", "revision"),
        "match_date": match_date,
        "season": info.get("season"),
        "event_name": event_name,
        "event_match_number": event_match_number,
        "match_type": info.get("match_type"),
        "match_type_number": info.get("match_type_number"),
        "gender": info.get("gender"),
        "team_type": info.get("team_type"),
        "venue": info.get("venue"),
        "city": info.get("city"),
        "overs": info.get("overs"),
        "balls_per_over": info.get("balls_per_over", 6),
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "winner": winner,
        "result_type": result_type,
        "result_margin": result_margin,
        "result_text": result_text,
        "method": method,
    }


def _flatten_teams(match_id: str, raw: dict) -> list[dict]:
    """Extract team rows (one per team) from JSON info block."""
    info = raw.get("info", {})
    teams = info.get("teams", [])
    return [{"match_id": match_id, "team": t} for t in teams]


def _flatten_innings(match_id: str, raw: dict) -> list[dict]:
    """Extract innings-level aggregates from the full innings list."""
    rows = []
    innings_list = raw.get("innings", [])
    for idx, inn in enumerate(innings_list, start=1):
        team = inn.get("team", "")
        deliveries_raw = inn.get("overs", [])

        total_runs = total_wickets = total_balls = 0
        extras_byes = extras_legbyes = extras_noballs = extras_wides = extras_penalty = 0

        for over_obj in deliveries_raw:
            for d in over_obj.get("deliveries", []):
                runs = d.get("runs", {})
                total_runs += runs.get("total", 0)
                total_balls += 1
                extras = d.get("extras", {})
                extras_byes += extras.get("byes", 0)
                extras_legbyes += extras.get("legbyes", 0)
                extras_noballs += extras.get("noballs", 0)
                extras_wides += extras.get("wides", 0)
                extras_penalty += extras.get("penalty", 0)
                if "wickets" in d:
                    total_wickets += len(d["wickets"])

        rows.append({
            "match_id": match_id,
            "innings_number": idx,
            "team": team,
            "total_runs": total_runs,
            "total_wickets": total_wickets,
            "total_balls": total_balls,
            "extras_byes": extras_byes,
            "extras_legbyes": extras_legbyes,
            "extras_noballs": extras_noballs,
            "extras_wides": extras_wides,
            "extras_penalty": extras_penalty,
        })
    return rows


def _flatten_deliveries_wickets(match_id: str, raw: dict) -> tuple[list[dict], list[dict]]:
    """Extract ball-by-ball delivery rows and wicket rows from innings."""
    deliveries = []
    wickets = []
    innings_list = raw.get("innings", [])

    for inn_idx, inn in enumerate(innings_list, start=1):
        batting_team = inn.get("team", "")
        for over_obj in inn.get("overs", []):
            over_num = over_obj.get("over", 0)
            for ball_idx, d in enumerate(over_obj.get("deliveries", []), start=1):
                runs = d.get("runs", {})
                extras = d.get("extras", {})
                review = d.get("review", {})
                replacement = d.get("replacements", {})

                row = {
                    "match_id": match_id,
                    "innings_number": inn_idx,
                    "over": over_num,
                    "ball_in_over": ball_idx,
                    "batting_team": batting_team,
                    "batter": d.get("batter"),
                    "bowler": d.get("bowler"),
                    "non_striker": d.get("non_striker"),
                    "runs_batter": runs.get("batter", 0),
                    "runs_extras": runs.get("extras", 0),
                    "runs_total": runs.get("total", 0),
                    "extras_byes": extras.get("byes", 0),
                    "extras_legbyes": extras.get("legbyes", 0),
                    "extras_noballs": extras.get("noballs", 0),
                    "extras_wides": extras.get("wides", 0),
                    "extras_penalty": extras.get("penalty", 0),
                    "review_by": review.get("by"),
                    "review_batter": review.get("batter"),
                    "review_decision": review.get("decision"),
                    "review_type": review.get("type"),
                    "replacement_role": None,
                    "replacement_team": None,
                    "replacement_in": None,
                    "replacement_out": None,
                }

                # Parse replacement info (nested structure)
                for role, reps in replacement.items():
                    for rep in (reps if isinstance(reps, list) else [reps]):
                        row["replacement_role"] = role
                        row["replacement_team"] = rep.get("team")
                        row["replacement_in"] = rep.get("in")
                        row["replacement_out"] = rep.get("out")

                deliveries.append(row)

                # Extract wickets for this delivery
                for wkt in d.get("wickets", []):
                    fielders_list = wkt.get("fielders", [])
                    fielders_str = "|".join(
                        f.get("name", "") if isinstance(f, dict) else str(f)
                        for f in fielders_list
                    ) if fielders_list else None
                    wickets.append({
                        "match_id": match_id,
                        "innings_number": inn_idx,
                        "over": over_num,
                        "ball_in_over": ball_idx,
                        "batting_team": batting_team,
                        "player_out": wkt.get("player_out"),
                        "kind": wkt.get("kind"),
                        "fielders": fielders_str,
                    })

    return deliveries, wickets


def _flatten_powerplays(match_id: str, raw: dict) -> list[dict]:
    rows = []
    for inn_idx, inn in enumerate(raw.get("innings", []), start=1):
        for pp in inn.get("powerplays", []):
            rows.append({
                "match_id": match_id,
                "innings_number": inn_idx,
                "powerplay_type": pp.get("type"),
                "from_over": pp.get("from"),
                "to_over": pp.get("to"),
            })
    return rows


def _flatten_player_of_match(match_id: str, raw: dict) -> list[dict]:
    pom = raw.get("info", {}).get("player_of_match", [])
    return [{"match_id": match_id, "player": p} for p in (pom or [])]


def _flatten_officials(match_id: str, raw: dict) -> list[dict]:
    rows = []
    officials = raw.get("info", {}).get("officials", {})
    for role, names in officials.items():
        for name in (names or []):
            rows.append({
                "match_id": match_id,
                "official_role": role,
                "official_name": name,
            })
    return rows


# ── Main converter ────────────────────────────────────────────────────────────

def convert_json_file(json_path: Path) -> dict[str, list[dict]]:
    """
    Parse a single Cricsheet JSON file and return all table rows as lists.
    The match_id is derived from the filename stem (without extension).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Completely exclude Women's matches from the dataset
    gender = _safe(raw, "info", "gender")
    if gender and str(gender).lower() == "female":
        return {
            "matches": [], "match_teams": [], "innings": [], 
            "deliveries": [], "wickets": [], "powerplays": [], 
            "player_of_match": [], "officials": []
        }

    match_id = json_path.stem  # e.g. "1234567" from "1234567.json"
    deliveries, wickets = _flatten_deliveries_wickets(match_id, raw)

    return {
        "matches":         [_flatten_match(match_id, raw)],
        "match_teams":     _flatten_teams(match_id, raw),
        "innings":         _flatten_innings(match_id, raw),
        "deliveries":      deliveries,
        "wickets":         wickets,
        "powerplays":      _flatten_powerplays(match_id, raw),
        "player_of_match": _flatten_player_of_match(match_id, raw),
        "officials":       _flatten_officials(match_id, raw),
    }


def convert_json_folder(json_dir: Path, out_dir: Path, append: bool = True) -> None:
    """
    Convert all JSON files in json_dir and append (or overwrite) CSVs in out_dir.
    If append=True, rows are appended to existing CSVs (no dedup here; use
    ingest_historical.py for idempotent DB loads).
    """
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        log.warning(f"No JSON files found in {json_dir}")
        return

    log.info(f"Converting {len(json_files)} JSON files from {json_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate into per-table buffers
    buffers: dict[str, list[dict]] = {t: [] for t in [
        "matches", "match_teams", "innings", "deliveries",
        "wickets", "powerplays", "player_of_match", "officials",
    ]}

    for jf in json_files:
        try:
            result = convert_json_file(jf)
            for table, rows in result.items():
                buffers[table].extend(rows)
        except Exception as e:
            log.error(f"Failed to parse {jf.name}: {e}")

    # Write CSVs
    for table, rows in buffers.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        csv_path = out_dir / f"{table}.csv"
        mode = "a" if (append and csv_path.exists()) else "w"
        header = not (append and csv_path.exists())
        df.to_csv(csv_path, index=False, mode=mode, header=header)
        log.info(f"  {table}: wrote {len(df):,} rows → {csv_path} (mode={mode})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Cricsheet JSON drop folder to normalized CSVs"
    )
    parser.add_argument("--json-dir", required=True, help="Folder containing .json files")
    parser.add_argument(
        "--out-dir",
        default="data/raw/cricsheet_csv_all",
        help="Output folder for CSV tables (default: data/raw/cricsheet_csv_all)",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Overwrite CSVs instead of appending",
    )
    args = parser.parse_args()
    convert_json_folder(
        Path(args.json_dir),
        Path(args.out_dir),
        append=not args.no_append,
    )


if __name__ == "__main__":
    main()
