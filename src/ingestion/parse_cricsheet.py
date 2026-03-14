"""
Parse Cricsheet JSON match files into structured DataFrames.

Handles the complete Cricsheet JSON format including:
- Match metadata (teams, venue, dates, toss, outcome, event/tournament)
- Ball-by-ball delivery data (runs, wickets, extras)
- Player registry mappings
- Innings-level aggregations

Usage:
    from src.ingestion.parse_cricsheet import CricsheetParser

    parser = CricsheetParser()
    deliveries_df, matches_df = parser.parse_directory("data/raw/t20i")
"""

import json
import sys
from pathlib import Path
from typing import Optional

import polars as pl
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    POWERPLAY_END,
    MIDDLE_OVERS_END,
    DEATH_OVERS_START,
)


class CricsheetParser:
    """Parse Cricsheet JSON files into structured Polars DataFrames."""

    def __init__(self):
        self.parse_errors = []

    # ── Single Match Parsing ──────────────────────────────────

    def parse_match_file(self, filepath: Path) -> Optional[dict]:
        """
        Parse a single Cricsheet JSON match file.

        Returns a dict with:
            - match_info: dict of match-level metadata
            - deliveries: list of dicts, one per ball
            - player_registry: dict mapping player name → cricsheet ID
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.parse_errors.append({"file": str(filepath), "error": str(e)})
            return None

        match_id = filepath.stem  # filename without extension
        info = data.get("info", {})
        meta = data.get("meta", {})
        innings_data = data.get("innings", [])

        # ── Extract match info ──
        match_info = self._extract_match_info(match_id, info, meta)

        # ── Extract player registry ──
        player_registry = {}
        registry = info.get("registry", {})
        if "people" in registry:
            player_registry = registry["people"]

        # ── Extract ball-by-ball deliveries ──
        deliveries = []
        for innings_idx, innings in enumerate(innings_data):
            team = innings.get("team", "Unknown")
            target = innings.get("target", {})
            overs = innings.get("overs", [])

            for over_data in overs:
                over_num = over_data.get("over", 0)

                for ball_idx, delivery in enumerate(over_data.get("deliveries", [])):
                    ball_record = self._extract_delivery(
                        match_id=match_id,
                        innings_num=innings_idx + 1,
                        batting_team=team,
                        over_num=over_num,
                        ball_idx=ball_idx,
                        delivery=delivery,
                        match_info=match_info,
                    )
                    deliveries.append(ball_record)

        return {
            "match_info": match_info,
            "deliveries": deliveries,
            "player_registry": player_registry,
        }

    def _extract_match_info(self, match_id: str, info: dict, meta: dict) -> dict:
        """Extract match-level metadata."""
        teams = info.get("teams", [])
        dates = info.get("dates", [])
        outcome = info.get("outcome", {})
        toss = info.get("toss", {})
        event = info.get("event", {})

        # Determine winner
        winner = outcome.get("winner", None)
        result_type = None
        result_margin = None

        if "by" in outcome:
            by = outcome["by"]
            if "runs" in by:
                result_type = "runs"
                result_margin = by["runs"]
            elif "wickets" in by:
                result_type = "wickets"
                result_margin = by["wickets"]
        elif "result" in outcome:
            result_type = outcome["result"]  # e.g., "tie", "no result", "draw"

        return {
            "match_id": match_id,
            "data_version": meta.get("data_version", ""),
            "created_date": meta.get("created", ""),
            "match_type": info.get("match_type", ""),
            "match_type_number": info.get("match_type_number", None),
            "gender": info.get("gender", ""),
            "season": info.get("season", ""),
            "team1": teams[0] if len(teams) > 0 else None,
            "team2": teams[1] if len(teams) > 1 else None,
            "match_date": dates[0] if dates else None,
            "venue": info.get("venue", ""),
            "city": info.get("city", ""),
            "toss_winner": toss.get("winner", None),
            "toss_decision": toss.get("decision", None),
            "winner": winner,
            "result_type": result_type,
            "result_margin": result_margin,
            "player_of_match": (
                info.get("player_of_match", [None])[0]
                if info.get("player_of_match")
                else None
            ),
            "event_name": event.get("name", ""),
            "event_match_number": event.get("match_number", None),
            "event_group": event.get("group", ""),
            "event_stage": event.get("stage", ""),
            "overs_per_side": info.get("overs", 20),
            "balls_per_over": info.get("balls_per_over", 6),
            "team1_players": info.get("players", {}).get(teams[0], []) if teams else [],
            "team2_players": info.get("players", {}).get(teams[1], []) if len(teams) > 1 else [],
        }

    def _extract_delivery(
        self,
        match_id: str,
        innings_num: int,
        batting_team: str,
        over_num: int,
        ball_idx: int,
        delivery: dict,
        match_info: dict,
    ) -> dict:
        """Extract a single delivery/ball into a flat record."""
        runs = delivery.get("runs", {})
        extras = delivery.get("extras", {})

        # Wicket info
        wickets = delivery.get("wickets", [])
        is_wicket = len(wickets) > 0
        wicket_kind = wickets[0].get("kind", "") if is_wicket else ""
        player_out = wickets[0].get("player_out", "") if is_wicket else ""

        # Fielders involved in dismissal
        fielders = []
        if is_wicket and "fielders" in wickets[0]:
            fielders = [f.get("name", "") for f in wickets[0]["fielders"]]

        # Determine bowling team
        bowling_team = (
            match_info["team2"]
            if batting_team == match_info["team1"]
            else match_info["team1"]
        )

        # Over phase classification
        if over_num < POWERPLAY_END:
            phase = "powerplay"
        elif over_num < MIDDLE_OVERS_END:
            phase = "middle"
        else:
            phase = "death"

        # Calculate ball number in innings (1-indexed)
        ball_number_in_over = ball_idx + 1

        return {
            "match_id": match_id,
            "innings": innings_num,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "over": over_num,
            "ball": ball_number_in_over,
            "ball_id": f"{over_num}.{ball_number_in_over}",
            "phase": phase,
            "batter": delivery.get("batter", ""),
            "bowler": delivery.get("bowler", ""),
            "non_striker": delivery.get("non_striker", ""),
            "batter_runs": runs.get("batter", 0),
            "extras_runs": runs.get("extras", 0),
            "total_runs": runs.get("total", 0),
            "is_wide": "wides" in extras,
            "wide_runs": extras.get("wides", 0),
            "is_noball": "noballs" in extras,
            "noball_runs": extras.get("noballs", 0),
            "is_bye": "byes" in extras,
            "bye_runs": extras.get("byes", 0),
            "is_legbye": "legbyes" in extras,
            "legbye_runs": extras.get("legbyes", 0),
            "is_penalty": "penalty" in extras,
            "penalty_runs": extras.get("penalty", 0),
            "is_wicket": is_wicket,
            "wicket_kind": wicket_kind,
            "player_out": player_out,
            "fielders": ", ".join(fielders),
            "is_boundary_four": runs.get("batter", 0) == 4 and runs.get("non_boundary", None) is None,
            "is_boundary_six": runs.get("batter", 0) == 6 and runs.get("non_boundary", None) is None,
            "is_dot_ball": runs.get("total", 0) == 0 and not ("wides" in extras or "noballs" in extras),
            "match_date": match_info.get("match_date", ""),
            "venue": match_info.get("venue", ""),
            "event_name": match_info.get("event_name", ""),
        }

    # ── Directory Parsing ─────────────────────────────────────

    def parse_directory(
        self, directory: Path, source_label: str = "unknown"
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Parse all JSON match files in a directory.

        Args:
            directory: Path to directory containing Cricsheet JSON files
            source_label: Label for the data source (e.g., 't20i', 'ipl')

        Returns:
            (deliveries_df, matches_df, player_registry_df)
        """
        directory = Path(directory)
        json_files = sorted(directory.glob("*.json"))

        if not json_files:
            print(f"⚠️  No JSON files found in {directory}")
            return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

        print(f"\n🏏 Parsing {len(json_files)} match files from: {directory}")
        print(f"   Source: {source_label}")

        all_deliveries = []
        all_match_info = []
        all_registries = []

        for filepath in tqdm(json_files, desc=f"Parsing {source_label}"):
            result = self.parse_match_file(filepath)
            if result is None:
                continue

            # Add source label
            result["match_info"]["source"] = source_label

            all_match_info.append(result["match_info"])

            for d in result["deliveries"]:
                d["source"] = source_label
            all_deliveries.extend(result["deliveries"])

            # Player registry
            for name, pid in result["player_registry"].items():
                all_registries.append(
                    {
                        "player_name": name,
                        "cricsheet_id": pid,
                        "match_id": result["match_info"]["match_id"],
                        "source": source_label,
                    }
                )

        # Convert to Polars DataFrames
        deliveries_df = pl.DataFrame(all_deliveries) if all_deliveries else pl.DataFrame()
        
        # For matches, we need to handle the list columns specially
        matches_records = []
        for m in all_match_info:
            record = {k: v for k, v in m.items() if k not in ("team1_players", "team2_players")}
            record["team1_players"] = ", ".join(m.get("team1_players", []))
            record["team2_players"] = ", ".join(m.get("team2_players", []))
            matches_records.append(record)
        
        matches_df = pl.DataFrame(matches_records) if matches_records else pl.DataFrame()
        registry_df = pl.DataFrame(all_registries) if all_registries else pl.DataFrame()

        print(f"   ✅ Parsed: {len(all_match_info)} matches, {len(all_deliveries)} deliveries")

        if self.parse_errors:
            print(f"   ⚠️  Parse errors: {len(self.parse_errors)}")

        return deliveries_df, matches_df, registry_df

    # ── Multi-Source Parsing ──────────────────────────────────

    def parse_all_sources(
        self, source_dirs: dict[str, Path]
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Parse multiple data sources and combine into unified DataFrames.

        Args:
            source_dirs: dict mapping source label to directory path
                e.g., {"t20i": Path("data/raw/t20i"), "ipl": Path("data/raw/ipl")}

        Returns:
            (combined_deliveries_df, combined_matches_df, combined_registry_df)
        """
        all_deliveries = []
        all_matches = []
        all_registries = []

        for label, directory in source_dirs.items():
            directory = Path(directory)
            if not directory.exists():
                print(f"⚠️  Skipping {label}: directory not found ({directory})")
                continue

            deliveries_df, matches_df, registry_df = self.parse_directory(
                directory, source_label=label
            )

            if not deliveries_df.is_empty():
                all_deliveries.append(deliveries_df)
            if not matches_df.is_empty():
                all_matches.append(matches_df)
            if not registry_df.is_empty():
                all_registries.append(registry_df)

        # Combine all sources
        combined_deliveries = (
            pl.concat(all_deliveries) if all_deliveries else pl.DataFrame()
        )
        combined_matches = pl.concat(all_matches) if all_matches else pl.DataFrame()
        combined_registries = (
            pl.concat(all_registries) if all_registries else pl.DataFrame()
        )

        print("\n" + "=" * 60)
        print("📊 COMBINED DATA SUMMARY")
        print("=" * 60)
        print(f"  Total matches:    {combined_matches.height}")
        print(f"  Total deliveries: {combined_deliveries.height}")
        print(f"  Total registry:   {combined_registries.height}")
        if self.parse_errors:
            print(f"  Parse errors:     {len(self.parse_errors)}")
        print("=" * 60)

        return combined_deliveries, combined_matches, combined_registries


def load_people_register(filepath: Path) -> pl.DataFrame:
    """Load the Cricsheet People Register CSV."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"⚠️  People register not found: {filepath}")
        return pl.DataFrame()

    df = pl.read_csv(filepath)
    print(f"✅ Loaded People Register: {df.height} entries, {df.width} columns")
    print(f"   Columns: {df.columns}")
    return df


# ── CLI Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    from config.settings import RAW_T20I_DIR, RAW_IPL_DIR, RAW_BBL_DIR, RAW_CPL_DIR, RAW_REGISTER_DIR

    parser = CricsheetParser()

    # Parse whichever directories exist
    source_dirs = {}
    for label, directory in [
        ("t20i", RAW_T20I_DIR),
        ("ipl", RAW_IPL_DIR),
        ("bbl", RAW_BBL_DIR),
        ("cpl", RAW_CPL_DIR),
    ]:
        if directory.exists() and any(directory.glob("*.json")):
            source_dirs[label] = directory

    if not source_dirs:
        print("❌ No data found. Run download_data.py first:")
        print("   python src/ingestion/download_data.py")
        sys.exit(1)

    deliveries_df, matches_df, registry_df = parser.parse_all_sources(source_dirs)

    # Show sample data
    if not matches_df.is_empty():
        print("\n📋 Sample Match Data:")
        print(matches_df.select(["match_id", "team1", "team2", "venue", "winner", "source"]).head(5))

    if not deliveries_df.is_empty():
        print("\n📋 Sample Delivery Data:")
        print(
            deliveries_df.select(
                ["match_id", "innings", "over", "ball", "batter", "bowler", "total_runs", "is_wicket"]
            ).head(10)
        )

    # Load people register
    register_path = RAW_REGISTER_DIR / "people.csv"
    if register_path.exists():
        people_df = load_people_register(register_path)
        print(f"\n📋 People Register Sample:")
        print(people_df.head(5))
