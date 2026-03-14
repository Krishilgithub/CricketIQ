from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def safe_json_dumps(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def first_date(info: dict[str, Any]) -> str:
    dates = info.get("dates", [])
    if isinstance(dates, list) and dates:
        return str(dates[0])
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Cricsheet JSON files to normalized CSV tables"
    )
    parser.add_argument(
        "--input-dir",
        default="data/t20s_json",
        help="Directory containing Cricsheet JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/cricsheet_csv",
        help="Directory where CSV files will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in: {input_dir}")

    matches_path = output_dir / "matches.csv"
    teams_path = output_dir / "match_teams.csv"
    pom_path = output_dir / "player_of_match.csv"
    officials_path = output_dir / "officials.csv"
    innings_path = output_dir / "innings.csv"
    powerplays_path = output_dir / "powerplays.csv"
    deliveries_path = output_dir / "deliveries.csv"
    wickets_path = output_dir / "wickets.csv"

    with (
        matches_path.open("w", newline="", encoding="utf-8") as f_matches,
        teams_path.open("w", newline="", encoding="utf-8") as f_teams,
        pom_path.open("w", newline="", encoding="utf-8") as f_pom,
        officials_path.open("w", newline="", encoding="utf-8") as f_officials,
        innings_path.open("w", newline="", encoding="utf-8") as f_innings,
        powerplays_path.open("w", newline="", encoding="utf-8") as f_powerplays,
        deliveries_path.open("w", newline="", encoding="utf-8") as f_deliveries,
        wickets_path.open("w", newline="", encoding="utf-8") as f_wickets,
    ):
        w_matches = csv.writer(f_matches)
        w_teams = csv.writer(f_teams)
        w_pom = csv.writer(f_pom)
        w_officials = csv.writer(f_officials)
        w_innings = csv.writer(f_innings)
        w_powerplays = csv.writer(f_powerplays)
        w_deliveries = csv.writer(f_deliveries)
        w_wickets = csv.writer(f_wickets)

        w_matches.writerow(
            [
                "match_id",
                "data_version",
                "created",
                "revision",
                "match_date",
                "season",
                "event_name",
                "event_match_number",
                "match_type",
                "match_type_number",
                "gender",
                "team_type",
                "venue",
                "city",
                "overs",
                "balls_per_over",
                "toss_winner",
                "toss_decision",
                "winner",
                "result_type",
                "result_margin",
                "result_text",
                "method",
            ]
        )

        w_teams.writerow(["match_id", "team"])
        w_pom.writerow(["match_id", "player"])
        w_officials.writerow(["match_id", "official_role", "official_name"])

        w_innings.writerow(
            [
                "match_id",
                "innings_number",
                "team",
                "total_runs",
                "total_wickets",
                "total_balls",
                "extras_byes",
                "extras_legbyes",
                "extras_noballs",
                "extras_wides",
                "extras_penalty",
            ]
        )

        w_powerplays.writerow(
            [
                "match_id",
                "innings_number",
                "powerplay_type",
                "from_over",
                "to_over",
            ]
        )

        w_deliveries.writerow(
            [
                "match_id",
                "innings_number",
                "over",
                "ball_in_over",
                "batting_team",
                "batter",
                "bowler",
                "non_striker",
                "runs_batter",
                "runs_extras",
                "runs_total",
                "extras_byes",
                "extras_legbyes",
                "extras_noballs",
                "extras_wides",
                "extras_penalty",
                "review_by",
                "review_batter",
                "review_decision",
                "review_type",
                "replacement_role",
                "replacement_team",
                "replacement_in",
                "replacement_out",
            ]
        )

        w_wickets.writerow(
            [
                "match_id",
                "innings_number",
                "over",
                "ball_in_over",
                "batting_team",
                "player_out",
                "kind",
                "fielders",
            ]
        )

        for json_file in json_files:
            match_id = json_file.stem
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            meta = data.get("meta", {})
            info = data.get("info", {})

            outcome = info.get("outcome", {}) or {}
            toss = info.get("toss", {}) or {}
            event = info.get("event", {})

            winner = outcome.get("winner", "")
            method = outcome.get("method", "")
            result_type = ""
            result_margin: int | str = ""
            result_text = ""

            by = outcome.get("by", {})
            if isinstance(by, dict) and by:
                result_type = next(iter(by.keys()))
                result_margin = by.get(result_type, "")
                result_text = f"{winner} won by {result_margin} {result_type}" if winner else ""
            elif outcome.get("result"):
                result_text = str(outcome.get("result"))

            w_matches.writerow(
                [
                    match_id,
                    meta.get("data_version", ""),
                    meta.get("created", ""),
                    meta.get("revision", ""),
                    first_date(info),
                    info.get("season", ""),
                    event.get("name", "") if isinstance(event, dict) else str(event),
                    event.get("match_number", "") if isinstance(event, dict) else "",
                    info.get("match_type", ""),
                    info.get("match_type_number", ""),
                    info.get("gender", ""),
                    info.get("team_type", ""),
                    info.get("venue", ""),
                    info.get("city", ""),
                    info.get("overs", ""),
                    info.get("balls_per_over", ""),
                    toss.get("winner", ""),
                    toss.get("decision", ""),
                    winner,
                    result_type,
                    result_margin,
                    result_text,
                    method,
                ]
            )

            for team in info.get("teams", []) or []:
                w_teams.writerow([match_id, team])

            for pom in info.get("player_of_match", []) or []:
                w_pom.writerow([match_id, pom])

            officials = info.get("officials", {}) or {}
            if isinstance(officials, dict):
                for role, names in officials.items():
                    if isinstance(names, list):
                        for name in names:
                            w_officials.writerow([match_id, role, name])

            innings_list = data.get("innings", []) or []
            for innings_index, innings in enumerate(innings_list, start=1):
                team = innings.get("team", "")
                overs = innings.get("overs", []) or []
                powerplays = innings.get("powerplays", []) or []

                total_runs = 0
                total_wickets = 0
                total_balls = 0
                innings_extras = {
                    "byes": 0,
                    "legbyes": 0,
                    "noballs": 0,
                    "wides": 0,
                    "penalty": 0,
                }

                for pp in powerplays:
                    w_powerplays.writerow(
                        [
                            match_id,
                            innings_index,
                            pp.get("type", ""),
                            pp.get("from", ""),
                            pp.get("to", ""),
                        ]
                    )

                for over_obj in overs:
                    over_num = over_obj.get("over", "")
                    deliveries = over_obj.get("deliveries", []) or []
                    for ball_idx, delivery in enumerate(deliveries, start=1):
                        total_balls += 1
                        runs = delivery.get("runs", {}) or {}
                        extras = delivery.get("extras", {}) or {}
                        review = delivery.get("review", {}) or {}
                        replacement = delivery.get("replacements", {}) or {}

                        runs_batter = runs.get("batter", 0)
                        runs_extras = runs.get("extras", 0)
                        runs_total = runs.get("total", 0)

                        total_runs += int(runs_total or 0)

                        byes = int(extras.get("byes", 0) or 0)
                        legbyes = int(extras.get("legbyes", 0) or 0)
                        noballs = int(extras.get("noballs", 0) or 0)
                        wides = int(extras.get("wides", 0) or 0)
                        penalty = int(extras.get("penalty", 0) or 0)

                        innings_extras["byes"] += byes
                        innings_extras["legbyes"] += legbyes
                        innings_extras["noballs"] += noballs
                        innings_extras["wides"] += wides
                        innings_extras["penalty"] += penalty

                        review_by = ""
                        review_batter = ""
                        review_decision = ""
                        review_type = ""
                        if isinstance(review, dict):
                            review_by = review.get("by", "")
                            review_batter = review.get("batter", "")
                            review_decision = review.get("decision", "")
                            review_type = review.get("type", "")

                        replacement_role = ""
                        replacement_team = ""
                        replacement_in = ""
                        replacement_out = ""
                        if isinstance(replacement, dict):
                            replacement_role = replacement.get("role", "")
                            replacement_team = replacement.get("team", "")
                            match_like = replacement.get("match", {})
                            if isinstance(match_like, dict):
                                replacement_in = match_like.get("in", "")
                                replacement_out = match_like.get("out", "")

                        w_deliveries.writerow(
                            [
                                match_id,
                                innings_index,
                                over_num,
                                ball_idx,
                                team,
                                delivery.get("batter", ""),
                                delivery.get("bowler", ""),
                                delivery.get("non_striker", ""),
                                runs_batter,
                                runs_extras,
                                runs_total,
                                byes,
                                legbyes,
                                noballs,
                                wides,
                                penalty,
                                review_by,
                                review_batter,
                                review_decision,
                                review_type,
                                replacement_role,
                                replacement_team,
                                replacement_in,
                                replacement_out,
                            ]
                        )

                        wickets = delivery.get("wickets", []) or []
                        for wicket in wickets:
                            total_wickets += 1
                            fielders = wicket.get("fielders", [])
                            fielder_names: list[str] = []
                            if isinstance(fielders, list):
                                for fielder in fielders:
                                    if isinstance(fielder, dict):
                                        name = fielder.get("name")
                                        if name:
                                            fielder_names.append(str(name))
                                    elif isinstance(fielder, str):
                                        fielder_names.append(fielder)

                            w_wickets.writerow(
                                [
                                    match_id,
                                    innings_index,
                                    over_num,
                                    ball_idx,
                                    team,
                                    wicket.get("player_out", ""),
                                    wicket.get("kind", ""),
                                    "|".join(fielder_names),
                                ]
                            )

                w_innings.writerow(
                    [
                        match_id,
                        innings_index,
                        team,
                        total_runs,
                        total_wickets,
                        total_balls,
                        innings_extras["byes"],
                        innings_extras["legbyes"],
                        innings_extras["noballs"],
                        innings_extras["wides"],
                        innings_extras["penalty"],
                    ]
                )

    print(f"Converted {len(json_files)} JSON files from {input_dir} to CSVs in {output_dir}")


if __name__ == "__main__":
    main()
