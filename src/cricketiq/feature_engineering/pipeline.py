from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    team_form_window: int
    match_filters: Dict[str, str]


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_config(config_path: Path) -> PipelineConfig:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return PipelineConfig(
        input_dir=Path(data["input_dir"]),
        output_dir=Path(data["output_dir"]),
        team_form_window=int(data.get("team_form_window", 5)),
        match_filters=data.get("match_filters", {}),
    )


def _build_match_team_lookup(match_teams_rows: List[Dict[str, str]]) -> Dict[str, List[str]]:
    teams_by_match: Dict[str, List[str]] = defaultdict(list)
    for row in match_teams_rows:
        match_id = row["match_id"]
        team = row["team"].strip()
        if team and team not in teams_by_match[match_id]:
            teams_by_match[match_id].append(team)

    for match_id, teams in teams_by_match.items():
        teams.sort()
        teams_by_match[match_id] = teams[:2]
    return teams_by_match


def _initialize_match_records(
    matches_rows: List[Dict[str, str]],
    teams_by_match: Dict[str, List[str]],
    filters: Dict[str, str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for row in matches_rows:
        match_id = row["match_id"]
        teams = teams_by_match.get(match_id, [])
        if len(teams) != 2:
            continue

        keep = True
        for key, expected in filters.items():
            if row.get(key, "") != expected:
                keep = False
                break
        if not keep:
            continue

        team_1, team_2 = teams[0], teams[1]
        winner = row.get("winner", "").strip()
        target = ""
        outcome_class = "no_result"

        if winner == team_1:
            target = "1"
            outcome_class = "win"
        elif winner == team_2:
            target = "0"
            outcome_class = "loss"
        elif row.get("result_text", "").strip().lower().startswith("tie"):
            outcome_class = "tie"

        records.append(
            {
                "match_id": match_id,
                "match_date": row["match_date"],
                "season": row.get("season", ""),
                "venue": row.get("venue", ""),
                "city": row.get("city", ""),
                "team_1": team_1,
                "team_2": team_2,
                "toss_winner": row.get("toss_winner", ""),
                "toss_decision": row.get("toss_decision", ""),
                "winner": winner,
                "result_type": row.get("result_type", ""),
                "result_margin": row.get("result_margin", ""),
                "team_1_win": target,
                "outcome_class": outcome_class,
            }
        )

    records.sort(key=lambda r: (r["match_date"], _safe_int(r["match_id"])))
    return records


def _aggregate_innings(
    innings_rows: List[Dict[str, str]],
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], Dict[str, float]]:
    by_match_team: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    teams_per_match: Dict[str, Dict[str, float]] = defaultdict(dict)
    first_innings_runs_by_match: Dict[str, float] = {}

    for row in innings_rows:
        innings_number = _safe_int(row.get("innings_number", "0"))
        if innings_number > 2:
            continue

        match_id = row["match_id"]
        team = row["team"]
        key = (match_id, team)

        runs = _safe_int(row.get("total_runs", "0"))
        wickets = _safe_int(row.get("total_wickets", "0"))
        balls = _safe_int(row.get("total_balls", "0"))

        by_match_team[key]["runs_scored"] += runs
        by_match_team[key]["wickets_lost"] += wickets
        by_match_team[key]["balls_faced"] += balls

        if innings_number == 1:
            first_innings_runs_by_match[match_id] = float(runs)

        teams_per_match[match_id][team] = runs

    for match_id, score_map in teams_per_match.items():
        teams = list(score_map.keys())
        if len(teams) < 2:
            continue
        for team in teams:
            opp_runs = sum(score_map[t] for t in teams if t != team)
            by_match_team[(match_id, team)]["runs_conceded"] = opp_runs

    return by_match_team, first_innings_runs_by_match


def _aggregate_deliveries(
    deliveries_rows: List[Dict[str, str]],
    teams_by_match: Dict[str, List[str]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    by_match_team: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for row in deliveries_rows:
        innings_number = _safe_int(row.get("innings_number", "0"))
        if innings_number > 2:
            continue

        match_id = row["match_id"]
        batting_team = row["batting_team"]
        teams = teams_by_match.get(match_id, [])
        if len(teams) != 2 or batting_team not in teams:
            continue

        bowling_team = teams[1] if batting_team == teams[0] else teams[0]

        over = _safe_int(row.get("over", "0"))
        runs_total = _safe_int(row.get("runs_total", "0"))
        wides = _safe_int(row.get("extras_wides", "0"))
        noballs = _safe_int(row.get("extras_noballs", "0"))
        legal_ball = 1 if (wides == 0 and noballs == 0) else 0

        bat_key = (match_id, batting_team)
        bowl_key = (match_id, bowling_team)

        by_match_team[bat_key]["balls_seen"] += 1
        by_match_team[bat_key]["legal_balls_seen"] += legal_ball
        by_match_team[bat_key]["dot_balls"] += 1 if runs_total == 0 else 0

        if over <= 5:
            by_match_team[bat_key]["powerplay_runs"] += runs_total
            by_match_team[bat_key]["powerplay_legal_balls"] += legal_ball

        if over >= 15:
            by_match_team[bat_key]["death_runs"] += runs_total
            by_match_team[bat_key]["death_legal_balls"] += legal_ball

        by_match_team[bowl_key]["wides_conceded"] += wides
        by_match_team[bowl_key]["noballs_conceded"] += noballs
        by_match_team[bowl_key]["balls_bowled"] += 1

    return by_match_team


def _aggregate_wickets(
    wickets_rows: List[Dict[str, str]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    by_match_team: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for row in wickets_rows:
        innings_number = _safe_int(row.get("innings_number", "0"))
        if innings_number > 2:
            continue
        match_id = row["match_id"]
        batting_team = row["batting_team"]
        key = (match_id, batting_team)
        by_match_team[key]["wickets_lost_events"] += 1

    return by_match_team


def _build_match_team_stats(
    match_records: List[Dict[str, Any]],
    innings_stats: Dict[Tuple[str, str], Dict[str, float]],
    delivery_stats: Dict[Tuple[str, str], Dict[str, float]],
    wicket_stats: Dict[Tuple[str, str], Dict[str, float]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    stats: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)

    for match in match_records:
        match_id = match["match_id"]
        for team in (match["team_1"], match["team_2"]):
            key = (match_id, team)
            i = innings_stats.get(key, {})
            d = delivery_stats.get(key, {})
            w = wicket_stats.get(key, {})

            runs_scored = float(i.get("runs_scored", 0.0))
            runs_conceded = float(i.get("runs_conceded", 0.0))
            wickets_lost = float(i.get("wickets_lost", 0.0))
            balls_seen = float(d.get("balls_seen", 0.0))
            legal_balls_seen = float(d.get("legal_balls_seen", 0.0))
            dot_balls = float(d.get("dot_balls", 0.0))
            powerplay_runs = float(d.get("powerplay_runs", 0.0))
            powerplay_legal_balls = float(d.get("powerplay_legal_balls", 0.0))
            death_runs = float(d.get("death_runs", 0.0))
            death_legal_balls = float(d.get("death_legal_balls", 0.0))
            wides_conceded = float(d.get("wides_conceded", 0.0))
            noballs_conceded = float(d.get("noballs_conceded", 0.0))
            balls_bowled = float(d.get("balls_bowled", 0.0))

            stats[key] = {
                "runs_scored": runs_scored,
                "runs_conceded": runs_conceded,
                "wickets_lost": wickets_lost,
                "balls_seen": balls_seen,
                "legal_balls_seen": legal_balls_seen,
                "dot_ball_pct": (dot_balls / balls_seen) if balls_seen else 0.0,
                "powerplay_run_rate": (powerplay_runs * 6.0 / powerplay_legal_balls)
                if powerplay_legal_balls
                else 0.0,
                "death_run_rate": (death_runs * 6.0 / death_legal_balls) if death_legal_balls else 0.0,
                "wides_conceded_rate": (wides_conceded / balls_bowled) if balls_bowled else 0.0,
                "noballs_conceded_rate": (noballs_conceded / balls_bowled) if balls_bowled else 0.0,
                "wickets_lost_events": float(w.get("wickets_lost_events", 0.0)),
            }

    return stats


def _rolling_team_features(
    match_records: List[Dict[str, Any]],
    match_team_stats: Dict[Tuple[str, str], Dict[str, float]],
    first_innings_runs_by_match: Dict[str, float],
    team_form_window: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    hist: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "wins": deque(maxlen=team_form_window),
            "runs_scored": deque(maxlen=team_form_window),
            "runs_conceded": deque(maxlen=team_form_window),
            "powerplay_rr": deque(maxlen=team_form_window),
            "death_rr": deque(maxlen=team_form_window),
            "dot_ball_pct": deque(maxlen=team_form_window),
            "wides_rate": deque(maxlen=team_form_window),
            "noballs_rate": deque(maxlen=team_form_window),
            "last_match_date": None,
            "venue_wins": defaultdict(int),
            "venue_matches": defaultdict(int),
        }
    )

    h2h_wins: Dict[Tuple[str, str], int] = defaultdict(int)
    h2h_matches: Dict[Tuple[str, str], int] = defaultdict(int)

    venue_first_innings_runs: Dict[str, List[float]] = defaultdict(list)
    venue_chase_success: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})

    match_level_rows: List[Dict[str, Any]] = []
    match_team_rows: List[Dict[str, Any]] = []

    for match in match_records:
        match_id = match["match_id"]
        venue = match["venue"]
        date_obj = _parse_date(match["match_date"])
        team_1 = match["team_1"]
        team_2 = match["team_2"]

        row: Dict[str, Any] = {
            "match_id": match_id,
            "match_date": match["match_date"],
            "season": match["season"],
            "venue": venue,
            "city": match["city"],
            "team_1": team_1,
            "team_2": team_2,
            "toss_winner_is_team_1": 1 if match["toss_winner"] == team_1 else 0,
            "toss_decision_bat": 1 if match["toss_decision"] == "bat" else 0,
            "outcome_class": match["outcome_class"],
            "team_1_win": match["team_1_win"],
        }

        for prefix, team in (("team_1", team_1), ("team_2", team_2)):
            t_hist = hist[team]
            last_date = t_hist["last_match_date"]
            days_since = (date_obj - last_date).days if last_date else -1

            row[f"{prefix}_win_rate_last_{team_form_window}"] = round(_mean(t_hist["wins"]), 6)
            row[f"{prefix}_avg_runs_last_{team_form_window}"] = round(_mean(t_hist["runs_scored"]), 6)
            row[f"{prefix}_avg_runs_conceded_last_{team_form_window}"] = round(
                _mean(t_hist["runs_conceded"]), 6
            )
            row[f"{prefix}_powerplay_run_rate_last_{team_form_window}"] = round(
                _mean(t_hist["powerplay_rr"]), 6
            )
            row[f"{prefix}_death_run_rate_last_{team_form_window}"] = round(_mean(t_hist["death_rr"]), 6)
            row[f"{prefix}_dot_ball_pct_last_{team_form_window}"] = round(
                _mean(t_hist["dot_ball_pct"]), 6
            )
            row[f"{prefix}_wides_conceded_rate_last_{team_form_window}"] = round(
                _mean(t_hist["wides_rate"]), 6
            )
            row[f"{prefix}_no_ball_rate_last_{team_form_window}"] = round(
                _mean(t_hist["noballs_rate"]), 6
            )
            row[f"{prefix}_days_since_last_match"] = days_since

            v_matches = t_hist["venue_matches"][venue]
            v_wins = t_hist["venue_wins"][venue]
            row[f"{prefix}_venue_win_rate"] = round((v_wins / v_matches) if v_matches else 0.0, 6)

        h2h_key = tuple(sorted((team_1, team_2)))
        h2h_total = h2h_matches[h2h_key]
        team_1_h2h_wins = h2h_wins.get((team_1, team_2), 0)
        row["head_to_head_team_1_win_rate"] = round(
            (team_1_h2h_wins / h2h_total) if h2h_total else 0.0, 6
        )

        first_inn_vals = venue_first_innings_runs.get(venue, [])
        chase_meta = venue_chase_success.get(venue, {"success": 0, "total": 0})
        row["venue_avg_first_innings_score"] = round(_mean(first_inn_vals), 6)
        row["venue_chase_success_rate"] = round(
            (chase_meta["success"] / chase_meta["total"]) if chase_meta["total"] else 0.0,
            6,
        )

        row[f"win_rate_diff_last_{team_form_window}"] = round(
            row[f"team_1_win_rate_last_{team_form_window}"]
            - row[f"team_2_win_rate_last_{team_form_window}"],
            6,
        )
        row[f"avg_runs_diff_last_{team_form_window}"] = round(
            row[f"team_1_avg_runs_last_{team_form_window}"]
            - row[f"team_2_avg_runs_last_{team_form_window}"],
            6,
        )
        row[f"powerplay_rr_diff_last_{team_form_window}"] = round(
            row[f"team_1_powerplay_run_rate_last_{team_form_window}"]
            - row[f"team_2_powerplay_run_rate_last_{team_form_window}"],
            6,
        )
        row[f"death_rr_diff_last_{team_form_window}"] = round(
            row[f"team_1_death_run_rate_last_{team_form_window}"]
            - row[f"team_2_death_run_rate_last_{team_form_window}"],
            6,
        )

        match_level_rows.append(row)

        # Update history only after features for this match are computed.
        for team in (team_1, team_2):
            team_key = (match_id, team)
            t_stats = match_team_stats.get(team_key, {})
            t_hist = hist[team]

            won = 1.0 if match["winner"] == team else 0.0
            if match["outcome_class"] in {"win", "loss"}:
                t_hist["wins"].append(won)

            t_hist["runs_scored"].append(float(t_stats.get("runs_scored", 0.0)))
            t_hist["runs_conceded"].append(float(t_stats.get("runs_conceded", 0.0)))
            t_hist["powerplay_rr"].append(float(t_stats.get("powerplay_run_rate", 0.0)))
            t_hist["death_rr"].append(float(t_stats.get("death_run_rate", 0.0)))
            t_hist["dot_ball_pct"].append(float(t_stats.get("dot_ball_pct", 0.0)))
            t_hist["wides_rate"].append(float(t_stats.get("wides_conceded_rate", 0.0)))
            t_hist["noballs_rate"].append(float(t_stats.get("noballs_conceded_rate", 0.0)))
            t_hist["last_match_date"] = date_obj

            t_hist["venue_matches"][venue] += 1
            if match["winner"] == team:
                t_hist["venue_wins"][venue] += 1

            match_team_rows.append(
                {
                    "match_id": match_id,
                    "match_date": match["match_date"],
                    "team": team,
                    "runs_scored": round(float(t_stats.get("runs_scored", 0.0)), 6),
                    "runs_conceded": round(float(t_stats.get("runs_conceded", 0.0)), 6),
                    "powerplay_run_rate": round(float(t_stats.get("powerplay_run_rate", 0.0)), 6),
                    "death_run_rate": round(float(t_stats.get("death_run_rate", 0.0)), 6),
                    "dot_ball_pct": round(float(t_stats.get("dot_ball_pct", 0.0)), 6),
                    "wides_conceded_rate": round(float(t_stats.get("wides_conceded_rate", 0.0)), 6),
                    "noballs_conceded_rate": round(float(t_stats.get("noballs_conceded_rate", 0.0)), 6),
                    f"win_rate_last_{team_form_window}": round(_mean(t_hist["wins"]), 6),
                }
            )

        h2h_matches[h2h_key] += 1
        if match["winner"] == team_1:
            h2h_wins[(team_1, team_2)] += 1
        elif match["winner"] == team_2:
            h2h_wins[(team_2, team_1)] += 1

        # Venue aggregates update after row creation.
        first_innings_score = float(first_innings_runs_by_match.get(match_id, 0.0))
        if first_innings_score > 0:
            venue_first_innings_runs[venue].append(first_innings_score)

        if match["outcome_class"] in {"win", "loss"}:
            venue_chase_success[venue]["total"] += 1
            if match.get("result_type", "") == "wickets":
                venue_chase_success[venue]["success"] += 1

    return match_level_rows, match_team_rows


def _fieldnames_match_level(team_form_window: int) -> List[str]:
    return [
        "match_id",
        "match_date",
        "season",
        "venue",
        "city",
        "team_1",
        "team_2",
        "toss_winner_is_team_1",
        "toss_decision_bat",
        "team_1_win_rate_last_{}".format(team_form_window),
        "team_2_win_rate_last_{}".format(team_form_window),
        "team_1_avg_runs_last_{}".format(team_form_window),
        "team_2_avg_runs_last_{}".format(team_form_window),
        "team_1_avg_runs_conceded_last_{}".format(team_form_window),
        "team_2_avg_runs_conceded_last_{}".format(team_form_window),
        "team_1_powerplay_run_rate_last_{}".format(team_form_window),
        "team_2_powerplay_run_rate_last_{}".format(team_form_window),
        "team_1_death_run_rate_last_{}".format(team_form_window),
        "team_2_death_run_rate_last_{}".format(team_form_window),
        "team_1_dot_ball_pct_last_{}".format(team_form_window),
        "team_2_dot_ball_pct_last_{}".format(team_form_window),
        "team_1_wides_conceded_rate_last_{}".format(team_form_window),
        "team_2_wides_conceded_rate_last_{}".format(team_form_window),
        "team_1_no_ball_rate_last_{}".format(team_form_window),
        "team_2_no_ball_rate_last_{}".format(team_form_window),
        "team_1_days_since_last_match",
        "team_2_days_since_last_match",
        "team_1_venue_win_rate",
        "team_2_venue_win_rate",
        "head_to_head_team_1_win_rate",
        "venue_avg_first_innings_score",
        "venue_chase_success_rate",
        "win_rate_diff_last_{}".format(team_form_window),
        "avg_runs_diff_last_{}".format(team_form_window),
        "powerplay_rr_diff_last_{}".format(team_form_window),
        "death_rr_diff_last_{}".format(team_form_window),
        "outcome_class",
        "team_1_win",
    ]


def run_feature_engineering(config: PipelineConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    matches_rows = _read_csv(config.input_dir / "matches.csv")
    match_teams_rows = _read_csv(config.input_dir / "match_teams.csv")
    innings_rows = _read_csv(config.input_dir / "innings.csv")
    deliveries_rows = _read_csv(config.input_dir / "deliveries.csv")
    wickets_rows = _read_csv(config.input_dir / "wickets.csv")

    teams_by_match = _build_match_team_lookup(match_teams_rows)
    match_records = _initialize_match_records(matches_rows, teams_by_match, config.match_filters)

    innings_stats, first_innings_runs_by_match = _aggregate_innings(innings_rows)
    deliveries_stats = _aggregate_deliveries(deliveries_rows, teams_by_match)
    wickets_stats = _aggregate_wickets(wickets_rows)

    match_team_stats = _build_match_team_stats(match_records, innings_stats, deliveries_stats, wickets_stats)

    match_level_rows, match_team_rows = _rolling_team_features(
        match_records, match_team_stats, first_innings_runs_by_match, config.team_form_window
    )

    match_level_fields = _fieldnames_match_level(config.team_form_window)
    match_team_fields = [
        "match_id",
        "match_date",
        "team",
        "runs_scored",
        "runs_conceded",
        "powerplay_run_rate",
        "death_run_rate",
        "dot_ball_pct",
        "wides_conceded_rate",
        "noballs_conceded_rate",
        "win_rate_last_{}".format(config.team_form_window),
    ]

    _write_csv(config.output_dir / "match_level_features.csv", match_level_rows, match_level_fields)
    _write_csv(config.output_dir / "match_team_features.csv", match_team_rows, match_team_fields)

    summary = {
        "matches_input": len(matches_rows),
        "matches_modeled": len(match_records),
        "match_level_feature_rows": len(match_level_rows),
        "match_team_feature_rows": len(match_team_rows),
        "output_dir": str(config.output_dir),
    }
    (config.output_dir / "feature_engineering_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
