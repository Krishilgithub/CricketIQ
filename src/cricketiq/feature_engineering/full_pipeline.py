from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Tuple


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    team_form_windows: List[int]
    match_filters: Dict[str, str]
    generate_live_state_features: bool


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _weighted_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    weights = list(range(1, len(vals) + 1))
    return sum(v * w for v, w in zip(vals, weights)) / float(sum(weights))


def _window_slice(values: Deque[float], window: int) -> List[float]:
    vals = list(values)
    if len(vals) <= window:
        return vals
    return vals[-window:]


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _parse_season_year(season: str) -> int:
    if not season:
        return 0
    prefix = season[:4]
    return _safe_int(prefix, 0)


def _phase_from_over(over: int) -> str:
    if over <= 5:
        return "powerplay"
    if over <= 14:
        return "middle"
    return "death"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_config(config_path: Path) -> PipelineConfig:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    windows = data.get("team_form_windows", [])
    if not windows:
        windows = [int(data.get("team_form_window", 5))]
    windows = sorted({int(w) for w in windows if int(w) > 0})
    if not windows:
        windows = [5]

    return PipelineConfig(
        input_dir=Path(data["input_dir"]),
        output_dir=Path(data["output_dir"]),
        team_form_windows=windows,
        match_filters=data.get("match_filters", {}),
        generate_live_state_features=bool(data.get("generate_live_state_features", True)),
    )


def _build_match_team_lookup(match_teams_rows: List[Dict[str, str]]) -> Dict[str, List[str]]:
    teams_by_match: Dict[str, List[str]] = defaultdict(list)
    for row in match_teams_rows:
        match_id = row["match_id"]
        team = row.get("team", "").strip()
        if team and team not in teams_by_match[match_id]:
            teams_by_match[match_id].append(team)

    for match_id, teams in teams_by_match.items():
        teams.sort()
        teams_by_match[match_id] = teams[:2]
    return teams_by_match


def _derive_outcome_class(row: Dict[str, str], team_1: str, team_2: str) -> Tuple[str, str]:
    winner = row.get("winner", "").strip()
    result_text = row.get("result_text", "").strip().lower()

    if winner == team_1:
        return "win", "1"
    if winner == team_2:
        return "loss", "0"
    if "tie" in result_text:
        return "tie", ""
    if "no result" in result_text or "abandon" in result_text or winner == "":
        return "no_result", ""
    return "no_result", ""


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
        outcome_class, target = _derive_outcome_class(row, team_1, team_2)

        city = row.get("city", "").strip()
        event_name = row.get("event_name", "").strip()
        season = row.get("season", "")

        records.append(
            {
                "match_id": match_id,
                "match_date": row["match_date"],
                "season": season,
                "season_year": _parse_season_year(season),
                "event_name": event_name,
                "event_name_missing_flag": 1 if event_name == "" else 0,
                "venue": row.get("venue", "").strip(),
                "city": city if city else "UNKNOWN_CITY",
                "city_missing_flag": 1 if city == "" else 0,
                "team_1": team_1,
                "team_2": team_2,
                "toss_winner": row.get("toss_winner", "").strip(),
                "toss_decision": row.get("toss_decision", "").strip(),
                "winner": row.get("winner", "").strip(),
                "result_type": row.get("result_type", "").strip(),
                "result_margin": row.get("result_margin", "").strip(),
                "outcome_class": outcome_class,
                "team_1_win": target,
                "label_is_binary": 1 if target in {"0", "1"} else 0,
            }
        )

    records.sort(key=lambda r: (r["match_date"], _safe_int(r["match_id"])))
    return records


def _aggregate_innings(
    innings_rows: List[Dict[str, str]],
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], Dict[str, float], Dict[Tuple[str, int, str], Dict[str, float]]]:
    by_match_team: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    first_innings_runs_by_match: Dict[str, float] = {}
    by_match_innings_team: Dict[Tuple[str, int, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    teams_per_match: Dict[str, Dict[str, float]] = defaultdict(dict)

    for row in innings_rows:
        innings_number = _safe_int(row.get("innings_number", "0"))
        if innings_number > 2:
            continue

        match_id = row["match_id"]
        team = row.get("team", "")
        runs = float(_safe_int(row.get("total_runs", "0")))
        wickets = float(_safe_int(row.get("total_wickets", "0")))
        balls = float(_safe_int(row.get("total_balls", "0")))

        key = (match_id, team)
        by_match_team[key]["runs_scored"] += runs
        by_match_team[key]["wickets_lost"] += wickets
        by_match_team[key]["balls_faced"] += balls

        ikey = (match_id, innings_number, team)
        by_match_innings_team[ikey]["innings_runs"] += runs
        by_match_innings_team[ikey]["innings_wickets"] += wickets

        if innings_number == 1:
            first_innings_runs_by_match[match_id] = runs

        teams_per_match[match_id][team] = runs

    for match_id, team_runs in teams_per_match.items():
        teams = list(team_runs.keys())
        if len(teams) < 2:
            continue
        for team in teams:
            opp_runs = sum(team_runs[t] for t in teams if t != team)
            by_match_team[(match_id, team)]["runs_conceded"] = opp_runs

    return by_match_team, first_innings_runs_by_match, by_match_innings_team


def _aggregate_deliveries(
    deliveries_rows: List[Dict[str, str]],
    teams_by_match: Dict[str, List[str]],
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, float]],
    Dict[Tuple[str, int, str], Dict[str, float]],
    int,
]:
    by_match_team: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_match_innings_team: Dict[Tuple[str, int, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    delivery_key_seen: set[Tuple[str, str, str, str, str, str]] = set()
    duplicate_delivery_keys = 0

    for row in deliveries_rows:
        innings_number = _safe_int(row.get("innings_number", "0"))
        if innings_number > 2:
            continue

        match_id = row["match_id"]
        batting_team = row.get("batting_team", "")
        teams = teams_by_match.get(match_id, [])
        if len(teams) != 2 or batting_team not in teams:
            continue

        bowling_team = teams[1] if batting_team == teams[0] else teams[0]

        over = _safe_int(row.get("over", "0"))
        runs_total = float(_safe_int(row.get("runs_total", "0")))
        runs_batter = _safe_int(row.get("runs_batter", "0"))
        runs_extras = float(_safe_int(row.get("runs_extras", "0")))
        wides = float(_safe_int(row.get("extras_wides", "0")))
        noballs = float(_safe_int(row.get("extras_noballs", "0")))
        legal_ball = 1.0 if (wides == 0 and noballs == 0) else 0.0

        dkey = (
            match_id,
            row.get("innings_number", ""),
            row.get("over", ""),
            row.get("ball_in_over", ""),
            row.get("batter", ""),
            row.get("bowler", ""),
        )
        if dkey in delivery_key_seen:
            duplicate_delivery_keys += 1
        else:
            delivery_key_seen.add(dkey)

        bat_key = (match_id, batting_team)
        bowl_key = (match_id, bowling_team)

        by_match_team[bat_key]["balls_seen"] += 1.0
        by_match_team[bat_key]["legal_balls_seen"] += legal_ball
        by_match_team[bat_key]["dot_balls"] += 1.0 if runs_total == 0 else 0.0
        by_match_team[bat_key]["boundary_balls"] += 1.0 if runs_batter in {4, 6} else 0.0

        phase = _phase_from_over(over)
        if phase == "powerplay":
            by_match_team[bat_key]["powerplay_runs"] += runs_total
            by_match_team[bat_key]["powerplay_legal_balls"] += legal_ball
        elif phase == "middle":
            by_match_team[bat_key]["middle_runs"] += runs_total
            by_match_team[bat_key]["middle_legal_balls"] += legal_ball
        else:
            by_match_team[bat_key]["death_runs"] += runs_total
            by_match_team[bat_key]["death_legal_balls"] += legal_ball

        by_match_team[bowl_key]["wides_conceded"] += wides
        by_match_team[bowl_key]["noballs_conceded"] += noballs
        by_match_team[bowl_key]["extras_conceded"] += runs_extras
        by_match_team[bowl_key]["balls_bowled"] += 1.0

        ikey = (match_id, innings_number, batting_team)
        by_match_innings_team[ikey]["innings_runs_from_deliveries"] += runs_total

    return by_match_team, by_match_innings_team, duplicate_delivery_keys


def _aggregate_wickets(
    wickets_rows: List[Dict[str, str]],
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, float]],
    Dict[Tuple[str, int, str], Dict[str, float]],
    Dict[Tuple[str, int, int, int], int],
    int,
    int,
]:
    by_match_team: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_match_innings_team: Dict[Tuple[str, int, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    wickets_by_delivery: Dict[Tuple[str, int, int, int], int] = defaultdict(int)

    wicket_key_seen: set[Tuple[str, str, str, str, str, str]] = set()
    duplicate_wicket_keys = 0

    suspicious_missing_fielders = 0
    fielder_expected_kinds = {"caught", "run out", "stumped", "caught and bowled"}

    for row in wickets_rows:
        innings_number = _safe_int(row.get("innings_number", "0"))
        if innings_number > 2:
            continue

        match_id = row["match_id"]
        batting_team = row.get("batting_team", "")
        kind = row.get("kind", "")
        over = _safe_int(row.get("over", "0"))
        ball_in_over = _safe_int(row.get("ball_in_over", "0"))

        wkey = (
            match_id,
            row.get("innings_number", ""),
            row.get("over", ""),
            row.get("ball_in_over", ""),
            row.get("player_out", ""),
            kind,
        )
        if wkey in wicket_key_seen:
            duplicate_wicket_keys += 1
        else:
            wicket_key_seen.add(wkey)

        team_key = (match_id, batting_team)
        by_match_team[team_key]["wickets_lost_events"] += 1.0
        if over >= 15:
            by_match_team[team_key]["death_wickets_lost_events"] += 1.0

        if kind == "caught":
            by_match_team[team_key]["dismissal_caught"] += 1.0
        elif kind == "bowled":
            by_match_team[team_key]["dismissal_bowled"] += 1.0
        elif kind == "lbw":
            by_match_team[team_key]["dismissal_lbw"] += 1.0
        elif kind == "run out":
            by_match_team[team_key]["dismissal_run_out"] += 1.0
        elif kind == "stumped":
            by_match_team[team_key]["dismissal_stumped"] += 1.0
        else:
            by_match_team[team_key]["dismissal_other"] += 1.0

        ikey = (match_id, innings_number, batting_team)
        by_match_innings_team[ikey]["innings_wickets_from_events"] += 1.0
        wickets_by_delivery[(match_id, innings_number, over, ball_in_over)] += 1

        fielders = row.get("fielders", "").strip()
        if kind in fielder_expected_kinds and fielders == "":
            suspicious_missing_fielders += 1

    return (
        by_match_team,
        by_match_innings_team,
        wickets_by_delivery,
        duplicate_wicket_keys,
        suspicious_missing_fielders,
    )


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

            balls_seen = float(d.get("balls_seen", 0.0))
            legal_balls_seen = float(d.get("legal_balls_seen", 0.0))
            dot_balls = float(d.get("dot_balls", 0.0))
            boundary_balls = float(d.get("boundary_balls", 0.0))

            powerplay_runs = float(d.get("powerplay_runs", 0.0))
            powerplay_legal_balls = float(d.get("powerplay_legal_balls", 0.0))
            middle_runs = float(d.get("middle_runs", 0.0))
            middle_legal_balls = float(d.get("middle_legal_balls", 0.0))
            death_runs = float(d.get("death_runs", 0.0))
            death_legal_balls = float(d.get("death_legal_balls", 0.0))

            wides_conceded = float(d.get("wides_conceded", 0.0))
            noballs_conceded = float(d.get("noballs_conceded", 0.0))
            balls_bowled = float(d.get("balls_bowled", 0.0))

            wickets_lost_events = float(w.get("wickets_lost_events", 0.0))
            death_wickets_lost_events = float(w.get("death_wickets_lost_events", 0.0))

            dismissal_total = max(wickets_lost_events, 1.0)

            stats[key] = {
                "runs_scored": runs_scored,
                "runs_conceded": runs_conceded,
                "run_rate": (runs_scored * 6.0 / legal_balls_seen) if legal_balls_seen else 0.0,
                "dot_ball_pct": (dot_balls / balls_seen) if balls_seen else 0.0,
                "boundary_pct": (boundary_balls / balls_seen) if balls_seen else 0.0,
                "powerplay_run_rate": (powerplay_runs * 6.0 / powerplay_legal_balls)
                if powerplay_legal_balls
                else 0.0,
                "middle_run_rate": (middle_runs * 6.0 / middle_legal_balls) if middle_legal_balls else 0.0,
                "death_run_rate": (death_runs * 6.0 / death_legal_balls) if death_legal_balls else 0.0,
                "wides_conceded_rate": (wides_conceded / balls_bowled) if balls_bowled else 0.0,
                "noballs_conceded_rate": (noballs_conceded / balls_bowled) if balls_bowled else 0.0,
                "wickets_lost_events": wickets_lost_events,
                "death_wickets_lost_rate": (death_wickets_lost_events / death_legal_balls)
                if death_legal_balls
                else 0.0,
                "dismissal_caught_pct": float(w.get("dismissal_caught", 0.0)) / dismissal_total,
                "dismissal_bowled_pct": float(w.get("dismissal_bowled", 0.0)) / dismissal_total,
                "dismissal_lbw_pct": float(w.get("dismissal_lbw", 0.0)) / dismissal_total,
                "dismissal_run_out_pct": float(w.get("dismissal_run_out", 0.0)) / dismissal_total,
                "dismissal_stumped_pct": float(w.get("dismissal_stumped", 0.0)) / dismissal_total,
            }

    return stats


def _append_team_history(
    hist: Dict[str, Dict[str, Any]],
    team: str,
    match: Dict[str, Any],
    team_stats: Dict[str, float],
    venue: str,
    date_obj: datetime,
    max_window: int,
) -> None:
    t_hist = hist[team]

    won = 1.0 if match["winner"] == team else 0.0
    if match["outcome_class"] in {"win", "loss"}:
        t_hist["wins"].append(won)

    t_hist["runs_scored"].append(float(team_stats.get("runs_scored", 0.0)))
    t_hist["runs_conceded"].append(float(team_stats.get("runs_conceded", 0.0)))
    t_hist["run_rate"].append(float(team_stats.get("run_rate", 0.0)))
    t_hist["powerplay_rr"].append(float(team_stats.get("powerplay_run_rate", 0.0)))
    t_hist["middle_rr"].append(float(team_stats.get("middle_run_rate", 0.0)))
    t_hist["death_rr"].append(float(team_stats.get("death_run_rate", 0.0)))
    t_hist["dot_ball_pct"].append(float(team_stats.get("dot_ball_pct", 0.0)))
    t_hist["boundary_pct"].append(float(team_stats.get("boundary_pct", 0.0)))
    t_hist["wides_rate"].append(float(team_stats.get("wides_conceded_rate", 0.0)))
    t_hist["noballs_rate"].append(float(team_stats.get("noballs_conceded_rate", 0.0)))
    t_hist["death_wickets_lost_rate"].append(float(team_stats.get("death_wickets_lost_rate", 0.0)))

    t_hist["last_match_date"] = date_obj
    t_hist["venue_matches"][venue] += 1
    if match["winner"] == team:
        t_hist["venue_wins"][venue] += 1


def _rolling_team_features(
    match_records: List[Dict[str, Any]],
    match_team_stats: Dict[Tuple[str, str], Dict[str, float]],
    first_innings_runs_by_match: Dict[str, float],
    team_form_windows: List[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    max_window = max(team_form_windows)

    hist: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "wins": deque(maxlen=max_window),
            "runs_scored": deque(maxlen=max_window),
            "runs_conceded": deque(maxlen=max_window),
            "run_rate": deque(maxlen=max_window),
            "powerplay_rr": deque(maxlen=max_window),
            "middle_rr": deque(maxlen=max_window),
            "death_rr": deque(maxlen=max_window),
            "dot_ball_pct": deque(maxlen=max_window),
            "boundary_pct": deque(maxlen=max_window),
            "wides_rate": deque(maxlen=max_window),
            "noballs_rate": deque(maxlen=max_window),
            "death_wickets_lost_rate": deque(maxlen=max_window),
            "last_match_date": None,
            "venue_wins": defaultdict(int),
            "venue_matches": defaultdict(int),
        }
    )

    h2h_wins: Dict[Tuple[str, str], int] = defaultdict(int)
    h2h_matches: Dict[Tuple[str, str], int] = defaultdict(int)

    venue_first_innings_runs: Dict[str, List[float]] = defaultdict(list)
    venue_chase_success: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})

    venue_season_first_runs: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    venue_season_chase: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})

    season_first_runs: Dict[int, List[float]] = defaultdict(list)
    season_chase: Dict[int, Dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})

    match_level_rows: List[Dict[str, Any]] = []
    match_team_rows: List[Dict[str, Any]] = []

    for match in match_records:
        match_id = match["match_id"]
        match_date = match["match_date"]
        date_obj = _parse_date(match_date)
        venue = match["venue"]
        season_year = int(match["season_year"])
        team_1 = match["team_1"]
        team_2 = match["team_2"]

        row: Dict[str, Any] = {
            "match_id": match_id,
            "match_date": match_date,
            "season": match["season"],
            "season_year": season_year,
            "event_name": match["event_name"],
            "event_name_missing_flag": match["event_name_missing_flag"],
            "venue": venue,
            "city": match["city"],
            "city_missing_flag": match["city_missing_flag"],
            "team_1": team_1,
            "team_2": team_2,
            "toss_winner_is_team_1": 1 if match["toss_winner"] == team_1 else 0,
            "toss_decision_bat": 1 if match["toss_decision"] == "bat" else 0,
            "outcome_class": match["outcome_class"],
            "team_1_win": match["team_1_win"],
            "label_is_binary": match["label_is_binary"],
        }

        for prefix, team in (("team_1", team_1), ("team_2", team_2)):
            t_hist = hist[team]
            last_date = t_hist["last_match_date"]
            row[f"{prefix}_days_since_last_match"] = (date_obj - last_date).days if last_date else -1

            v_matches = t_hist["venue_matches"][venue]
            v_wins = t_hist["venue_wins"][venue]
            row[f"{prefix}_venue_win_rate"] = round((v_wins / v_matches) if v_matches else 0.0, 6)

            for window in team_form_windows:
                wins_window = _window_slice(t_hist["wins"], window)
                p = _mean(wins_window)
                n = len(wins_window)
                uncertainty = math.sqrt(max(p * (1.0 - p), 0.0) / n) if n > 0 else 1.0

                row[f"{prefix}_sample_size_last_{window}"] = n
                row[f"{prefix}_win_rate_last_{window}"] = round(p, 6)
                row[f"{prefix}_win_rate_weighted_last_{window}"] = round(
                    _weighted_mean(wins_window), 6
                )
                row[f"{prefix}_win_rate_uncertainty_last_{window}"] = round(uncertainty, 6)

                row[f"{prefix}_avg_runs_last_{window}"] = round(
                    _mean(_window_slice(t_hist["runs_scored"], window)), 6
                )
                row[f"{prefix}_avg_runs_conceded_last_{window}"] = round(
                    _mean(_window_slice(t_hist["runs_conceded"], window)), 6
                )
                row[f"{prefix}_run_rate_last_{window}"] = round(
                    _mean(_window_slice(t_hist["run_rate"], window)), 6
                )
                row[f"{prefix}_powerplay_run_rate_last_{window}"] = round(
                    _mean(_window_slice(t_hist["powerplay_rr"], window)), 6
                )
                row[f"{prefix}_middle_run_rate_last_{window}"] = round(
                    _mean(_window_slice(t_hist["middle_rr"], window)), 6
                )
                row[f"{prefix}_death_run_rate_last_{window}"] = round(
                    _mean(_window_slice(t_hist["death_rr"], window)), 6
                )
                row[f"{prefix}_dot_ball_pct_last_{window}"] = round(
                    _mean(_window_slice(t_hist["dot_ball_pct"], window)), 6
                )
                row[f"{prefix}_boundary_pct_last_{window}"] = round(
                    _mean(_window_slice(t_hist["boundary_pct"], window)), 6
                )
                row[f"{prefix}_wides_conceded_rate_last_{window}"] = round(
                    _mean(_window_slice(t_hist["wides_rate"], window)), 6
                )
                row[f"{prefix}_no_ball_rate_last_{window}"] = round(
                    _mean(_window_slice(t_hist["noballs_rate"], window)), 6
                )
                row[f"{prefix}_death_wickets_lost_rate_last_{window}"] = round(
                    _mean(_window_slice(t_hist["death_wickets_lost_rate"], window)), 6
                )

                row[f"{prefix}_clutch_index_last_{window}"] = round(
                    row[f"{prefix}_death_run_rate_last_{window}"]
                    - row[f"{prefix}_middle_run_rate_last_{window}"],
                    6,
                )

        h2h_key = tuple(sorted((team_1, team_2)))
        h2h_total = h2h_matches[h2h_key]
        row["head_to_head_total_matches"] = h2h_total
        row["head_to_head_team_1_win_rate"] = round(
            (h2h_wins.get((team_1, team_2), 0) / h2h_total) if h2h_total else 0.0,
            6,
        )

        venue_vals = venue_first_innings_runs.get(venue, [])
        venue_chase_meta = venue_chase_success.get(venue, {"success": 0, "total": 0})
        row["venue_avg_first_innings_score"] = round(_mean(venue_vals), 6)
        row["venue_chase_success_rate"] = round(
            (venue_chase_meta["success"] / venue_chase_meta["total"]) if venue_chase_meta["total"] else 0.0,
            6,
        )

        vsy_key = (venue, season_year)
        vsy_vals = venue_season_first_runs.get(vsy_key, [])
        vsy_chase = venue_season_chase.get(vsy_key, {"success": 0, "total": 0})
        row["venue_season_avg_first_innings_score"] = round(_mean(vsy_vals), 6)
        row["venue_season_chase_success_rate"] = round(
            (vsy_chase["success"] / vsy_chase["total"]) if vsy_chase["total"] else 0.0,
            6,
        )

        season_vals = season_first_runs.get(season_year, [])
        season_chase_meta = season_chase.get(season_year, {"success": 0, "total": 0})
        row["season_avg_first_innings_score"] = round(_mean(season_vals), 6)
        row["season_chase_success_rate"] = round(
            (season_chase_meta["success"] / season_chase_meta["total"]) if season_chase_meta["total"] else 0.0,
            6,
        )

        row["venue_first_innings_vs_season"] = round(
            row["venue_avg_first_innings_score"] - row["season_avg_first_innings_score"],
            6,
        )
        row["venue_chase_vs_season"] = round(
            row["venue_chase_success_rate"] - row["season_chase_success_rate"],
            6,
        )

        primary_window = team_form_windows[0]
        row[f"win_rate_diff_last_{primary_window}"] = round(
            row[f"team_1_win_rate_last_{primary_window}"] - row[f"team_2_win_rate_last_{primary_window}"],
            6,
        )
        row[f"weighted_win_rate_diff_last_{primary_window}"] = round(
            row[f"team_1_win_rate_weighted_last_{primary_window}"]
            - row[f"team_2_win_rate_weighted_last_{primary_window}"],
            6,
        )
        row[f"avg_runs_diff_last_{primary_window}"] = round(
            row[f"team_1_avg_runs_last_{primary_window}"] - row[f"team_2_avg_runs_last_{primary_window}"],
            6,
        )
        row[f"powerplay_rr_diff_last_{primary_window}"] = round(
            row[f"team_1_powerplay_run_rate_last_{primary_window}"]
            - row[f"team_2_powerplay_run_rate_last_{primary_window}"],
            6,
        )
        row[f"middle_rr_diff_last_{primary_window}"] = round(
            row[f"team_1_middle_run_rate_last_{primary_window}"]
            - row[f"team_2_middle_run_rate_last_{primary_window}"],
            6,
        )
        row[f"death_rr_diff_last_{primary_window}"] = round(
            row[f"team_1_death_run_rate_last_{primary_window}"]
            - row[f"team_2_death_run_rate_last_{primary_window}"],
            6,
        )
        row[f"boundary_pct_diff_last_{primary_window}"] = round(
            row[f"team_1_boundary_pct_last_{primary_window}"]
            - row[f"team_2_boundary_pct_last_{primary_window}"],
            6,
        )
        row[f"dot_ball_pct_diff_last_{primary_window}"] = round(
            row[f"team_1_dot_ball_pct_last_{primary_window}"]
            - row[f"team_2_dot_ball_pct_last_{primary_window}"],
            6,
        )

        match_level_rows.append(row)

        # Update history after feature creation.
        for team in (team_1, team_2):
            t_stats = match_team_stats.get((match_id, team), {})
            _append_team_history(hist, team, match, t_stats, venue, date_obj, max_window)
            match_team_rows.append(
                {
                    "match_id": match_id,
                    "match_date": match_date,
                    "team": team,
                    "runs_scored": round(float(t_stats.get("runs_scored", 0.0)), 6),
                    "runs_conceded": round(float(t_stats.get("runs_conceded", 0.0)), 6),
                    "run_rate": round(float(t_stats.get("run_rate", 0.0)), 6),
                    "powerplay_run_rate": round(float(t_stats.get("powerplay_run_rate", 0.0)), 6),
                    "middle_run_rate": round(float(t_stats.get("middle_run_rate", 0.0)), 6),
                    "death_run_rate": round(float(t_stats.get("death_run_rate", 0.0)), 6),
                    "dot_ball_pct": round(float(t_stats.get("dot_ball_pct", 0.0)), 6),
                    "boundary_pct": round(float(t_stats.get("boundary_pct", 0.0)), 6),
                    "wides_conceded_rate": round(float(t_stats.get("wides_conceded_rate", 0.0)), 6),
                    "noballs_conceded_rate": round(float(t_stats.get("noballs_conceded_rate", 0.0)), 6),
                }
            )

        h2h_matches[h2h_key] += 1
        if match["winner"] == team_1:
            h2h_wins[(team_1, team_2)] += 1
        elif match["winner"] == team_2:
            h2h_wins[(team_2, team_1)] += 1

        first_innings_score = float(first_innings_runs_by_match.get(match_id, 0.0))
        if first_innings_score > 0:
            venue_first_innings_runs[venue].append(first_innings_score)
            venue_season_first_runs[(venue, season_year)].append(first_innings_score)
            season_first_runs[season_year].append(first_innings_score)

        if match["outcome_class"] in {"win", "loss"}:
            chase_success = 1 if match.get("result_type", "") == "wickets" else 0
            venue_chase_success[venue]["total"] += 1
            venue_chase_success[venue]["success"] += chase_success

            venue_season_chase[(venue, season_year)]["total"] += 1
            venue_season_chase[(venue, season_year)]["success"] += chase_success

            season_chase[season_year]["total"] += 1
            season_chase[season_year]["success"] += chase_success

    return match_level_rows, match_team_rows


def _build_quality_report(
    match_records: List[Dict[str, Any]],
    innings_totals: Dict[Tuple[str, int, str], Dict[str, float]],
    delivery_totals: Dict[Tuple[str, int, str], Dict[str, float]],
    wickets_totals: Dict[Tuple[str, int, str], Dict[str, float]],
    duplicate_delivery_keys: int,
    duplicate_wicket_keys: int,
    suspicious_missing_fielders: int,
) -> Dict[str, Any]:
    outcome_counts = Counter(r["outcome_class"] for r in match_records)

    city_missing = sum(1 for r in match_records if r.get("city_missing_flag", 0) == 1)
    event_name_missing = sum(1 for r in match_records if r.get("event_name_missing_flag", 0) == 1)

    innings_delivery_mismatches = 0
    innings_wicket_mismatches = 0

    for key, i_vals in innings_totals.items():
        i_runs = float(i_vals.get("innings_runs", 0.0))
        d_runs = float(delivery_totals.get(key, {}).get("innings_runs_from_deliveries", 0.0))
        if abs(i_runs - d_runs) > 0.001:
            innings_delivery_mismatches += 1

        i_w = float(i_vals.get("innings_wickets", 0.0))
        w_ev = float(wickets_totals.get(key, {}).get("innings_wickets_from_events", 0.0))
        if i_w < w_ev or abs(i_w - w_ev) > 1.001:
            innings_wicket_mismatches += 1

    return {
        "matches_modeled": len(match_records),
        "outcome_counts": dict(outcome_counts),
        "missing_city_rows": city_missing,
        "missing_event_name_rows": event_name_missing,
        "duplicate_delivery_keys": duplicate_delivery_keys,
        "duplicate_wicket_keys": duplicate_wicket_keys,
        "innings_delivery_mismatch_count": innings_delivery_mismatches,
        "innings_wicket_mismatch_count": innings_wicket_mismatches,
        "suspicious_missing_fielders": suspicious_missing_fielders,
    }


def _build_live_state_features(
    deliveries_rows: List[Dict[str, str]],
    match_records: List[Dict[str, Any]],
    teams_by_match: Dict[str, List[str]],
    first_innings_runs_by_match: Dict[str, float],
    wickets_by_delivery: Dict[Tuple[str, int, int, int], int],
) -> List[Dict[str, Any]]:
    modeled_match_ids = {m["match_id"] for m in match_records}
    match_date_by_id = {m["match_id"]: m["match_date"] for m in match_records}

    grouped: Dict[Tuple[str, int, str], List[Dict[str, str]]] = defaultdict(list)
    for row in deliveries_rows:
        match_id = row["match_id"]
        innings_number = _safe_int(row.get("innings_number", "0"))
        if match_id not in modeled_match_ids or innings_number > 2:
            continue
        batting_team = row.get("batting_team", "")
        grouped[(match_id, innings_number, batting_team)].append(row)

    out_rows: List[Dict[str, Any]] = []

    for key, rows in grouped.items():
        match_id, innings_number, batting_team = key
        teams = teams_by_match.get(match_id, [])
        if len(teams) != 2 or batting_team not in teams:
            continue

        bowling_team = teams[1] if batting_team == teams[0] else teams[0]

        rows.sort(key=lambda r: (_safe_int(r.get("over", "0")), _safe_int(r.get("ball_in_over", "0"))))

        score = 0
        wickets = 0
        legal_balls = 0

        target = int(first_innings_runs_by_match.get(match_id, 0.0)) + 1 if innings_number == 2 else 0

        for row in rows:
            over = _safe_int(row.get("over", "0"))
            ball_in_over = _safe_int(row.get("ball_in_over", "0"))
            runs_total = _safe_int(row.get("runs_total", "0"))
            wides = _safe_int(row.get("extras_wides", "0"))
            noballs = _safe_int(row.get("extras_noballs", "0"))

            score += runs_total
            if wides == 0 and noballs == 0:
                legal_balls += 1

            wickets += wickets_by_delivery.get((match_id, innings_number, over, ball_in_over), 0)

            phase = _phase_from_over(over)
            balls_remaining = max(120 - legal_balls, 0)

            runs_required = ""
            required_rr = ""
            current_rr = round((score * 6.0 / legal_balls), 6) if legal_balls else 0.0
            if innings_number == 2 and target > 0:
                runs_required_int = max(target - score, 0)
                runs_required = runs_required_int
                required_rr = round((runs_required_int * 6.0 / balls_remaining), 6) if balls_remaining else 0.0

            out_rows.append(
                {
                    "match_id": match_id,
                    "match_date": match_date_by_id.get(match_id, ""),
                    "innings_number": innings_number,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over": over,
                    "ball_in_over": ball_in_over,
                    "phase": phase,
                    "score_after_ball": score,
                    "wickets_after_ball": wickets,
                    "legal_balls_bowled": legal_balls,
                    "balls_remaining": balls_remaining,
                    "current_run_rate": current_rr,
                    "target_runs": target if target else "",
                    "runs_required": runs_required,
                    "required_run_rate": required_rr,
                }
            )

    out_rows.sort(
        key=lambda r: (
            r["match_date"],
            _safe_int(str(r["match_id"])),
            _safe_int(str(r["innings_number"])),
            _safe_int(str(r["over"])),
            _safe_int(str(r["ball_in_over"])),
        )
    )
    return out_rows


def run_feature_engineering(config: PipelineConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    matches_rows = _read_csv(config.input_dir / "matches.csv")
    match_teams_rows = _read_csv(config.input_dir / "match_teams.csv")
    innings_rows = _read_csv(config.input_dir / "innings.csv")
    deliveries_rows = _read_csv(config.input_dir / "deliveries.csv")
    wickets_rows = _read_csv(config.input_dir / "wickets.csv")

    teams_by_match = _build_match_team_lookup(match_teams_rows)
    match_records = _initialize_match_records(matches_rows, teams_by_match, config.match_filters)

    innings_stats, first_innings_runs_by_match, innings_totals = _aggregate_innings(innings_rows)
    deliveries_stats, delivery_totals, duplicate_delivery_keys = _aggregate_deliveries(deliveries_rows, teams_by_match)
    (
        wickets_stats,
        wickets_totals,
        wickets_by_delivery,
        duplicate_wicket_keys,
        suspicious_missing_fielders,
    ) = _aggregate_wickets(wickets_rows)

    match_team_stats = _build_match_team_stats(match_records, innings_stats, deliveries_stats, wickets_stats)

    match_level_rows, match_team_rows = _rolling_team_features(
        match_records,
        match_team_stats,
        first_innings_runs_by_match,
        config.team_form_windows,
    )

    quality_report = _build_quality_report(
        match_records,
        innings_totals,
        delivery_totals,
        wickets_totals,
        duplicate_delivery_keys,
        duplicate_wicket_keys,
        suspicious_missing_fielders,
    )

    _write_csv(config.output_dir / "match_level_features.csv", match_level_rows)
    _write_csv(config.output_dir / "match_team_features.csv", match_team_rows)

    live_rows: List[Dict[str, Any]] = []
    if config.generate_live_state_features:
        live_rows = _build_live_state_features(
            deliveries_rows,
            match_records,
            teams_by_match,
            first_innings_runs_by_match,
            wickets_by_delivery,
        )
        _write_csv(config.output_dir / "live_state_features.csv", live_rows)

    (config.output_dir / "feature_quality_report.json").write_text(
        json.dumps(quality_report, indent=2),
        encoding="utf-8",
    )

    summary = {
        "matches_input": len(matches_rows),
        "matches_modeled": len(match_records),
        "team_form_windows": config.team_form_windows,
        "match_level_feature_rows": len(match_level_rows),
        "match_team_feature_rows": len(match_team_rows),
        "live_state_feature_rows": len(live_rows),
        "output_dir": str(config.output_dir),
    }

    (config.output_dir / "feature_engineering_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary
