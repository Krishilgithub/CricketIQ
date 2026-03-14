"""
src/simulation/live_match_simulator.py
───────────────────────────────────────
Replays a historical match from DuckDB bronze tables as a sequence of
time-delayed events, simulating a live match feed.

Each emitted event is a dictionary representing the current ball state,
which downstream components (live inference, dashboard refresh) can
subscribe to via the provided callback or async queue.

Event types emitted:
  - MATCH_START    : toss + teams + venue
  - OVER_START     : over number, bowling team
  - DELIVERY       : full ball-by-ball state snapshot
  - WICKET         : dismissal details
  - OVER_END       : over score summary
  - INNINGS_END    : innings total
  - MATCH_END      : final result

Usage:
    # Simulate a specific match (prints events to stdout)
    python -m src.simulation.live_match_simulator --match-id <match_id>

    # Simulate the most recent match in the DB
    python -m src.simulation.live_match_simulator --latest

    # Adjust replay speed (0 = instant, 1 = real-time 1s/ball, default 0.5)
    python -m src.simulation.live_match_simulator --match-id <id> --speed 0.5
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import duckdb
import pandas as pd

from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

EventCallback = Callable[[dict], None]


# ── Event dataclasses ─────────────────────────────────────────────────────────

@dataclass
class MatchStartEvent:
    event_type: str = "MATCH_START"
    match_id: str = ""
    match_date: str = ""
    team_1: str = ""
    team_2: str = ""
    venue: str = ""
    toss_winner: str = ""
    toss_decision: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class DeliveryEvent:
    event_type: str = "DELIVERY"
    match_id: str = ""
    innings_number: int = 0
    over: int = 0
    ball_in_over: int = 0
    batting_team: str = ""
    batter: str = ""
    bowler: str = ""
    runs_batter: int = 0
    runs_extras: int = 0
    runs_total: int = 0
    # Running totals
    innings_runs_so_far: int = 0
    innings_wickets_so_far: int = 0
    innings_balls_so_far: int = 0
    # Wicket details (if any)
    is_wicket: bool = False
    wicket_player_out: str = ""
    wicket_kind: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class InningsEndEvent:
    event_type: str = "INNINGS_END"
    match_id: str = ""
    innings_number: int = 0
    team: str = ""
    total_runs: int = 0
    total_wickets: int = 0
    total_balls: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class MatchEndEvent:
    event_type: str = "MATCH_END"
    match_id: str = ""
    winner: str = ""
    result_text: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Simulator core ────────────────────────────────────────────────────────────

class LiveMatchSimulator:
    """
    Replays a historical match from DuckDB as a stream of ball-by-ball events.

    Args:
        duckdb_path: Path to the DuckDB file.
        match_id:    The match to replay.
        speed:       Seconds to wait between delivery events (0 = instant).
        callback:    Optional function(event_dict) called for each event.
    """

    def __init__(
        self,
        duckdb_path: str,
        match_id: str,
        speed: float = 0.5,
        callback: EventCallback | None = None,
    ) -> None:
        self.duckdb_path = duckdb_path
        self.match_id = match_id
        self.speed = speed
        self.callback = callback or self._default_callback
        self._con: duckdb.DuckDBPyConnection | None = None

    # ── Data loaders ──────────────────────────────────────────────────────────

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.duckdb_path, read_only=True)

    def _load_match(self) -> pd.Series:
        con = self._connect()
        df = con.execute(
            "SELECT * FROM bronze.matches WHERE match_id = ?", [self.match_id]
        ).df()
        con.close()
        if df.empty:
            raise ValueError(f"match_id '{self.match_id}' not found in bronze.matches")
        return df.iloc[0]

    def _load_teams(self) -> list[str]:
        con = self._connect()
        teams = con.execute(
            "SELECT team FROM bronze.match_teams WHERE match_id = ? ORDER BY team",
            [self.match_id],
        ).fetchall()
        con.close()
        return [t[0] for t in teams]

    def _load_deliveries(self) -> pd.DataFrame:
        con = self._connect()
        df = con.execute(
            """
            SELECT d.*, w.player_out, w.kind AS wicket_kind, w.fielders
            FROM bronze.deliveries d
            LEFT JOIN bronze.wickets w
              ON  d.match_id      = w.match_id
              AND d.innings_number = w.innings_number
              AND d.over           = w.over
              AND d.ball_in_over   = w.ball_in_over
              AND d.batter         = w.player_out
            WHERE d.match_id = ?
            ORDER BY d.innings_number, d.over, d.ball_in_over
            """,
            [self.match_id],
        ).df()
        con.close()
        return df

    # ── Event generation ──────────────────────────────────────────────────────

    def _emit(self, event) -> None:
        payload = asdict(event) if hasattr(event, "__dataclass_fields__") else event
        self.callback(payload)
        if self.speed > 0:
            time.sleep(self.speed)

    @staticmethod
    def _default_callback(event: dict) -> None:
        print(json.dumps(event, default=str))

    def run(self) -> None:
        """Execute the full match replay from MATCH_START to MATCH_END."""
        log.info(f"Starting simulator for match_id={self.match_id}")

        match = self._load_match()
        teams = self._load_teams()
        team_1 = teams[0] if len(teams) > 0 else ""
        team_2 = teams[1] if len(teams) > 1 else ""

        # MATCH_START
        self._emit(MatchStartEvent(
            match_id=self.match_id,
            match_date=str(match.get("match_date", "")),
            team_1=team_1,
            team_2=team_2,
            venue=str(match.get("venue", "")),
            toss_winner=str(match.get("toss_winner", "")),
            toss_decision=str(match.get("toss_decision", "")),
        ))

        deliveries = self._load_deliveries()
        if deliveries.empty:
            log.warning(f"No deliveries found for match_id={self.match_id}")
            return

        # Replay innings and balls
        for innings_num, inn_df in deliveries.groupby("innings_number"):
            runs_so_far = wickets_so_far = balls_so_far = 0
            batting_team = inn_df.iloc[0]["batting_team"]

            for _, ball in inn_df.iterrows():
                is_wide = bool(ball.get("extras_wides", 0))
                is_noball = bool(ball.get("extras_noballs", 0))
                # Only count legal deliveries toward balls faced
                is_legal = not (is_wide or is_noball)

                runs_so_far += int(ball.get("runs_total", 0))
                is_wicket = pd.notna(ball.get("player_out")) and ball.get("player_out", "") != ""
                if is_wicket:
                    wickets_so_far += 1
                if is_legal:
                    balls_so_far += 1

                self._emit(DeliveryEvent(
                    match_id=self.match_id,
                    innings_number=int(innings_num),
                    over=int(ball["over"]),
                    ball_in_over=int(ball["ball_in_over"]),
                    batting_team=str(ball["batting_team"]),
                    batter=str(ball.get("batter", "")),
                    bowler=str(ball.get("bowler", "")),
                    runs_batter=int(ball.get("runs_batter", 0)),
                    runs_extras=int(ball.get("runs_extras", 0)),
                    runs_total=int(ball.get("runs_total", 0)),
                    innings_runs_so_far=runs_so_far,
                    innings_wickets_so_far=wickets_so_far,
                    innings_balls_so_far=balls_so_far,
                    is_wicket=bool(is_wicket),
                    wicket_player_out=str(ball.get("player_out", "") or ""),
                    wicket_kind=str(ball.get("wicket_kind", "") or ""),
                ))

            # INNINGS_END
            self._emit(InningsEndEvent(
                match_id=self.match_id,
                innings_number=int(innings_num),
                team=batting_team,
                total_runs=runs_so_far,
                total_wickets=wickets_so_far,
                total_balls=balls_so_far,
            ))

        # MATCH_END
        self._emit(MatchEndEvent(
            match_id=self.match_id,
            winner=str(match.get("winner", "")),
            result_text=str(match.get("result_text", "")),
        ))
        log.info(f"Simulation complete for match_id={self.match_id}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _get_latest_match_id(duckdb_path: str) -> str:
    con = duckdb.connect(duckdb_path, read_only=True)
    mid = con.execute(
        "SELECT match_id FROM bronze.matches ORDER BY match_date DESC LIMIT 1"
    ).fetchone()
    con.close()
    if not mid:
        raise ValueError("No matches found in bronze.matches")
    return mid[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="CricketIQ live match simulator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--match-id", help="Specific match_id to simulate")
    group.add_argument("--latest", action="store_true", help="Simulate the most recent match")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Seconds between delivery events (0=instant, default=0.5)",
    )
    args = parser.parse_args()

    cfg = get_config()
    duckdb_path = str(resolve_path(cfg["paths"]["duckdb_path"]))

    match_id = args.match_id if not args.latest else _get_latest_match_id(duckdb_path)
    simulator = LiveMatchSimulator(duckdb_path=duckdb_path, match_id=match_id, speed=args.speed)
    simulator.run()


if __name__ == "__main__":
    main()
