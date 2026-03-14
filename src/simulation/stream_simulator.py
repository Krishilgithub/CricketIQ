"""
Data Source Simulation — Stream ball-by-ball events at intervals.

Simulates a real-time data feed by reading parsed match data
and publishing events with configurable delays. Supports:
  - Console output (default)
  - File-based output (CSV)
  - Kafka (if available)

Usage:
    python src/simulation/stream_simulator.py                     # All matches, console
    python src/simulation/stream_simulator.py --match-id 12345    # Single match
    python src/simulation/stream_simulator.py --output csv        # Write to CSV
    python src/simulation/stream_simulator.py --interval 2        # 2 sec/ball
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    RAW_T20I_DIR, SIMULATION_INTERVAL_SECONDS,
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_BALL_EVENTS, KAFKA_TOPIC_MATCH_SUMMARY,
)
from src.ingestion.parse_cricsheet import CricsheetParser


class BallEventSimulator:
    """Simulates ball-by-ball streaming from Cricsheet match files."""

    def __init__(self, data_dir: Path, interval: float = SIMULATION_INTERVAL_SECONDS):
        self.data_dir = Path(data_dir)
        self.interval = interval
        self.parser = CricsheetParser()
        self._kafka_producer = None

    def _get_match_files(self, match_id: str = None) -> list[Path]:
        """Get JSON match files to simulate."""
        if match_id:
            filepath = self.data_dir / f"{match_id}.json"
            if filepath.exists():
                return [filepath]
            else:
                print(f"❌ Match file not found: {filepath}")
                return []
        return sorted(self.data_dir.glob("*.json"))

    def stream_to_console(self, match_id: str = None, max_matches: int = 5):
        """Stream ball events to console output."""
        match_files = self._get_match_files(match_id)[:max_matches]

        for filepath in match_files:
            result = self.parser.parse_match_file(filepath)
            if not result:
                continue

            info = result["match_info"]
            print(f"\n{'='*60}")
            print(f"🏏 MATCH: {info['team1']} vs {info['team2']}")
            print(f"   📍 {info['venue']}, {info['match_date']}")
            print(f"   🏆 {info.get('event_name', 'N/A')}")
            print(f"{'='*60}")

            current_innings = 0
            for delivery in result["deliveries"]:
                # Innings change detection
                if delivery["innings"] != current_innings:
                    current_innings = delivery["innings"]
                    print(f"\n── Innings {current_innings}: {delivery['batting_team']} batting ──")

                # Format ball event
                event_str = self._format_ball_event(delivery)
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"  [{timestamp}] {event_str}")

                time.sleep(self.interval)

            # Match summary
            print(f"\n🏆 Result: {info.get('winner', 'No result')}", end="")
            if info.get("result_type") == "runs":
                print(f" won by {info['result_margin']} runs")
            elif info.get("result_type") == "wickets":
                print(f" won by {info['result_margin']} wickets")
            else:
                print(f" ({info.get('result_type', 'N/A')})")

    def stream_to_csv(self, output_dir: Path, match_id: str = None, max_matches: int = None):
        """Stream ball events to CSV files for downstream consumption."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ball_csv = output_dir / "live_ball_events.csv"
        summary_csv = output_dir / "live_match_summaries.csv"

        match_files = self._get_match_files(match_id)
        if max_matches:
            match_files = match_files[:max_matches]

        print(f"📝 Streaming {len(match_files)} matches to CSV...")
        print(f"   Ball events: {ball_csv}")
        print(f"   Summaries:   {summary_csv}")

        # Write headers
        ball_headers = [
            "timestamp", "match_id", "innings", "over", "ball", "batting_team",
            "batter", "bowler", "total_runs", "is_wicket", "wicket_kind",
            "cumulative_score", "cumulative_wickets", "phase"
        ]

        with open(ball_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ball_headers)

        summary_headers = [
            "timestamp", "match_id", "team1", "team2", "venue",
            "winner", "result_type", "result_margin"
        ]

        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(summary_headers)

        for filepath in match_files:
            result = self.parser.parse_match_file(filepath)
            if not result:
                continue

            info = result["match_info"]
            cum_runs = {1: 0, 2: 0}
            cum_wickets = {1: 0, 2: 0}

            for delivery in result["deliveries"]:
                innings = delivery["innings"]
                cum_runs[innings] += delivery["total_runs"]
                cum_wickets[innings] += 1 if delivery["is_wicket"] else 0

                with open(ball_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        delivery["match_id"],
                        innings,
                        delivery["over"],
                        delivery["ball"],
                        delivery["batting_team"],
                        delivery["batter"],
                        delivery["bowler"],
                        delivery["total_runs"],
                        delivery["is_wicket"],
                        delivery["wicket_kind"],
                        cum_runs[innings],
                        cum_wickets[innings],
                        delivery["phase"],
                    ])

                time.sleep(self.interval)

            # Write match summary
            with open(summary_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    info["match_id"],
                    info["team1"],
                    info["team2"],
                    info["venue"],
                    info.get("winner", ""),
                    info.get("result_type", ""),
                    info.get("result_margin", ""),
                ])

            print(f"  ✅ Streamed: {info['team1']} vs {info['team2']}")

    def stream_to_kafka(self, match_id: str = None, max_matches: int = None):
        """Stream ball events to Kafka topics."""
        try:
            from kafka import KafkaProducer
        except ImportError:
            print("❌ kafka-python not installed. Install with: pip install kafka-python")
            return

        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        match_files = self._get_match_files(match_id)
        if max_matches:
            match_files = match_files[:max_matches]

        print(f"📡 Streaming to Kafka ({KAFKA_BOOTSTRAP_SERVERS})")
        print(f"   Ball topic:    {KAFKA_TOPIC_BALL_EVENTS}")
        print(f"   Summary topic: {KAFKA_TOPIC_MATCH_SUMMARY}")

        for filepath in match_files:
            result = self.parser.parse_match_file(filepath)
            if not result:
                continue

            info = result["match_info"]
            for delivery in result["deliveries"]:
                delivery["timestamp"] = datetime.now().isoformat()
                producer.send(KAFKA_TOPIC_BALL_EVENTS, value=delivery)
                time.sleep(self.interval)

            # Publish match summary
            info["timestamp"] = datetime.now().isoformat()
            producer.send(KAFKA_TOPIC_MATCH_SUMMARY, value=info)
            print(f"  ✅ Streamed: {info['team1']} vs {info['team2']}")

        producer.flush()
        producer.close()

    def _format_ball_event(self, delivery: dict) -> str:
        """Format a ball event for console display."""
        over_ball = f"{delivery['over']}.{delivery['ball']}"
        runs = delivery["total_runs"]
        batter = delivery["batter"][:15].ljust(15)
        bowler = delivery["bowler"][:15].ljust(15)

        event = f"{over_ball:>4} | {batter} vs {bowler} | "

        if delivery["is_wicket"]:
            event += f"🔴 WICKET! {delivery['player_out']} ({delivery['wicket_kind']})"
        elif delivery["is_boundary_six"]:
            event += "💥 SIX!"
        elif delivery["is_boundary_four"]:
            event += "4️⃣  FOUR!"
        elif delivery["is_dot_ball"]:
            event += "⚫ Dot ball"
        elif delivery["is_wide"]:
            event += f"〰️  Wide (+{runs})"
        elif delivery["is_noball"]:
            event += f"❌ No ball (+{runs})"
        else:
            event += f"▶️  {runs} run{'s' if runs != 1 else ''}"

        return event


def main():
    p = argparse.ArgumentParser(description="Ball-by-ball streaming simulator")
    p.add_argument("--data-dir", type=Path, default=RAW_T20I_DIR,
                    help="Directory with JSON match files")
    p.add_argument("--match-id", type=str, default=None,
                    help="Specific match ID to simulate")
    p.add_argument("--interval", type=float, default=SIMULATION_INTERVAL_SECONDS,
                    help="Seconds between ball events")
    p.add_argument("--output", choices=["console", "csv", "kafka"], default="console",
                    help="Output mode")
    p.add_argument("--output-dir", type=Path, default=Path("data/simulation_output"),
                    help="Output directory for CSV mode")
    p.add_argument("--max-matches", type=int, default=5,
                    help="Max matches to simulate")
    args = p.parse_args()

    simulator = BallEventSimulator(args.data_dir, interval=args.interval)

    if args.output == "console":
        simulator.stream_to_console(args.match_id, args.max_matches)
    elif args.output == "csv":
        simulator.stream_to_csv(args.output_dir, args.match_id, args.max_matches)
    elif args.output == "kafka":
        simulator.stream_to_kafka(args.match_id, args.max_matches)


if __name__ == "__main__":
    main()
