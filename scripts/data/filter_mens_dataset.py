from __future__ import annotations

import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path.cwd()
    source_dir = root / "data" / "raw" / "cricsheet_csv_all"
    target_dir = root / "data" / "processed" / "cricsheet_csv_men"

    matches = read_csv(source_dir / "matches.csv")

    men_matches = [
        r
        for r in matches
        if r.get("gender", "") == "male"
        and r.get("match_type", "") == "T20"
        and r.get("team_type", "") == "international"
    ]

    men_match_ids = {r["match_id"] for r in men_matches}

    linked_files = [
        "match_teams.csv",
        "innings.csv",
        "deliveries.csv",
        "wickets.csv",
        "powerplays.csv",
        "player_of_match.csv",
        "officials.csv",
    ]

    write_csv(target_dir / "matches.csv", men_matches, list(men_matches[0].keys()) if men_matches else [])

    stats: dict[str, int] = {"matches.csv": len(men_matches)}

    for name in linked_files:
        rows = read_csv(source_dir / name)
        filtered = [r for r in rows if r.get("match_id", "") in men_match_ids]
        fieldnames = list(rows[0].keys()) if rows else []
        write_csv(target_dir / name, filtered, fieldnames)
        stats[name] = len(filtered)

    print("Created men-only dataset at:", target_dir)
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
