from __future__ import annotations

import csv
import json
from pathlib import Path


def profile_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        null_counts = {c: 0 for c in cols}
        unique_samples = {c: set() for c in cols}
        row_count = 0

        for row in reader:
            row_count += 1
            for c in cols:
                v = row.get(c, "")
                if v is None or str(v).strip() == "":
                    null_counts[c] += 1
                if len(unique_samples[c]) < 20 and v is not None and str(v).strip() != "":
                    unique_samples[c].add(str(v))

    null_pct = {
        c: (null_counts[c] / row_count * 100 if row_count else 0.0)
        for c in cols
    }

    return {
        "file": path.name,
        "rows": row_count,
        "columns": cols,
        "null_pct": null_pct,
        "sample_unique_values": {c: sorted(list(v)) for c, v in unique_samples.items()},
    }


def main() -> None:
    root = Path.cwd()
    data_dir = root / "data" / "cricsheet_csv"
    out = root / "artifacts"
    out.mkdir(parents=True, exist_ok=True)

    profiles = []
    for p in sorted(data_dir.glob("*.csv")):
        profiles.append(profile_csv(p))

    (out / "cricsheet_profile.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    print(f"Wrote profile for {len(profiles)} files to artifacts/cricsheet_profile.json")


if __name__ == "__main__":
    main()
