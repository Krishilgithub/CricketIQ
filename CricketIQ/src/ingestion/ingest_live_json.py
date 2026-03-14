"""
src/ingestion/ingest_live_json.py
──────────────────────────────────
Incremental live ingestion: watches a "drop folder" for new Cricsheet
JSON files, converts them to normalized rows, and appends only unseen
match_ids into the DuckDB Bronze tables.

Designed to run on a scheduler (Prefect/cron) every 5–15 minutes during
live tournaments.

Usage:
    # One-shot: process any new files not yet in the DB
    python -m src.ingestion.ingest_live_json

    # Continuous watch mode (polls every N seconds)
    python -m src.ingestion.ingest_live_json --watch --interval 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from src.config import get_config, resolve_path
from src.ingestion.convert_new_json_to_csv import convert_json_file
from src.ingestion.ingest_historical import (
    CRICSHEET_TABLES,
    get_connection,
    _ingest_table,
)
from src.logger import get_logger

log = get_logger(__name__)

DEFAULT_DROP_FOLDER = Path("data/raw/new_json_drops")


def get_ingested_match_ids(duckdb_path: str) -> set[str]:
    """Return the set of match_ids already in bronze.matches."""
    try:
        con = duckdb.connect(duckdb_path, read_only=True)
        rows = con.execute(
            "SELECT match_id FROM bronze.matches"
        ).fetchall()
        con.close()
        return {r[0] for r in rows}
    except Exception:
        # Table doesn't exist yet — return empty set
        return set()


def _write_rows_to_tmp_csv(rows: list[dict], tmp_path: Path) -> None:
    """Write a list of dicts to a temporary CSV file for bulk DuckDB load."""
    import pandas as pd
    pd.DataFrame(rows).to_csv(tmp_path, index=False)


def ingest_new_json_files(
    duckdb_path: str,
    drop_folder: Path,
    tmp_dir: Path,
) -> dict[str, int]:
    """
    Scan drop_folder for .json files whose match_id is not yet in bronze.
    Convert, then insert rows into DuckDB bronze tables.
    Returns summary of {table: rows_inserted}.
    """
    drop_folder.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = get_ingested_match_ids(duckdb_path)
    json_files = sorted(drop_folder.glob("*.json"))
    new_files = [f for f in json_files if f.stem not in existing_ids]

    if not new_files:
        log.info("No new JSON match files to ingest.")
        return {}

    log.info(f"Found {len(new_files)} new match file(s) to ingest.")
    con = get_connection(duckdb_path)
    summary: dict[str, int] = {}

    for jf in new_files:
        log.info(f"Processing: {jf.name}")
        try:
            all_rows = convert_json_file(jf)
            for table_name, rows in all_rows.items():
                if not rows:
                    continue
                meta = CRICSHEET_TABLES.get(table_name, {"pk": ["match_id"]})
                tmp_csv = tmp_dir / f"_tmp_{table_name}.csv"

                # Write parsed rows to temp CSV then bulk-load into DuckDB
                _write_rows_to_tmp_csv(rows, tmp_csv)

                import pandas as pd
                chunk = pd.read_csv(tmp_csv, low_memory=False)
                con.register("_chunk", chunk)

                # Ensure table exists
                cols = []
                for col in chunk.columns:
                    _dt = meta.get("dtype", {})
                    if col in _dt and _dt[col] in ("string", "object"):
                        cols.append(f'CAST("{col}" AS VARCHAR) AS "{col}"')
                    else:
                        cols.append(f'"{col}"')
                select_clause = ", ".join(cols)
                
                con.execute(
                    f"CREATE TABLE IF NOT EXISTS bronze.{table_name} AS "
                    f"SELECT {select_clause} FROM _chunk WHERE 1=0"
                )

                # Idempotent insert
                pks = meta["pk"]
                pk_condition = " AND ".join(
                    [f"t.{p} = c.{p}" for p in pks]
                )
                count_before = con.execute(
                    f"SELECT COUNT(*) FROM bronze.{table_name}"
                ).fetchone()[0]
                con.execute(
                    f"INSERT INTO bronze.{table_name} "
                    f"SELECT c.* FROM _chunk c "
                    f"WHERE NOT EXISTS ("
                    f"  SELECT 1 FROM bronze.{table_name} t WHERE {pk_condition}"
                    f")"
                )
                count_after = con.execute(
                    f"SELECT COUNT(*) FROM bronze.{table_name}"
                ).fetchone()[0]
                inserted = count_after - count_before
                summary[table_name] = summary.get(table_name, 0) + inserted
                tmp_csv.unlink(missing_ok=True)

        except Exception as e:
            log.error(f"Failed to ingest {jf.name}: {e}")

    con.close()
    log.info(f"Live ingestion complete. Summary: {summary}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CricketIQ live/incremental JSON ingestion")
    parser.add_argument(
        "--drop-folder",
        default=str(DEFAULT_DROP_FOLDER),
        help="Folder to watch for new Cricsheet JSON files",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously, polling the folder every --interval seconds",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Poll interval in seconds when --watch is set (default: 300)",
    )
    args = parser.parse_args()

    cfg = get_config()
    duckdb_path = str(resolve_path(cfg["paths"]["duckdb_path"]))
    drop_folder = Path(args.drop_folder)
    tmp_dir = resolve_path("artifacts/.tmp_ingestion")

    if args.watch:
        log.info(f"Watch mode: polling {drop_folder} every {args.interval}s")
        while True:
            ingest_new_json_files(duckdb_path, drop_folder, tmp_dir)
            time.sleep(args.interval)
    else:
        ingest_new_json_files(duckdb_path, drop_folder, tmp_dir)


if __name__ == "__main__":
    main()
