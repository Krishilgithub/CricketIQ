"""
src/ingestion/ingest_historical.py
───────────────────────────────────
Batch ingestion: loads all 8 Cricsheet normalized CSVs from
`data/raw/cricsheet_csv_all/` into the DuckDB Bronze schema.

Bronze tables are created as exact copies of the source CSVs —
no cleaning, no filtering. Immutability is maintained by using
INSERT OR IGNORE semantics on the primary-key column (match_id).

Usage:
    python -m src.ingestion.ingest_historical
    python -m src.ingestion.ingest_historical --source given   # load given_data_csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from src.config import get_config, resolve_path
from src.logger import get_logger

log = get_logger(__name__)

# ── Table registry: name → primary key(s) for idempotent loads ───────────────
CRICSHEET_TABLES: dict[str, dict] = {
    "matches": {
        "pk": ["match_id"],
        "parse_dates": ["match_date"],
        "dtype": {
            "method": "string",
            "result_type": "string",
            "result_margin": "string",
            "city": "string"
        },
    },
    "match_teams": {
        "pk": ["match_id", "team"],
        "parse_dates": [],
    },
    "innings": {
        "pk": ["match_id", "innings_number", "team"],
        "parse_dates": [],
    },
    "deliveries": {
        "pk": ["match_id", "innings_number", "over", "ball_in_over", "batter", "bowler"],
        "parse_dates": [],
        "dtype": {
            "review_by": "string",
            "review_batter": "string",
            "review_decision": "string",
            "review_type": "string",
            "replacement_role": "string",
            "replacement_team": "string",
            "replacement_in": "string",
            "replacement_out": "string",
        },
    },
    "wickets": {
        "pk": ["match_id", "innings_number", "over", "ball_in_over", "player_out"],
        "parse_dates": [],
    },
    "powerplays": {
        "pk": ["match_id", "innings_number", "powerplay_type"],
        "parse_dates": [],
    },
    "player_of_match": {
        "pk": ["match_id", "player"],
        "parse_dates": [],
    },
    "officials": {
        "pk": ["match_id", "official_role", "official_name"],
        "parse_dates": [],
    },
}

# Given hackathon data tables (for optional load via --source given)
GIVEN_TABLES: dict[str, dict] = {
    "given_matches":            {"pk": ["match_no"], "parse_dates": ["date"]},
    "given_batting_stats":      {"pk": ["player", "team"], "parse_dates": []},
    "given_bowling_stats":      {"pk": ["player", "team"], "parse_dates": []},
    "given_key_scorecards":     {"pk": ["match", "innings", "player"], "parse_dates": []},
    "given_squads":             {"pk": ["team", "player_name"], "parse_dates": []},
    "given_points_table":       {"pk": ["group", "team"], "parse_dates": []},
    "given_venues":             {"pk": ["venue_name"], "parse_dates": []},
    "given_awards":             {"pk": ["award", "player_or_detail"], "parse_dates": []},
    "given_tournament_summary": {"pk": ["field"], "parse_dates": []},
}

GIVEN_FILE_MAP = {
    "given_matches":            "matches.csv",
    "given_batting_stats":      "batting_stats.csv",
    "given_bowling_stats":      "bowling_stats.csv",
    "given_key_scorecards":     "key_scorecards.csv",
    "given_squads":             "squads.csv",
    "given_points_table":       "points_table.csv",
    "given_venues":             "venues.csv",
    "given_awards":             "awards.csv",
    "given_tournament_summary": "tournament_summary.csv",
}


def get_connection(duckdb_path: str) -> duckdb.DuckDBPyConnection:
    """Open or create a DuckDB connection, initialising the bronze schema."""
    con = duckdb.connect(duckdb_path)
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    return con


def _ingest_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    csv_path: Path,
    meta: dict,
    schema: str = "bronze",
    chunk_size: int = 200_000,
) -> int:
    """
    Load a single CSV into a DuckDB bronze table.
    Uses chunked reading for large files (deliveries ~95 MB).
    Returns total rows inserted.
    """
    if not csv_path.exists():
        log.warning(f"CSV not found, skipping: {csv_path}")
        return 0

    full_table = f"{schema}.{table_name}"
    dtype = meta.get("dtype", {})
    parse_dates = meta.get("parse_dates", [])

    total_inserted = 0
    for i, chunk in enumerate(
        pd.read_csv(
            csv_path,
            dtype=dtype if dtype else None,
            parse_dates=parse_dates if parse_dates else False,
            chunksize=chunk_size,
            low_memory=False,
        )
    ):
        # Register as a DuckDB view so we can INSERT via SQL
        con.register("_chunk", chunk)

        if i == 0:
            # First chunk: CREATE TABLE IF NOT EXISTS from chunk schema
            cols = []
            for col in chunk.columns:
                if dtype and col in dtype and dtype[col] in ("string", "object"):
                    cols.append(f'CAST("{col}" AS VARCHAR) AS "{col}"')
                else:
                    cols.append(f'"{col}"')
            select_clause = ", ".join(cols)
            
            con.execute(
                f"CREATE TABLE IF NOT EXISTS {full_table} AS "
                f"SELECT {select_clause} FROM _chunk WHERE 1=0"
            )

        # Count before insert for delta tracking
        count_before = con.execute(f"SELECT COUNT(*) FROM {full_table}").fetchone()[0]

        # INSERT only rows whose PK doesn't yet exist
        pks = meta["pk"]
        pk_condition = " AND ".join(
            [f"t.{p} = c.{p}" for p in pks]
        )
        con.execute(
            f"INSERT INTO {full_table} "
            f"SELECT c.* FROM _chunk c "
            f"WHERE NOT EXISTS ("
            f"  SELECT 1 FROM {full_table} t WHERE {pk_condition}"
            f")"
        )
        count_after = con.execute(f"SELECT COUNT(*) FROM {full_table}").fetchone()[0]
        rows_added = count_after - count_before
        total_inserted += rows_added
        log.debug(f"  chunk {i}: +{rows_added} rows → {full_table}")

    log.info(f"✓ {table_name}: {total_inserted} new rows ingested into {full_table}")
    return total_inserted


def ingest_cricsheet(duckdb_path: str, data_dir: Path) -> dict[str, int]:
    """
    Ingest all Cricsheet CSVs into DuckDB bronze schema.
    Returns a summary dict of {table_name: rows_inserted}.
    """
    con = get_connection(duckdb_path)
    summary: dict[str, int] = {}

    log.info(f"Starting Cricsheet ingestion from: {data_dir}")
    for table_name, meta in CRICSHEET_TABLES.items():
        csv_path = data_dir / f"{table_name}.csv"
        rows = _ingest_table(con, table_name, csv_path, meta)
        summary[table_name] = rows

    con.close()
    log.info(f"Cricsheet ingestion complete. Summary: {summary}")
    return summary


def ingest_given(duckdb_path: str, data_dir: Path) -> dict[str, int]:
    """
    Ingest hackathon given_data_csv tables into DuckDB bronze schema.
    Returns a summary dict of {table_name: rows_inserted}.
    """
    con = get_connection(duckdb_path)
    summary: dict[str, int] = {}

    log.info(f"Starting given-data ingestion from: {data_dir}")
    for table_name, meta in GIVEN_TABLES.items():
        file_name = GIVEN_FILE_MAP[table_name]
        csv_path = data_dir / file_name
        rows = _ingest_table(con, table_name, csv_path, meta)
        summary[table_name] = rows

    con.close()
    log.info(f"Given-data ingestion complete. Summary: {summary}")
    return summary


def print_bronze_row_counts(duckdb_path: str) -> None:
    """Print row counts for all bronze tables (useful for verification)."""
    con = duckdb.connect(duckdb_path)
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'bronze' ORDER BY table_name"
    ).fetchall()
    print("\n── Bronze Row Counts ──────────────────────")
    for (t,) in tables:
        count = con.execute(f"SELECT COUNT(*) FROM bronze.{t}").fetchone()[0]
        print(f"  {t:40s}: {count:>10,}")
    print("───────────────────────────────────────────")
    con.close()


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="CricketIQ historical batch ingestion")
    parser.add_argument(
        "--source",
        choices=["cricsheet", "given", "both"],
        default="both",
        help="Which dataset(s) to ingest (default: both)",
    )
    args = parser.parse_args()

    cfg = get_config()
    duckdb_path = resolve_path(cfg["paths"]["duckdb_path"])
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    cricsheet_dir = resolve_path(cfg["paths"]["data_raw_cricsheet"])
    given_dir = resolve_path(cfg["paths"]["data_raw_given"])

    if args.source in ("cricsheet", "both"):
        ingest_cricsheet(str(duckdb_path), cricsheet_dir)

    if args.source in ("given", "both"):
        ingest_given(str(duckdb_path), given_dir)

    print_bronze_row_counts(str(duckdb_path))


if __name__ == "__main__":
    main()
