"""
Squad Selection Optimization — Select optimal playing XI using constrained optimization.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_player_pool(conn, team: str, source: str = "t20i") -> pd.DataFrame:
    """Get player pool with batting and bowling stats for squad selection."""
    df = conn.execute(f"""
        WITH batting AS (
            SELECT b.player_key,
                   COUNT(*) AS bat_innings,
                   ROUND(AVG(b.runs_scored), 1) AS avg_runs,
                   ROUND(AVG(b.strike_rate), 1) AS avg_sr,
                   SUM(b.runs_scored) AS total_runs
            FROM gold.fact_batting_innings b
            JOIN gold.dim_team t ON b.team_key = t.team_key
            WHERE t.team_name = '{team}' AND b.source = '{source}'
            GROUP BY b.player_key
        ),
        bowling AS (
            SELECT b.player_key,
                   COUNT(*) AS bowl_innings,
                   ROUND(AVG(b.wickets_taken), 2) AS avg_wickets,
                   ROUND(AVG(b.economy_rate), 2) AS avg_economy,
                   SUM(b.wickets_taken) AS total_wickets
            FROM gold.fact_bowling_innings b
            JOIN gold.dim_team t ON b.team_key = t.team_key
            WHERE t.team_name = '{team}' AND b.source = '{source}'
            GROUP BY b.player_key
        )
        SELECT p.player_name, p.player_key,
               COALESCE(ba.bat_innings, 0) AS bat_innings,
               COALESCE(ba.avg_runs, 0) AS avg_runs,
               COALESCE(ba.avg_sr, 0) AS avg_sr,
               COALESCE(ba.total_runs, 0) AS total_runs,
               COALESCE(bo.bowl_innings, 0) AS bowl_innings,
               COALESCE(bo.avg_wickets, 0) AS avg_wickets,
               COALESCE(bo.avg_economy, 0) AS avg_economy,
               COALESCE(bo.total_wickets, 0) AS total_wickets,
               CASE
                   WHEN COALESCE(ba.bat_innings, 0) >= 5 AND COALESCE(bo.bowl_innings, 0) >= 5
                       THEN 'All-Rounder'
                   WHEN COALESCE(bo.bowl_innings, 0) >= 5
                       THEN 'Bowler'
                   ELSE 'Batter'
               END AS role
        FROM gold.dim_player p
        LEFT JOIN batting ba ON p.player_key = ba.player_key
        LEFT JOIN bowling bo ON p.player_key = bo.player_key
        WHERE (COALESCE(ba.bat_innings, 0) >= 3 OR COALESCE(bo.bowl_innings, 0) >= 3)
          AND p.teams_played LIKE '%{team}%'
        ORDER BY COALESCE(ba.total_runs, 0) + COALESCE(bo.total_wickets, 0) * 25 DESC
    """).df()
    return df


def compute_player_value(player_pool: pd.DataFrame) -> pd.DataFrame:
    """Compute a composite value score for each player."""
    df = player_pool.copy()

    # Normalize stats to 0-1 scale
    for col in ["avg_runs", "avg_sr", "total_runs", "avg_wickets", "total_wickets"]:
        col_max = df[col].max()
        if col_max > 0:
            df[f"{col}_norm"] = df[col] / col_max
        else:
            df[f"{col}_norm"] = 0

    # Economy: lower is better
    econ_max = df["avg_economy"].max()
    if econ_max > 0:
        df["economy_norm"] = 1 - (df["avg_economy"] / econ_max)
    else:
        df["economy_norm"] = 0

    # Overall value
    df["value"] = (
        df["avg_runs_norm"] * 0.25 +
        df["avg_sr_norm"] * 0.1 +
        df["total_runs_norm"] * 0.15 +
        df["avg_wickets_norm"] * 0.2 +
        df["total_wickets_norm"] * 0.15 +
        df["economy_norm"] * 0.15
    )

    return df.sort_values("value", ascending=False)


def select_squad(
    player_pool: pd.DataFrame,
    squad_size: int = 11,
    min_batters: int = 4,
    min_bowlers: int = 4,
    min_allrounders: int = 1,
    max_batters: int = 6,
    max_bowlers: int = 5,
) -> pd.DataFrame:
    """
    Select optimal playing XI using greedy constrained selection.

    Constraints:
    - Exactly squad_size players
    - Minimum/maximum batters, bowlers, all-rounders
    """
    df = compute_player_value(player_pool)

    if len(df) < squad_size:
        return df

    selected = []
    selected_names = set()

    # Phase 1: Mandatory all-rounders (highest value)
    ars = df[df["role"] == "All-Rounder"]
    for _, p in ars.head(min_allrounders).iterrows():
        selected.append(p)
        selected_names.add(p["player_name"])

    # Phase 2: Mandatory bowlers
    bowlers = df[(df["role"] == "Bowler") & (~df["player_name"].isin(selected_names))]
    for _, p in bowlers.head(min_bowlers).iterrows():
        selected.append(p)
        selected_names.add(p["player_name"])

    # Phase 3: Mandatory batters
    batters = df[(df["role"] == "Batter") & (~df["player_name"].isin(selected_names))]
    for _, p in batters.head(min_batters).iterrows():
        selected.append(p)
        selected_names.add(p["player_name"])

    # Phase 4: Fill remaining spots with highest value
    remaining_spots = squad_size - len(selected)
    remaining = df[~df["player_name"].isin(selected_names)]

    # Respect maximums
    current_batters = sum(1 for s in selected if s["role"] == "Batter")
    current_bowlers = sum(1 for s in selected if s["role"] == "Bowler")

    for _, p in remaining.iterrows():
        if len(selected) >= squad_size:
            break

        if p["role"] == "Batter" and current_batters >= max_batters:
            continue
        if p["role"] == "Bowler" and current_bowlers >= max_bowlers:
            continue

        selected.append(p)
        selected_names.add(p["player_name"])
        if p["role"] == "Batter":
            current_batters += 1
        elif p["role"] == "Bowler":
            current_bowlers += 1

    result = pd.DataFrame(selected)
    result = result.sort_values("value", ascending=False).reset_index(drop=True)
    result.insert(0, "Selection #", range(1, len(result) + 1))

    return result[["Selection #", "player_name", "role", "avg_runs", "avg_sr",
                    "avg_wickets", "avg_economy", "value"]]


if __name__ == "__main__":
    from src.warehouse.schema import get_connection

    conn = get_connection()
    pool = get_player_pool(conn, "India", source="t20i")
    conn.close()

    if not pool.empty:
        squad = select_squad(pool)
        print("\nOptimal Playing XI for India:")
        print(squad.to_string(index=False))
