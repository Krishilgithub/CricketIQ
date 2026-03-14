"""
Batting Order Optimization — Maximize expected total score using LP / heuristic.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_player_phase_stats(conn, team: str, source: str = "t20i") -> pd.DataFrame:
    """Get player batting stats by phase for a team."""
    df = conn.execute(f"""
        SELECT p.player_name, p.player_key,
               COUNT(*) AS innings,
               ROUND(AVG(b.runs_scored), 1) AS avg_runs,
               ROUND(AVG(b.strike_rate), 1) AS avg_sr,
               ROUND(AVG(b.pp_runs), 1) AS avg_pp_runs,
               ROUND(AVG(b.middle_runs), 1) AS avg_mid_runs,
               ROUND(AVG(b.death_runs), 1) AS avg_death_runs,
               SUM(b.sixes) AS total_sixes,
               SUM(b.fours) AS total_fours
        FROM gold.fact_batting_innings b
        JOIN gold.dim_player p ON b.player_key = p.player_key
        JOIN gold.dim_team t ON b.team_key = t.team_key
        WHERE t.team_name = '{team}'
          AND b.source = '{source}'
        GROUP BY p.player_name, p.player_key
        HAVING COUNT(*) >= 3
        ORDER BY avg_runs DESC
    """).df()
    return df


def optimize_batting_order(player_stats: pd.DataFrame, playing_xi: list = None) -> pd.DataFrame:
    """
    Optimize batting order based on phase specialization.

    Strategy:
    - Positions 1-3 (Powerplay): Best PP run-scorers with high SR
    - Positions 4-5 (Middle/Anchor): Highest average, consistency
    - Positions 6-7 (Finishers): Best death-over scorers, high SR
    - Positions 8-11 (Lower order): Remaining by overall average

    Returns DataFrame with optimized order.
    """
    df = player_stats.copy()

    if playing_xi:
        df = df[df["player_name"].isin(playing_xi)]

    if len(df) < 3:
        return df

    n = min(len(df), 11)

    # Score each player for each role
    df["pp_score"] = (
        df["avg_pp_runs"].fillna(0) * 0.6 +
        df["avg_sr"].fillna(0) * 0.4 / 100 * df["avg_pp_runs"].fillna(0)
    )
    df["anchor_score"] = (
        df["avg_runs"].fillna(0) * 0.7 +
        df["avg_mid_runs"].fillna(0) * 0.3
    )
    df["finisher_score"] = (
        df["avg_death_runs"].fillna(0) * 0.5 +
        df["avg_sr"].fillna(0) * 0.3 / 100 * df["avg_death_runs"].fillna(0) +
        df["total_sixes"].fillna(0) * 0.2 / max(df["total_sixes"].max(), 1) * 30
    )

    assigned = set()
    order = []

    # Positions 1-3: Best PP scorers
    pp_candidates = df[~df["player_name"].isin(assigned)].nlargest(
        min(3, n), "pp_score"
    )
    for _, p in pp_candidates.iterrows():
        order.append(p["player_name"])
        assigned.add(p["player_name"])

    # Positions 4-5: Anchors
    remaining = df[~df["player_name"].isin(assigned)]
    anchors = remaining.nlargest(min(2, max(0, n - len(order))), "anchor_score")
    for _, p in anchors.iterrows():
        order.append(p["player_name"])
        assigned.add(p["player_name"])

    # Positions 6-7: Finishers
    remaining = df[~df["player_name"].isin(assigned)]
    finishers = remaining.nlargest(min(2, max(0, n - len(order))), "finisher_score")
    for _, p in finishers.iterrows():
        order.append(p["player_name"])
        assigned.add(p["player_name"])

    # Positions 8-11: Rest by overall average
    remaining = df[~df["player_name"].isin(assigned)]
    rest = remaining.nlargest(max(0, n - len(order)), "avg_runs")
    for _, p in rest.iterrows():
        order.append(p["player_name"])

    # Build result
    result = []
    for i, name in enumerate(order):
        player = df[df["player_name"] == name].iloc[0]
        role = "Opener" if i < 2 else ("Top Order" if i == 2 else
               ("Middle Order" if i < 5 else ("Finisher" if i < 7 else "Lower Order")))
        result.append({
            "Position": i + 1,
            "Player": name,
            "Role": role,
            "Avg Runs": player["avg_runs"],
            "Avg SR": player["avg_sr"],
            "PP Runs": player["avg_pp_runs"],
            "Death Runs": player["avg_death_runs"],
            "Innings": player["innings"],
        })

    return pd.DataFrame(result)


if __name__ == "__main__":
    from src.warehouse.schema import get_connection

    conn = get_connection()
    stats = get_player_phase_stats(conn, "India", source="t20i")
    conn.close()

    if not stats.empty:
        order = optimize_batting_order(stats)
        print("\nOptimized Batting Order for India:")
        print(order.to_string(index=False))
