# Power BI Guide — ICC T20 WC 2026 Predictor

## Importing Data

1. Open **Power BI Desktop**
2. Click **Get Data → Text/CSV**
3. Navigate to `data/powerbi_export/`
4. Import these files:

| File | Description |
|------|-------------|
| `dim_team.csv` | Team dimension with ICC membership |
| `dim_player.csv` | Players with teams and match counts |
| `dim_venue.csv` | Venues with avg scores and match counts |
| `dim_date.csv` | Date dimension (year, month, quarter) |
| `dim_tournament.csv` | Tournament details |
| `fact_innings_summary.csv` | Per-innings aggregated stats |
| `fact_batting_innings.csv` | Player batting per match |
| `fact_bowling_innings.csv` | Player bowling per match |
| `kpi_coach_team_performance.csv` | Pre-computed team KPIs |
| `kpi_analyst_player_impact.csv` | Pre-computed player impact scores |
| `kpi_broadcaster_h2h.csv` | Head-to-head records |

## Data Model Relationships

Create these relationships in Power BI Model view:

```
dim_team.team_key → fact_innings_summary.team_key
dim_venue.venue_key → fact_innings_summary.venue_key
dim_player.player_key → fact_batting_innings.player_key
dim_player.player_key → fact_bowling_innings.player_key
dim_team.team_key → fact_batting_innings.team_key
dim_date.date_key → fact_batting_innings.date_key
```

## Recommended DAX Measures

```dax
// Win Rate %
Win Rate = 
DIVIDE(
    CALCULATE(COUNTROWS(fact_match_results), fact_match_results[winner_key] = SELECTEDVALUE(dim_team[team_key])),
    COUNTROWS(fact_match_results)
) * 100

// Batting Average
Batting Avg = DIVIDE(SUM(fact_batting_innings[runs_scored]), COUNTROWS(fact_batting_innings))

// Strike Rate
Strike Rate = DIVIDE(SUM(fact_batting_innings[runs_scored]), SUM(fact_batting_innings[balls_faced])) * 100

// Economy Rate
Economy Rate = DIVIDE(SUM(fact_bowling_innings[runs_conceded]) * 6, SUM(fact_bowling_innings[balls_bowled]))

// Boundary %
Boundary Pct = DIVIDE(SUM(fact_innings_summary[boundary_runs]), SUM(fact_innings_summary[total_runs])) * 100

// Powerplay Net RR
PP Net RR = AVERAGE(fact_innings_summary[pp_run_rate]) - 7.0
```

## Suggested Visuals by Persona

### Coach Dashboard
- **Card** visuals for Win Rate, Bat First %, Chase %
- **Clustered Bar** for phase-wise runs (PP/Middle/Death)
- **Line chart** for win rate trend over time
- **Slicer** by team and source

### Analyst Dashboard
- **Matrix** for feature correlation (use conditional formatting)
- **KPI** card for player impact scores
- **Scatter** plot for batting avg vs strike rate

### Broadcaster Dashboard
- **Table** for records and milestones
- **Donut** chart for head-to-head results
- **Ribbon** chart for team rankings over seasons

### Fan / ICC Dashboard
- **Gauge** for entertainment metrics
- **Map** visual for venue locations (if geo data added)
- **Treemap** for team performance hierarchy

## Generating the Export

Run this command from the project root:
```bash
python src/dashboards/powerbi_export.py
```
