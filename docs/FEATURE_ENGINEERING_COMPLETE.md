# Feature Engineering Completion Summary

This document summarizes the completed feature engineering implementation for men-only international T20 data.

## Input Scope

- Source folder: `data/cricsheet_csv_men`
- Filters enforced:
  - `match_type = T20`
  - `team_type = international`
  - `gender = male`
  - `overs = 20`

## Implemented Pipeline

Main implementation:

- `src/cricketiq/feature_engineering/full_pipeline.py`

Runner:

- `scripts/run_feature_engineering.py`

Config:

- `configs/feature_engineering.json`

## Implemented Outputs

- `artifacts/features/match_level_features.csv`
- `artifacts/features/match_team_features.csv`
- `artifacts/features/live_state_features.csv`
- `artifacts/features/feature_quality_report.json`
- `artifacts/features/feature_engineering_summary.json`

## Implemented Feature Families

### 1) Label and Outcome Engineering

- `outcome_class` with classes: `win`, `loss`, `tie`, `no_result`
- `team_1_win` binary target for valid win/loss matches
- `label_is_binary` flag for training filtering

### 2) Rolling Team Form (5 and 10 match windows)

- win rate and weighted win rate
- win rate uncertainty
- sample size per window
- average runs scored and conceded
- run rate
- powerplay, middle, death run rates
- dot-ball percentage
- boundary percentage
- wides and no-ball conceded rates
- death wickets lost rate
- clutch index (death RR minus middle RR)

### 3) Matchup and Venue Context

- team venue win rates
- head-to-head counts and team_1 H2H win rate
- venue average first innings score
- venue chase success rate
- venue-season average first innings score
- venue-season chase success rate
- season average first innings score
- season chase success rate
- venue-vs-season normalized deltas

### 4) Difference Features

For primary window (`5`):

- win-rate difference
- weighted win-rate difference
- average-runs difference
- phase run-rate differences (powerplay, middle, death)
- boundary and dot-ball differences

### 5) Live State Features (Ball-by-ball)

- score and wickets after each ball
- legal balls bowled and balls remaining
- current run rate
- target, runs required, required run rate for second innings
- batting and bowling team context
- phase label per ball

## Data Quality and Integrity Checks Implemented

Quality report generated at:

- `artifacts/features/feature_quality_report.json`

Checks included:

- duplicate delivery key detection
- duplicate wicket key detection
- innings total consistency: innings table vs deliveries sums
- innings wicket consistency: innings table vs wicket events
- suspicious missing fielders for dismissal kinds where fielders are expected
- missing city and event name counts
- class distribution report

## Current Run Summary

From `feature_engineering_summary.json`:

- matches_input: 3211
- matches_modeled: 3203
- team_form_windows: [5, 10]
- match_level_feature_rows: 3203
- match_team_feature_rows: 6406
- live_state_feature_rows: 722512

From `feature_quality_report.json`:

- duplicate_delivery_keys: 0
- duplicate_wicket_keys: 0
- innings_delivery_mismatch_count: 0
- innings_wicket_mismatch_count: 0

## Notes

- Feature generation is strictly chronological to avoid leakage from future matches.
- Historical statistics are computed before updating state with the current match.
- Super over innings are excluded from baseline historical aggregates.
