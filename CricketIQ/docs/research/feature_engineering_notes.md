# CricketIQ

## Overview

CricketIQ is a cricket analytics project focused on predicting T20 match outcomes using historical Cricsheet-style data. The current workspace contains match-level, innings-level, ball-by-ball, wicket, powerplay, team, and official data that can support three modeling stages:

1. Pre-match winner prediction
2. Post-toss winner prediction
3. Live in-match win probability prediction

The safest baseline for this dataset is a post-toss match winner model built from historical aggregates only. Any feature used for a given match must be computed from matches played before that match date.

## Current Data

| File | Grain | Main Use |
| --- | --- | --- |
| `data/cricsheet_csv/matches.csv` | One row per match | Target, context, toss, venue |
| `data/cricsheet_csv/match_teams.csv` | Two rows per match | Team normalization into team_1 and team_2 |
| `data/cricsheet_csv/innings.csv` | One row per innings | Historical team batting and bowling aggregates |
| `data/cricsheet_csv/deliveries.csv` | One row per ball | Phase, style, player, and discipline features |
| `data/cricsheet_csv/wickets.csv` | One row per wicket | Dismissal timing and dismissal-type features |
| `data/cricsheet_csv/powerplays.csv` | One row per powerplay segment | Phase boundaries for powerplay metrics |
| `data/cricsheet_csv/player_of_match.csv` | One row per award entry | Historical player impact proxy |
| `data/cricsheet_csv/officials.csv` | One row per official assignment | Optional officiating-context features |

## Modeling Target

Recommended baseline target:

- Build a canonical match table with `team_1` and `team_2` from `match_teams.csv`.
- Create `team_1_win = 1` if `matches.winner == team_1`, else `0`.
- For ties, no result, abandoned, or missing winner rows, either drop them for the baseline or model them separately.

Recommended canonical team setup:

- Sort the two teams alphabetically inside each match to produce stable `team_1` and `team_2`.
- Express toss and all team-level historical features relative to `team_1` and `team_2`.
- Create difference features such as `win_rate_last_5_diff = team_1_win_rate_last_5 - team_2_win_rate_last_5`.

## Feature Engineering Rules

1. Never use `winner`, `result_margin`, `result_type`, `result_text`, current-match innings totals, or current-match ball outcomes as predictors for a pre-match or toss-time model.
2. Compute rolling features using only matches before the prediction date.
3. Split team features by innings role where useful, such as batting first versus chasing.
4. Derive phase features from ball-by-ball data using T20 phases:
   - powerplay: overs 0 to 5
   - middle: overs 6 to 14
   - death: overs 15 to 19
5. Prefer aggregates and comparative features over raw identifiers.

## Recommended Output Feature Table

The first training table should have one row per match and should include:

- match identifiers and match date
- `team_1`, `team_2`
- venue and city
- toss winner relative to team_1 or team_2
- toss decision
- rolling team form features for both teams
- venue-adjusted batting and bowling strength features for both teams
- head-to-head features
- phase features from historical deliveries
- target `team_1_win`

## Column-by-Column Feature Engineering Plan

### matches.csv

| Column | Role | Feature Engineering Plan | Use Stage | Leakage Risk |
| --- | --- | --- | --- | --- |
| `match_id` | Join key | Keep as primary key for joins across all tables. | All | None |
| `data_version` | Metadata | Drop. Source metadata only. | None | None |
| `created` | Metadata | Drop. File creation timestamp, not cricket signal. | None | None |
| `revision` | Metadata | Drop. Source revision, not match signal. | None | None |
| `match_date` | Core time column | Parse to date; derive year, month, quarter, day_of_week, days_since_last_match by team, days_since_last_match_at_venue, rolling windows. | All | Safe if based on past only |
| `season` | Context | Standardize to numeric season start year or season label; use for season trend features if needed. | All | Low |
| `event_name` | Context | Clean series or tournament name; optional one-hot or grouped category such as bilateral, league, world event if you later map events. | Pre-match, Post-toss | Low |
| `event_match_number` | Context | Convert to numeric; optional proxy for series stage or pressure. | Pre-match, Post-toss | Low |
| `match_type` | Filter | Filter to `T20`; drop after filtering unless multiple formats are modeled together. | Preprocessing | None |
| `match_type_number` | Metadata-like identifier | Usually drop. Global numbering has no cricket meaning for prediction. | None | None |
| `gender` | Filter | Filter to `male` if scope is ICC Men's T20. Drop after filtering. | Preprocessing | None |
| `team_type` | Filter | Filter to `international` if project scope is international T20. Drop after filtering. | Preprocessing | None |
| `venue` | Core categorical | Standardize names; encode venue; create venue win rate, average first innings score, chase success rate, venue-scoring index. | All | Safe if historical |
| `city` | Context categorical | Clean missing values; use with venue or as fallback when venue sparsity is high. | All | Low |
| `overs` | Match config | Usually constant at 20; drop unless shortened matches exist and you want an availability flag. | Preprocessing | Low |
| `balls_per_over` | Match config | Usually constant at 6; keep only if anomalies exist. | Preprocessing | Low |
| `toss_winner` | Core categorical | Re-express as `toss_winner_is_team_1`; optionally combine with venue and toss decision. | Post-toss, Live | Safe for post-toss only |
| `toss_decision` | Core categorical | One-hot encode `bat` or `field`; interact with venue chase bias and toss winner. | Post-toss, Live | Safe for post-toss only |
| `winner` | Target | Use only to create label `team_1_win`. Never use as predictor. | Target only | High |
| `result_type` | Outcome | Drop for pre-match and toss-time models. Can be a label-analysis field only. | None | High |
| `result_margin` | Outcome | Drop for winner model features. Use only for post-hoc analysis. | None | High |
| `result_text` | Outcome text | Drop. Free-text restatement of outcome. | None | High |
| `method` | Outcome context | Usually missing; if it indicates DLS or awarded result it is not known pre-match, so exclude from predictors. | None | High |

### match_teams.csv

| Column | Role | Feature Engineering Plan | Use Stage | Leakage Risk |
| --- | --- | --- | --- | --- |
| `match_id` | Join key | Group by `match_id` and pivot to two columns. | All | None |
| `team` | Core team identifier | Standardize team names; pivot two rows into `team_1` and `team_2`; build relative team features and difference features. | All | None |

### innings.csv

| Column | Role | Feature Engineering Plan | Use Stage | Leakage Risk |
| --- | --- | --- | --- | --- |
| `match_id` | Join key | Join to match table; aggregate historical innings metrics before each target match. | All | None |
| `innings_number` | Innings role | Keep to separate first-innings and second-innings historical behavior. | All | Safe if historical |
| `team` | Team identifier | Standardize and use to build batting-first, chasing, and bowling-conceded histories by team. | All | Safe if historical |
| `total_runs` | Historical aggregate input | Create rolling average runs, median runs, venue-adjusted scoring index, batting-first average, chasing average. | All | High if from same match |
| `total_wickets` | Historical aggregate input | Create wickets-lost rate, collapse tendency, bowling wickets-taken features from opponent perspective. | All | High if from same match |
| `total_balls` | Historical aggregate input | Use to derive run rate, finish speed, balls used in chase, defend efficiency. | All | High if from same match |
| `extras_byes` | Discipline or fielding noise | Aggregate historical byes conceded or received; low-priority feature. | All | High if from same match |
| `extras_legbyes` | Discipline or fielding noise | Aggregate historical leg-byes conceded or received; low-priority feature. | All | High if from same match |
| `extras_noballs` | Discipline | Build no-ball rate conceded and received. Stronger as bowling-discipline feature. | All | High if from same match |
| `extras_wides` | Discipline | Build wides rate conceded and received. Useful bowling-discipline feature. | All | High if from same match |
| `extras_penalty` | Rare event | Usually sparse; keep as rare-event count if present, otherwise drop. | All | High if from same match |

### deliveries.csv

| Column | Role | Feature Engineering Plan | Use Stage | Leakage Risk |
| --- | --- | --- | --- | --- |
| `match_id` | Join key | Join ball events to matches and innings for historical aggregation. | All | None |
| `innings_number` | Phase context | Separate first-innings and second-innings behavior; useful for batting-first versus chase splits. | All | Safe if historical |
| `over` | Phase builder | Bucket into powerplay, middle, and death overs; derive phase-specific scoring and wicket features. | All | High if from same match |
| `ball_in_over` | Ball index input | Combine with over to create `ball_number_in_innings` and legal-ball progress counters. | All | High if from same match |
| `batting_team` | Team identifier | Aggregate historical batting phase features by team and by venue. | All | Safe if historical |
| `batter` | Player identifier | Build optional player batting form, strike rate, dot-ball rate, boundary rate; use only if expected playing XI data exists. | All | Safe if historical |
| `bowler` | Player identifier | Build optional player bowling economy, wicket rate, death-over economy; use only if lineup mapping exists. | All | Safe if historical |
| `non_striker` | Player context | Usually low-priority; can be used for partnership features if player-level modeling is in scope. | All | Safe if historical |
| `runs_batter` | Ball outcome input | Aggregate to strike rate, singles rate, twos rate, boundary rate, average runs per ball, phase scoring strength. | All | High if from same match |
| `runs_extras` | Ball outcome input | Aggregate to extras share and opponent discipline indicators. | All | High if from same match |
| `runs_total` | Ball outcome input | Aggregate to run rate, scoring consistency, phase scoring curves, pressure scoring metrics. | All | High if from same match |
| `extras_byes` | Extra type | Historical byes rate; very low-priority signal. | All | High if from same match |
| `extras_legbyes` | Extra type | Historical leg-byes rate; low-priority signal. | All | High if from same match |
| `extras_noballs` | Discipline | Historical no-ball rate by bowling side and bowler. | All | High if from same match |
| `extras_wides` | Discipline | Historical wides rate by bowling side and bowler. | All | High if from same match |
| `extras_penalty` | Rare event | Sparse; use as rare-event count only if enough coverage exists. | All | High if from same match |
| `review_by` | DRS context | Optional. Create historical review frequency only if non-null coverage is meaningful. | Live or advanced analysis | High if from same match |
| `review_batter` | DRS context | Optional and sparse. Usually drop for baseline model. | None | High |
| `review_decision` | DRS context | Optional; can support officiating analysis, not a baseline winner model feature. | None | High |
| `review_type` | DRS context | Optional; usually drop in baseline. | None | High |
| `replacement_role` | Substitution context | Optional. Count concussion or replacement events historically if enough data exists. | Advanced analysis | High if from same match |
| `replacement_team` | Substitution context | Optional; use only in advanced player availability work. | Advanced analysis | High if from same match |
| `replacement_in` | Player substitution | Optional and sparse. Use only in advanced player availability analysis. | Advanced analysis | High if from same match |
| `replacement_out` | Player substitution | Optional and sparse. Use only in advanced player availability analysis. | Advanced analysis | High if from same match |

### wickets.csv

| Column | Role | Feature Engineering Plan | Use Stage | Leakage Risk |
| --- | --- | --- | --- | --- |
| `match_id` | Join key | Join to matches and deliveries for wicket event timing. | All | None |
| `innings_number` | Innings context | Separate wicket behavior in first innings and chase. | All | Safe if historical |
| `over` | Timing input | Derive wicket timing by phase and collapse windows. | All | High if from same match |
| `ball_in_over` | Timing input | Create exact wicket-ball index inside innings. | All | High if from same match |
| `batting_team` | Team identifier | Build batting collapse tendency and wickets-lost-in-phase features. | All | Safe if historical |
| `player_out` | Player identifier | Optional player dismissal tendency if player-level lineup features are added. | All | Safe if historical |
| `kind` | Dismissal type | Encode dismissal mix such as caught, bowled, run out, lbw; useful for batting vulnerability and bowling style summaries. | All | Safe if historical |
| `fielders` | Fielding context | Split by `|` to count fielders involved; optional fielding-activity feature. | All | Safe if historical |

### powerplays.csv

| Column | Role | Feature Engineering Plan | Use Stage | Leakage Risk |
| --- | --- | --- | --- | --- |
| `match_id` | Join key | Join to deliveries to identify official powerplay windows. | All | None |
| `innings_number` | Innings context | Separate first-innings and second-innings powerplay behavior. | All | Safe if historical |
| `powerplay_type` | Phase label | Usually `mandatory` in T20; keep as phase descriptor if multiple powerplay types appear in other formats. | All | Low |
| `from_over` | Phase boundary | Convert to start ball index if needed; mostly used to define phase windows. | All | None |
| `to_over` | Phase boundary | Convert to end ball index if needed; mostly used to define phase windows. | All | None |

### player_of_match.csv

| Column | Role | Feature Engineering Plan | Use Stage | Leakage Risk |
| --- | --- | --- | --- | --- |
| `match_id` | Join key | Join historically if computing prior player awards. | All | None |
| `player` | Player identifier | Build historical player-of-the-match counts, recent impact counts, team-level award concentration. Only useful if squad or playing XI data is available. | Pre-match, Post-toss | High if from same match |

### officials.csv

| Column | Role | Feature Engineering Plan | Use Stage | Leakage Risk |
| --- | --- | --- | --- | --- |
| `match_id` | Join key | Join and pivot if modeling officiating context. | All | None |
| `official_role` | Official type | Pivot to umpire, tv umpire, match referee, reserve umpire groups. | Pre-match, Post-toss | Low |
| `official_name` | Official identifier | Optional low-priority feature. Can encode umpire pair frequency or venue-official combinations if enough support exists. | Pre-match, Post-toss | Low |

## Derived Feature Set For Baseline Model

Recommended first-pass engineered features:

| Feature | Description |
| --- | --- |
| `team_1_win_rate_last_5` | Team 1 match win rate in previous 5 matches |
| `team_2_win_rate_last_5` | Team 2 match win rate in previous 5 matches |
| `win_rate_diff_last_5` | Team 1 minus Team 2 recent win rate |
| `team_1_avg_runs_last_5` | Team 1 average runs scored in previous 5 innings |
| `team_2_avg_runs_last_5` | Team 2 average runs scored in previous 5 innings |
| `avg_runs_diff_last_5` | Team 1 minus Team 2 recent scoring average |
| `team_1_avg_runs_conceded_last_5` | Team 1 average runs conceded in previous 5 innings |
| `team_2_avg_runs_conceded_last_5` | Team 2 average runs conceded in previous 5 innings |
| `team_1_powerplay_run_rate_last_5` | Team 1 average powerplay scoring rate in previous 5 matches |
| `team_2_powerplay_run_rate_last_5` | Team 2 average powerplay scoring rate in previous 5 matches |
| `team_1_death_run_rate_last_5` | Team 1 death-overs scoring rate in previous 5 matches |
| `team_2_death_run_rate_last_5` | Team 2 death-overs scoring rate in previous 5 matches |
| `team_1_dot_ball_pct_last_5` | Team 1 batting dot-ball percentage in previous 5 matches |
| `team_2_dot_ball_pct_last_5` | Team 2 batting dot-ball percentage in previous 5 matches |
| `team_1_wides_conceded_rate_last_5` | Team 1 bowling discipline proxy |
| `team_2_wides_conceded_rate_last_5` | Team 2 bowling discipline proxy |
| `team_1_no_ball_rate_last_5` | Team 1 no-ball rate conceded |
| `team_2_no_ball_rate_last_5` | Team 2 no-ball rate conceded |
| `head_to_head_team_1_win_rate` | Team 1 win rate against Team 2 before match date |
| `team_1_venue_win_rate` | Team 1 historical win rate at the venue |
| `team_2_venue_win_rate` | Team 2 historical win rate at the venue |
| `venue_avg_first_innings_score` | Historical first innings average at the venue |
| `venue_chase_success_rate` | Historical chasing success rate at the venue |
| `toss_winner_is_team_1` | Binary indicator for toss outcome |
| `toss_decision_bat` | Binary toss decision feature |
| `team_1_days_since_last_match` | Team 1 freshness proxy |
| `team_2_days_since_last_match` | Team 2 freshness proxy |

## Implementation Order

1. Standardize team and venue names across all files.
2. Build a canonical match table with `team_1`, `team_2`, `match_date`, and target `team_1_win`.
3. Create historical team innings aggregates from `innings.csv`.
4. Create phase-based historical features from `deliveries.csv` and `powerplays.csv`.
5. Create wicket-style features from `wickets.csv`.
6. Join everything back to the match table using rolling windows that exclude the current match.
7. Train the baseline model on the resulting match-level feature set.

## What To Avoid

- Do not use current-match `winner`, `result_type`, `result_margin`, or `result_text` as predictors.
- Do not use current-match innings totals in a pre-match or toss-time model.
- Do not use current-match ball-by-ball fields in a pre-match or toss-time model.
- Do not mix future matches into rolling features.
- Do not leave team names unstandardized before joins.

## Next Build Artifact

The next file to create should be a feature builder script that:

1. Reads all CSVs.
2. Creates the canonical match table.
3. Builds rolling team and venue aggregates.
4. Produces a training dataset for winner prediction.
