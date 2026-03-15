# Dataset Comparison: Given Hackathon Data vs Found Cricsheet Data

## Goal

This document compares the two datasets in `data/`:

- `data/given_data_csv/`: hackathon-provided curated tournament dataset
- `data/cricsheet_csv/`: found detailed historical ball-by-ball dataset

The comparison answers four questions:

1. What is missing in the found Cricsheet dataset relative to the given dataset?
2. What required information in the given dataset can be created from the found dataset?
3. What is present in the given dataset but not really required if you already have the found dataset?
4. What extra information exists in the found dataset that can improve model performance beyond the given dataset?

## High-Level Conclusion

The two datasets serve different purposes:

- The given dataset is a curated tournament summary dataset. It is compact, presentation-ready, and directly useful for reporting and hackathon dashboards.
- The found Cricsheet dataset is a raw event-level historical dataset. It is much richer for feature engineering and predictive modeling.

Short version:

- The given dataset has cleaner tournament-specific business fields like `stage`, `group`, `qualified`, venue capacity, squads, and tournament summary fields.
- The found dataset has much stronger modeling depth because it includes innings-level, wicket-level, powerplay-level, and ball-by-ball data.
- Many given summary tables can be derived from the found dataset, but some tournament metadata cannot.
- For model performance, the found dataset is more valuable than the given dataset.

## File-Level Comparison

| Given File | Purpose | Direct Match in Cricsheet | Can Be Derived from Cricsheet | Missing from Cricsheet |
| --- | --- | --- | --- | --- |
| `matches.csv` | Tournament match schedule and results | Partial via `cricsheet_csv/matches.csv` and `match_teams.csv` | Yes, mostly | `stage`, `group`, given-style `match_no` |
| `batting_stats.csv` | Tournament batting aggregates | No direct table | Yes, from `deliveries.csv` and wickets context | None if player innings history is available and filtered correctly |
| `bowling_stats.csv` | Tournament bowling aggregates | No direct table | Yes, from `deliveries.csv` and `wickets.csv` | Best-figures text formatting not direct |
| `key_scorecards.csv` | Player innings scorecards | No direct table | Yes, from `deliveries.csv` and `wickets.csv` | Given-style match labels like `Final` or `1st` innings text |
| `squads.csv` | Team squad and roles | No direct equivalent | No, not reliably | `role`, `designation`, explicit squad lists |
| `points_table.csv` | Group standings | No direct table | Partially, if tournament grouping is separately known | `group`, `qualified` not directly available |
| `venues.csv` | Venue metadata | Partial via `matches.csv` venue and city | No direct venue metadata | `country`, `capacity`, `stages_hosted` |
| `awards.csv` | Tournament awards | Partial via `player_of_match.csv` | Partially | Tournament-level awards like `Most Runs`, `Most Wickets`, `Player of the Tournament` are not directly stored |
| `tournament_summary.csv` | Tournament metadata | No direct table | Mostly no | Tournament edition, host nation, team count, final winner metadata unless externally curated |

## What Is Missing In The Found Dataset

These are the important gaps where the given dataset contains information that the found Cricsheet dataset does not directly provide.

### Missing Tournament Structure Metadata

Cricsheet `matches.csv` does not directly provide:

- `stage`
- `group`
- `match_no` in the curated tournament numbering sense
- qualification status
- explicit final, semi-final, super-eight, group-stage labels

Impact:

- You cannot directly reproduce tournament standings or group tables without extra tournament-structure metadata.
- Stage pressure features are not available unless manually engineered from an external tournament schedule.

### Missing Squad Metadata

Cricsheet does not provide a squad master table equivalent to `given_data_csv/squads.csv`.

Missing fields:

- team squad list
- `role`
- `designation`

Impact:

- You cannot directly build lineup balance features such as batter count, bowler count, all-rounder balance, or captain-presence features unless you source squad data separately.

### Missing Venue Master Data

Cricsheet has venue names and city, but not venue metadata such as:

- `country`
- `capacity`
- `stages_hosted`

Impact:

- You can compute venue behavior from history, but not venue business metadata.

### Missing Tournament Summary Table

Cricsheet does not provide a `tournament_summary.csv` equivalent with curated fields like:

- tournament name
- edition
- host country
- number of teams
- final winner summary

Impact:

- This is not a modeling problem, but it is useful for dashboards and narrative reporting.

### Missing Ready-Made Aggregated Tables

Cricsheet does not directly include:

- batting aggregates by player
- bowling aggregates by player
- scorecards by innings
- points table

Impact:

- These are not true information gaps because most can be engineered, but they do require work.

## What Can Be Achieved From The Found Dataset That Is Required In The Given Dataset

This is the strongest part of the comparison. Many important given tables can be rebuilt from the found dataset.

### 1. Rebuilding Given `matches.csv`

Given `matches.csv` columns:

| Given Column | Available in Cricsheet | How to Create or Map | Notes |
| --- | --- | --- | --- |
| `match_no` | No direct | Derive only if the tournament schedule is sorted and match numbering is known | Not reliable globally |
| `stage` | No direct | External mapping required from tournament schedule or event guide | Missing |
| `group` | No direct | External mapping required | Missing |
| `date` | Yes | Map from `cricsheet matches.match_date` | Direct |
| `venue` | Yes | Map from `cricsheet matches.venue` | Direct |
| `city` | Yes | Map from `cricsheet matches.city` | Direct, with some nulls |
| `team1` | Yes, indirect | Pivot `match_teams.csv` into two team columns | Derivable |
| `team2` | Yes, indirect | Pivot `match_teams.csv` into two team columns | Derivable |
| `toss_winner` | Yes | Map from `cricsheet matches.toss_winner` | Direct |
| `toss_decision` | Yes | Map from `cricsheet matches.toss_decision` | Direct |
| `winner` | Yes | Map from `cricsheet matches.winner` | Direct |
| `result` | Partial | Use `cricsheet matches.result_text` | Direct text equivalent |
| `margin` | Partial | Build from `result_margin` and `result_type` | Derivable |

Verdict:

- The given `matches.csv` can be mostly recreated from Cricsheet.
- The only real gaps are tournament `stage`, `group`, and curated `match_no`.

### 2. Rebuilding Given `batting_stats.csv`

Given `batting_stats.csv` columns:

| Given Column | Available in Cricsheet | How to Create or Map | Notes |
| --- | --- | --- | --- |
| `player` | Yes | Use `deliveries.batter` plus dismissal joins | Direct |
| `team` | Yes | Use `deliveries.batting_team` | Direct |
| `matches` | Yes | Count distinct `match_id` by batter | Derivable |
| `innings` | Yes | Count batter innings from deliveries | Derivable |
| `runs` | Yes | Sum `runs_batter` | Derivable |
| `average` | Yes | Runs divided by dismissals from wicket joins | Derivable |
| `strike_rate` | Yes | `100 * runs / balls_faced` | Derivable |
| `fours` | Yes | Count balls where `runs_batter = 4` | Derivable |
| `sixes` | Yes | Count balls where `runs_batter = 6` | Derivable |
| `hundreds` | Yes | Count innings with 100 or more runs | Derivable |
| `fifties` | Yes | Count innings with 50 to 99 runs | Derivable |

Verdict:

- This entire file is derivable from Cricsheet.
- Cricsheet is actually better because it supports rolling batting form and phase batting features, not just aggregate tournament totals.

### 3. Rebuilding Given `bowling_stats.csv`

Given `bowling_stats.csv` columns:

| Given Column | Available in Cricsheet | How to Create or Map | Notes |
| --- | --- | --- | --- |
| `player` | Yes | Use `deliveries.bowler` | Direct |
| `team` | Partial | Infer from match and opposition context or separate team mapping | Derivable but needs care |
| `matches` | Yes | Count distinct `match_id` by bowler | Derivable |
| `overs` | Yes | Convert legal balls to overs | Derivable |
| `balls` | Yes | Count legal deliveries bowled | Derivable |
| `wickets` | Yes | Count bowling wickets from `wickets.csv`, excluding non-bowler dismissals where needed | Derivable |
| `average` | Yes | Runs conceded divided by wickets | Derivable |
| `runs_conceded` | Yes | Sum conceded runs from deliveries | Derivable |
| `economy` | Yes | `6 * runs_conceded / legal_balls` | Derivable |
| `four_wicket_hauls` | Yes | Count innings with at least 4 wickets | Derivable |
| `five_wicket_hauls` | Yes | Count innings with at least 5 wickets | Derivable |
| `best_figures` | Partial | Derive numeric best figures, then format text | Derivable with formatting step |

Verdict:

- This file is derivable from Cricsheet.
- Cricsheet additionally enables death-over economy, powerplay wicket rate, and matchup bowling features.

### 4. Rebuilding Given `key_scorecards.csv`

Given `key_scorecards.csv` columns:

| Given Column | Available in Cricsheet | How to Create or Map | Notes |
| --- | --- | --- | --- |
| `match` | No direct | Create from event or match labels manually | Missing as curated label |
| `innings` | Partial | Map from `innings_number` to `1st` or `2nd` | Derivable |
| `team` | Yes | Use innings batting team | Direct |
| `player` | Yes | Use batter | Direct |
| `runs` | Yes | Aggregate `runs_batter` by innings | Derivable |
| `balls` | Yes | Count legal balls faced | Derivable |
| `fours` | Yes | Count 4s | Derivable |
| `sixes` | Yes | Count 6s | Derivable |
| `dismissal` | Partial | Combine `wickets.kind`, `bowler`, and `fielders` into dismissal text | Derivable with formatting |

Verdict:

- The scorecard file can be created from Cricsheet.
- The only missing part is the exact display-style match naming.

### 5. Rebuilding Given `points_table.csv`

Given `points_table.csv` columns:

| Given Column | Available in Cricsheet | How to Create or Map | Notes |
| --- | --- | --- | --- |
| `group` | No | External group mapping required | Missing |
| `team` | Yes | From match teams | Direct |
| `matches_played` | Yes | Count matches by team in tournament subset | Derivable |
| `won` | Yes | Count wins | Derivable |
| `lost` | Yes | Count losses | Derivable |
| `no_result` | Partial | Map from `winner` null or result logic if present | Derivable with rules |
| `net_run_rate` | Yes | Compute from runs scored and overs faced versus conceded | Derivable |
| `points` | Partial | Derive using tournament rules | Derivable if points rules are defined |
| `qualified` | No direct | Infer from ranking and rules or use external truth | Missing as direct label |

Verdict:

- The standings logic is mostly derivable.
- The missing dependency is tournament grouping and official qualification labels.

### 6. Rebuilding Given `awards.csv`

Given `awards.csv` columns:

| Given Column | Available in Cricsheet | How to Create or Map | Notes |
| --- | --- | --- | --- |
| `award` | Partial | Some awards can be generated by ranking stats | Derivable for many but not all |
| `player_or_detail` | Partial | Use player rankings or summary text | Derivable for stat-based awards |
| `team` | Partial | Join winning player or team | Derivable in many cases |

Verdict:

- Stat-based awards like `Most Runs` and `Most Wickets` are derivable from Cricsheet.
- Narrative awards like `Player of the Tournament` require an external rule or curated source.
- `Player of the Match` is directly supported by `player_of_match.csv`.

## What Is In The Given Dataset But Not Required If You Already Have Cricsheet

This section does not mean the fields are useless. It means they are less important for predictive modeling because they are either static metadata, final summaries, or can already be created from Cricsheet.

### Lower-Priority or Redundant Given Files

| Given File | Why It Is Less Critical For Modeling |
| --- | --- |
| `batting_stats.csv` | Fully derivable from Cricsheet and less useful than rolling or phase-based features |
| `bowling_stats.csv` | Fully derivable from Cricsheet and less informative than delivery-derived phase features |
| `key_scorecards.csv` | Presentation-friendly, but Cricsheet can rebuild more detailed innings-level data |
| `awards.csv` | Mostly for reporting and storytelling, not strong direct predictors |
| `tournament_summary.csv` | Useful for dashboard text, not for match winner modeling |
| `venues.csv` metadata columns | Capacity and stages hosted are not primary predictors compared with historical venue behavior |

### Given Columns That Are Usually Not Required For Baseline Prediction

| File | Column | Reason |
| --- | --- | --- |
| `given_data_csv/matches.csv` | `result` | Outcome text, leakage for prediction |
| `given_data_csv/matches.csv` | `margin` | Post-match outcome summary, leakage |
| `given_data_csv/matches.csv` | `winner` | Target, not a feature |
| `given_data_csv/venues.csv` | `capacity` | Weak predictive signal compared with actual venue scoring history |
| `given_data_csv/venues.csv` | `stages_hosted` | Useful for reporting, weak baseline predictor |
| `given_data_csv/awards.csv` | all columns | Mostly post-tournament descriptive output |
| `given_data_csv/tournament_summary.csv` | all columns | Descriptive metadata, not predictive features |

## What The Found Dataset Has That Can Improve Model Performance

This is where Cricsheet is substantially stronger.

### Cricsheet-Only Files That Add Modeling Value

| Cricsheet File | Why It Helps |
| --- | --- |
| `deliveries.csv` | Ball-by-ball events enable phase features, pressure features, player form, and matchup features |
| `innings.csv` | Provides compact innings aggregates for scoring and bowling trend features |
| `wickets.csv` | Enables dismissal-type features, collapse analysis, and wicket timing features |
| `powerplays.csv` | Enables official powerplay phase calculations instead of assuming fixed phase boundaries |
| `officials.csv` | Optional officiating-context features |
| `player_of_match.csv` | Historical impact signal for players and teams |

### Cricsheet-Only Columns That Are Especially Useful

#### From `cricsheet_csv/matches.csv`

| Column | Helpful For Model Performance |
| --- | --- |
| `event_name` | Tournament or series context |
| `event_match_number` | Match sequence pressure proxy |
| `result_type` | Useful for labeling and outcome diagnostics, not as feature |
| `result_margin` | Useful for building team strength targets or post-hoc confidence analysis |
| `method` | Useful to filter abnormal results such as weather-adjusted outcomes |

#### From `cricsheet_csv/innings.csv`

| Column | Helpful For Model Performance |
| --- | --- |
| `total_runs` | Historical team scoring strength |
| `total_wickets` | Collapse tendency and wicket preservation |
| `total_balls` | Scoring speed and chase efficiency |
| `extras_noballs` | Bowling discipline |
| `extras_wides` | Bowling discipline |

#### From `cricsheet_csv/deliveries.csv`

| Column | Helpful For Model Performance |
| --- | --- |
| `over` | Phase segmentation |
| `ball_in_over` | Exact innings progress |
| `batter` | Player batting form |
| `bowler` | Player bowling form and matchup features |
| `non_striker` | Partnership context |
| `runs_batter` | Strike rate, boundary rate, scoring pattern |
| `runs_total` | Team phase run rate |
| `runs_extras` | Opponent discipline |
| `extras_noballs` | Discipline and free-hit proxy |
| `extras_wides` | Discipline and pressure leakage |
| `review_by` | Advanced pressure and DRS usage analysis |
| `replacement_role` | Rare but useful substitution context |

#### From `cricsheet_csv/wickets.csv`

| Column | Helpful For Model Performance |
| --- | --- |
| `kind` | Dismissal mix and batting vulnerability |
| `fielders` | Fielding activity and dismissal composition |
| `over` | Wicket timing features |

#### From `cricsheet_csv/powerplays.csv`

| Column | Helpful For Model Performance |
| --- | --- |
| `from_over` | Accurate phase boundary setup |
| `to_over` | Accurate phase boundary setup |
| `powerplay_type` | Useful when comparing across formats or special rules |

### Extra Features You Can Build Only Because Cricsheet Exists

These are not available from the given dataset alone.

| Feature | Source |
| --- | --- |
| Powerplay run rate | `deliveries.csv` + `powerplays.csv` |
| Death-over scoring rate | `deliveries.csv` |
| Dot-ball percentage | `deliveries.csv` |
| Boundary percentage | `deliveries.csv` |
| Bowling economy by phase | `deliveries.csv` |
| Wicket rate by phase | `deliveries.csv` + `wickets.csv` |
| Collapse index | `wickets.csv` + innings ordering |
| Chase efficiency | `innings.csv` |
| Venue-adjusted scoring index | `innings.csv` + `matches.csv` |
| Head-to-head record | `matches.csv` + `match_teams.csv` |
| Toss impact by venue | `matches.csv` |
| Discipline rates | `deliveries.csv` + `innings.csv` |
| Player recent form | `deliveries.csv` |
| Bowler versus batter matchup history | `deliveries.csv` + `wickets.csv` |

## Column-by-Column Comparison For The Given Dataset

### given_data_csv/matches.csv

| Column | Status vs Cricsheet | Cricsheet Source | Comment |
| --- | --- | --- | --- |
| `match_no` | Missing direct | None | Needs tournament-specific schedule numbering |
| `stage` | Missing direct | None | Not available in found dataset |
| `group` | Missing direct | None | Not available in found dataset |
| `date` | Direct | `matches.match_date` | Same information |
| `venue` | Direct | `matches.venue` | Same information |
| `city` | Direct | `matches.city` | Same information with some missing values |
| `team1` | Derivable | `match_teams.team` | Must pivot 2 rows into 2 columns |
| `team2` | Derivable | `match_teams.team` | Must pivot 2 rows into 2 columns |
| `toss_winner` | Direct | `matches.toss_winner` | Same information |
| `toss_decision` | Direct | `matches.toss_decision` | Same information |
| `winner` | Direct | `matches.winner` | Same information |
| `result` | Derivable | `matches.result_text` | Equivalent narrative field |
| `margin` | Derivable | `matches.result_margin` + `matches.result_type` | Formatting required |

### given_data_csv/batting_stats.csv

| Column | Status vs Cricsheet | Cricsheet Source | Comment |
| --- | --- | --- | --- |
| `player` | Direct | `deliveries.batter` | Same entity |
| `team` | Direct | `deliveries.batting_team` | Same entity |
| `matches` | Derivable | `deliveries.match_id` | Distinct match count |
| `innings` | Derivable | `deliveries.match_id` + innings grouping | Count player innings batted |
| `runs` | Derivable | `deliveries.runs_batter` | Sum |
| `average` | Derivable | `runs_batter` + wickets join | Need dismissal count |
| `strike_rate` | Derivable | `runs_batter` + legal balls faced | Formula |
| `fours` | Derivable | `runs_batter` | Count value 4 |
| `sixes` | Derivable | `runs_batter` | Count value 6 |
| `hundreds` | Derivable | innings-level batter aggregates | Count innings >= 100 |
| `fifties` | Derivable | innings-level batter aggregates | Count innings between 50 and 99 |

### given_data_csv/bowling_stats.csv

| Column | Status vs Cricsheet | Cricsheet Source | Comment |
| --- | --- | --- | --- |
| `player` | Direct | `deliveries.bowler` | Same entity |
| `team` | Derivable | infer from match/team context | Not a direct column |
| `matches` | Derivable | `deliveries.match_id` | Distinct match count |
| `overs` | Derivable | legal balls in `deliveries` | Convert balls to overs |
| `balls` | Derivable | legal deliveries in `deliveries` | Count legal balls |
| `wickets` | Derivable | `wickets` + `deliveries.bowler` | Filter dismissal kinds carefully |
| `average` | Derivable | wickets + runs conceded | Formula |
| `runs_conceded` | Derivable | `runs_total` excluding byes/leg-byes where needed | Formula choice must be consistent |
| `economy` | Derivable | `runs_conceded` and legal balls | Formula |
| `four_wicket_hauls` | Derivable | bowler innings aggregates | Count innings >= 4 wickets |
| `five_wicket_hauls` | Derivable | bowler innings aggregates | Count innings >= 5 wickets |
| `best_figures` | Derivable | bowler innings aggregates | Needs text formatting |

### given_data_csv/key_scorecards.csv

| Column | Status vs Cricsheet | Cricsheet Source | Comment |
| --- | --- | --- | --- |
| `match` | Missing direct | None | Curated label like `Final` not stored directly |
| `innings` | Derivable | `innings_number` | Convert to `1st` or `2nd` |
| `team` | Direct | `deliveries.batting_team` or `innings.team` | Same entity |
| `player` | Direct | `deliveries.batter` | Same entity |
| `runs` | Derivable | `runs_batter` | Sum per batter innings |
| `balls` | Derivable | legal balls faced | Count per batter innings |
| `fours` | Derivable | `runs_batter` | Count value 4 |
| `sixes` | Derivable | `runs_batter` | Count value 6 |
| `dismissal` | Derivable | `wickets.kind`, `fielders`, `deliveries.bowler` | Needs formatting logic |

### given_data_csv/squads.csv

| Column | Status vs Cricsheet | Cricsheet Source | Comment |
| --- | --- | --- | --- |
| `team` | Missing as squad master | None | Match team exists, squad list does not |
| `player_name` | Missing as squad master | Players appear in deliveries and wickets | No official squad roster |
| `role` | Missing | None | Not present in found dataset |
| `designation` | Missing | None | Not present in found dataset |

### given_data_csv/points_table.csv

| Column | Status vs Cricsheet | Cricsheet Source | Comment |
| --- | --- | --- | --- |
| `group` | Missing direct | None | Tournament structure not stored |
| `team` | Direct | `match_teams.team` | Same entity |
| `matches_played` | Derivable | `match_teams` | Count matches per team |
| `won` | Derivable | `matches.winner` | Count wins |
| `lost` | Derivable | `match_teams` + winner logic | Count losses |
| `no_result` | Derivable | `matches.winner`, `method`, result handling | Rules needed |
| `net_run_rate` | Derivable | `innings.csv` | Can compute |
| `points` | Derivable | wins/losses/no result + rules | Need points rules |
| `qualified` | Missing direct | None | Requires final standings logic or external label |

### given_data_csv/venues.csv

| Column | Status vs Cricsheet | Cricsheet Source | Comment |
| --- | --- | --- | --- |
| `venue_name` | Direct | `matches.venue` | Equivalent |
| `city` | Direct | `matches.city` | Equivalent |
| `country` | Missing | None | Not stored directly |
| `capacity` | Missing | None | Not stored directly |
| `stages_hosted` | Missing | None | Not stored directly |

### given_data_csv/awards.csv

| Column | Status vs Cricsheet | Cricsheet Source | Comment |
| --- | --- | --- | --- |
| `award` | Partial | generated from stats or `player_of_match` | Not a direct field |
| `player_or_detail` | Partial | generated from stats or player names | Not direct for narrative awards |
| `team` | Partial | derive from player-team mapping or award context | Not direct in all cases |

### given_data_csv/tournament_summary.csv

| Column | Status vs Cricsheet | Cricsheet Source | Comment |
| --- | --- | --- | --- |
| `field` | Missing | None | Curated metadata structure |
| `value` | Missing | None | Curated metadata structure |

## Recommended Use Strategy

Use both datasets together, but for different jobs.

### Use The Given Dataset For

- hackathon dashboards
- tournament narrative summaries
- squad information
- venue metadata
- stage and group context
- quick presentation tables

### Use The Found Cricsheet Dataset For

- feature engineering
- historical training data
- rolling form features
- venue behavior features
- batting and bowling phase features
- richer model training and better generalization

## Best Combined Approach

The strongest solution is not to choose one dataset over the other.

Build the final pipeline like this:

1. Use `given_data_csv/matches.csv` to anchor the 2026 tournament structure, stage, and group labels.
2. Use `cricsheet_csv/matches.csv`, `innings.csv`, `deliveries.csv`, `wickets.csv`, and `powerplays.csv` to generate historical team and player features.
3. Use `given_data_csv/squads.csv` to add lineup composition and role-balance features.
4. Use `given_data_csv/venues.csv` to enrich venue metadata.
5. Train the model on Cricsheet-derived historical features and score the given 2026 tournament matches.

## plan.md Coverage Check

This section compares the requirements in [plan.md](plan.md) with the columns available across both datasets.

### Overall Coverage Summary

| Planned Output From plan.md | Coverage Status | Why |
| --- | --- | --- |
| Pre-match winner prediction | Full | Historical match, innings, venue, toss-history, player, and team context are available across both datasets |
| Post-toss winner prediction | Full | Toss columns exist directly in both match datasets |
| Live in-match win probability | Partial | Historical event-level features exist in Cricsheet, but live 2026 streaming columns or feed are not present yet |
| Reproducible feature tables | Full | Both datasets are CSV-based and enough columns exist to build bronze, silver, and gold layers |
| Data quality checks | Full | Enough schema, type, relationship, and completeness columns exist for validation |
| Persona dashboards | Full to Partial | Most KPIs are supported; playing-XI and stage-specific insights need squad and stage context |
| MLOps workflow | Full for offline pipeline | Dataset supports training and evaluation, but monitoring and registry need infrastructure, not columns |
| Drift and performance monitoring | Partial | Historical scoring is possible, but live production signals depend on deployed inference logs |
| Dockerized end-to-end deployment | Not a data issue | This depends on engineering setup, not dataset columns |

### Phase-by-Phase Coverage Against plan.md

#### Phase 1: Data Acquisition and Simulation

| Requirement | Coverage Status | Available Columns / Tables | Gap |
| --- | --- | --- | --- |
| Historical base data | Full | All files in both folders | None |
| Near real-time toss, playing XI, innings progression, score snapshots | Partial | Toss exists in match data; innings progression can be simulated from `deliveries.csv` | No real live feed and no official playing XI table in Cricsheet |
| Simulated live stream | Full | `deliveries.csv`, `innings.csv`, `matches.csv` support replay simulation | Needs code, not new columns |

#### Phase 2: Data Warehouse Design

| Gold Object In plan.md | Coverage Status | Supporting Columns | Gap |
| --- | --- | --- | --- |
| `fact_matches` | Full | Match result, toss, venue, date, teams from `matches.csv` and `match_teams.csv` | None |
| `fact_innings_snapshots` | Full | `deliveries.csv`, `innings.csv`, `powerplays.csv`, `wickets.csv` | None |
| `fact_player_match_performance` | Full | Batter, bowler, runs, wickets, balls, dismissals from Cricsheet | None |
| `dim_team` | Full | Team names from `match_teams.csv`, `given_data_csv/matches.csv`, `points_table.csv`, `squads.csv` | Team standardization needed |
| `dim_player` | Partial | Player names from deliveries, wickets, scorecards, squads | No universal player master ID |
| `dim_venue` | Full | Venue and city from Cricsheet, plus country/capacity/stages from given venues | None after merge |
| `dim_date` | Full | `match_date` and given `date` | None |
| `dim_tournament_stage` | Partial | `stage`, `group` from given matches only | Historical Cricsheet stage labels missing |

#### Phase 3: ETL + Data Quality + Profiling

| Requirement | Coverage Status | Supporting Columns | Gap |
| --- | --- | --- | --- |
| Missing value strategy | Full | Null-prone fields exist and can be profiled | None |
| Type standardization | Full | Dates, numeric metrics, categories are available | None |
| Outlier checks for margin, strike rate, economy | Full | Margin from matches; strike rate and economy derivable from Cricsheet | None |
| Feature-safe joins across match, team, player entities | Full | `match_id`, team names, player names | Team and player standardization needed |
| Schema and range validations | Full | All CSVs have structured columns | None |

#### Phase 4: EDA and Data Quality Dashboard

| Dashboard Output In plan.md | Coverage Status | Supporting Columns | Gap |
| --- | --- | --- | --- |
| Team performance trends by stage and venue | Partial | Venue is available; stage from given matches only | Historical stage labels missing in Cricsheet |
| Toss impact on win probability | Full | Toss winner, toss decision, winner | None |
| Batting-bowling matchup patterns | Full | `batter`, `bowler`, runs, wickets, dismissals | None |
| Venue behavior | Full | Venue, innings totals, match winner, toss data | None |
| Data quality status panel | Full | All tables can be validated | None |

#### Phase 5: Persona and KPI Framework

| KPI In plan.md | Coverage Status | Supporting Columns | Gap |
| --- | --- | --- | --- |
| Pre-match win probability | Full | Match history, team history, venue history, toss-history patterns | None |
| Best playing XI confidence score | Partial | `squads.csv` provides roles, but actual playing XI is not directly available historically in both datasets | Need lineup source or assumption logic |
| Venue-adjusted expected score | Full | Venue, innings totals, phase scoring from Cricsheet | None |
| Powerplay performance index | Full | `deliveries.csv`, `powerplays.csv`, `innings.csv` | None |
| Death overs risk index | Full | Phase scoring and wickets from deliveries and wickets | None |
| Toss decision recommendation confidence | Full | Toss, venue chase bias, team batting-first and chase history | None |
| Player consistency index | Full | Batting and bowling historical aggregates can be computed from Cricsheet | None |
| Form momentum score | Full | Rolling match and player windows from Cricsheet | None |
| Opposition weakness heatmap | Full | Team-vs-team and batter-vs-bowler event history | None |
| Match excitement index | Full | Ball-by-ball scoring swings, close finish margins, wicket clusters | None |
| Key player impact tracker | Full | Player-level batting, bowling, wickets, player-of-match | None |

#### Phase 6: ML Use Cases and Modeling

| Planned Feature In plan.md | Coverage Status | Supporting Columns | Gap |
| --- | --- | --- | --- |
| Team form last N matches | Full | Match winner, innings performance by team | None |
| Venue-adjusted batting and bowling strength | Full | Venue, innings totals, deliveries, wickets | None |
| Toss and toss decision effect | Full | `toss_winner`, `toss_decision`, `winner` | None |
| Head-to-head record | Full | Team pairs across matches | None |
| Squad role balance | Partial | `given_data_csv/squads.csv` has roles | Historical playing XI missing, only squad list available |
| Pressure or stage context | Partial | `stage`, `group`, `event_match_number` | Historical stage labels missing in Cricsheet |
| Expected total prediction | Full | Innings totals, venue, batting and bowling context | None |
| Player impact clustering | Full | Player-level batting and bowling stats can be derived | None |
| Association rules for winning patterns | Full | Match outcomes, toss, venue, team features | None |

### Output-Level Column Sufficiency

This is the direct answer to whether the current columns are enough to produce the outputs in the plan.

#### Outputs We Can Produce Immediately

| Output | Status | Main Supporting Columns |
| --- | --- | --- |
| Match winner training table | Yes | `match_id`, `match_date`, `team`, `venue`, `city`, `toss_winner`, `toss_decision`, `winner` |
| Team form features | Yes | `winner`, `team`, `match_date`, innings totals |
| Venue behavior features | Yes | `venue`, `city`, `total_runs`, `winner`, `toss_decision` |
| Powerplay and death-overs features | Yes | `over`, `ball_in_over`, `runs_total`, `runs_batter`, `powerplay_type`, wicket events |
| Player batting and bowling form | Yes | `batter`, `bowler`, `runs_batter`, `runs_total`, wicket joins |
| Toss recommendation logic | Yes | Toss data plus venue and chase history |
| Scorecards and batting or bowling aggregates | Yes | Cricsheet deliveries and wickets |

#### Outputs We Can Produce, But Only Partially

| Output | Status | Why Partial |
| --- | --- | --- |
| Stage-aware dashboards | Partial | Only the given 2026 dataset has explicit `stage` and `group` |
| Group standings and qualification views | Partial | Need tournament group structure and qualification rules |
| Playing XI confidence score | Partial | We have squad roles, but not full playing XI history or final XI data |
| Live in-match scoring | Partial | Replay simulation is possible, but real-time live feed is not attached |
| Historical tournament-stage pressure features | Partial | Cricsheet historical data lacks explicit stage labels |

#### Outputs That Need External Data Or Manual Rules

| Output | Missing Dependency |
| --- | --- |
| Official playing XI based model | Playing XI source or lineup announcements |
| True stage-pressure historical features | Historical tournament stage mapping |
| Qualification probability engine | Tournament rules plus live standings updates |
| Real live API ingestion | External feed or API |

### Final Coverage Verdict Against plan.md

The current columns are enough to deliver the core predictive outputs in [plan.md](plan.md):

- match outcome prediction
- expected score prediction
- player and team form features
- venue-adjusted and phase-based features
- toss and matchup analytics
- most persona KPIs

The current columns are not enough by themselves for these advanced or context-heavy outputs:

- explicit stage-aware historical modeling
- playing XI optimization with confidence
- official qualification logic and group-based progression analytics
- real live production ingestion without an external source

So the plan is feasible with the current datasets, but four areas remain partial: stage context, group logic, playing XI certainty, and real-time live feed integration.

## Final Answer In One Sentence

The given dataset is better for tournament structure and reporting, but the found Cricsheet dataset is far better for feature engineering and model performance; the best solution is to use the given dataset for context and the found dataset for predictive features.
