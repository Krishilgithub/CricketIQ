# Outcome Accuracy Improvement Brainstorm (Based on All Current CSVs)

## 1) What Was Reviewed

All CSV files in `data/cricsheet_csv` were reviewed:

- `matches.csv` (5071 rows)
- `match_teams.csv` (10142 rows)
- `innings.csv` (10153 rows)
- `deliveries.csv` (1147474 rows)
- `wickets.csv` (63793 rows)
- `powerplays.csv` (10064 rows)
- `player_of_match.csv` (4593 rows)
- `officials.csv` (19883 rows)

## 2) Key Data Findings That Directly Affect Outcome Quality

### 2.1 Matches-level findings

- `match_type` is consistently `T20`, but `overs` has 15 rows with value `50`.
- `winner` is missing for 150 matches (likely no-result/tie/abandoned).
- `city` is missing for 179 matches.   (drop column)
- `event_name` is missing for 66 matches.
- `method` is present only in 169 rows (mostly D/L or awarded conditions, expected sparsity).

### 2.2 Deliveries-level findings

- Data is rich and complete for core ball-by-ball fields.
- `innings_number > 2` appears in 486 rows (super overs and edge cases).
- `ball_in_over > 6` appears in 57559 rows (valid for wides/no-balls but needs careful feature logic).
- Review fields are sparse (1776 rows with review info).
- Replacement fields are almost empty in parsed columns (`replacement_team/in/out` mostly blank) while `replacement_role` sometimes carries serialized object text.

### 2.3 Wickets-level findings

- `fielders` is missing in 23530 rows, which is expected for some dismissal types but should be dismissal-aware.
- Dismissal mix is dominated by `caught`, `bowled`, `run out`, `lbw`, `stumped`.

### 2.4 Data shape implications

- You have enough granularity for state-based and phase-based match outcome models.
- Biggest gains will come from better feature design and strict handling of special cases (D/L, no result, super overs, extra deliveries).

## 3) Outcome Dependency Factors To Include

## 3.1 Pre-match factors (must include)

1. Team recent form: last 5, last 10 weighted performance.
2. Batting strength index: top-7 weighted strike-rate and boundary rate.
3. Bowling strength index: wicket rate, economy, death-over control.
4. Venue-adjusted strength: team performance at venue and similar venues.
5. Head-to-head trend: recency-weighted matchup advantage.
6. Toss probability impact by venue and chase/defend bias.
7. Match context: tournament stage, pressure level, knockout indicator.
8. Gender/competition segment effects (if mixed datasets are used together).
9. Travel/rest gap and schedule congestion (if added from fixtures).
10. Team composition balance: pace-spin split, all-rounder depth.

## 3.2 In-match dynamic factors (high impact)

1. Current score, wickets, overs, required run rate.
2. Phase momentum: powerplay/middle/death run-rate delta.
3. Resource state proxy: wickets-in-hand x balls-remaining.
4. Boundary pressure indicators: dot-ball streaks, boundary bursts.
5. Bowler matchup context: current batter vs bowler historical profile.
6. Extras pressure: wides/no-balls concentration by phase.
7. Wicket type pattern: pace/spin dismissal tendencies.
8. Review events and overturn tendencies where present.
9. Super-over likelihood signals in close chases.

## 3.3 External/contextual factors (add for major uplift)

1. Weather: rain probability, humidity, wind.
2. Pitch proxy: historical average first innings score by venue and month.
3. Dew likelihood for night games.
4. Playing XI confirmed lineup quality delta (vs expected XI).
5. Injury/replacement late updates.
6. ICC rankings / ELO rating trajectories.

## 4) High-Impact Ways To Improve Prediction Accuracy

## 4.1 Data quality and consistency improvements

1. Enforce match filter: keep only valid T20 with `overs = 20` for main model; treat 50-over anomalies separately.
2. Separate target classes explicitly: `win`, `loss`, `tie`, `no_result`.
3. Add dismissal-aware null logic for `fielders` to avoid false missingness penalties.
4. Correct replacement parsing in converter to capture `replacement_team`, `replacement_in`, `replacement_out` from nested JSON.
5. Create canonical team and venue mapping tables to remove naming drift.
6. Add strict primary keys and uniqueness checks:
   - deliveries key: `(match_id, innings_number, over, ball_in_over, batter, bowler)`
   - wickets key: `(match_id, innings_number, over, ball_in_over, player_out, kind)`
7. Add integrity checks:
   - innings total consistency vs deliveries aggregate.
   - wickets count consistency between innings and wickets table.

## 4.2 Feature engineering improvements

1. Use recency-weighted rolling features instead of static aggregates.
2. Build phase-specific features (PP 1-6, middle 7-15, death 16-20).
3. Build venue-season interaction features (venue x month x innings).
4. Add context-normalized metrics (relative to tournament average in same season).
5. Build clutch indicators (performance in high pressure windows).
6. Add uncertainty features (sample size, variance, confidence of estimates).
7. Encode no-ball and wide pressure as quality signals for bowling units.
8. Include super-over historical tendency as separate signal.

## 4.3 Modeling strategy improvements

1. Train separate models for:
   - pre-match outcome
   - innings-break outcome
   - live ball-by-ball win probability
2. Use calibrated probabilistic models (Brier and calibration curve optimization).
3. Use time-aware splits (no random split) to avoid leakage.
4. Use grouped CV by series/tournament to improve generalization.
5. Stack models: linear baseline + gradient boosting + calibrated meta model.
6. Add class weighting or focal objective for rare outcomes (tie/no-result).
7. Track per-segment metrics (venue type, associate teams, close games).

## 4.4 Target and label engineering improvements

1. For pre-match model, exclude matches with unknown/invalid outcome labels.
2. For live model, generate state snapshots every ball and label with final result.
3. Build secondary labels:
   - win margin bucket
   - chase success probability by over
   - collapse risk (next 12 balls)

## 4.5 MLOps and retraining improvements

1. Trigger retraining on both time and drift thresholds.
2. Maintain champion/challenger with promotion gates:
   - Log Loss improvement
   - Brier non-regression
   - calibration error threshold
3. Monitor feature drift separately for pre-match and live features.
4. Add data freshness SLAs for live inference reliability.
5. Keep rollback-ready previous model versions.

## 5) Suggested Prioritization (Impact vs Effort)

### Tier 1 (Do immediately)

1. Clean target classes and filter inconsistent matches (`overs != 20`).
2. Build recency + phase features from deliveries.
3. Use time-based evaluation and probability calibration.
4. Create automated data integrity checks (innings-deliveries-wickets consistency).

### Tier 2 (Next sprint)

1. Add venue behavior and matchup interaction features.
2. Implement pre-match and live models separately.
3. Improve replacement/review parsing and enrich sparse special-event features.
4. Add segment-wise monitoring dashboard.

### Tier 3 (Advanced uplift)

1. Integrate weather and pitch/dew context.
2. Add confirmed playing XI and injury updates from live sources.
3. Add ratings/ELO and tactical matchup graph features.

## 6) Practical Checklist To Improve Outcomes Fast

- [ ] Create a clean modeling table for pre-match with one row per match.
- [ ] Create a live state table with one row per ball state snapshot.
- [ ] Add null-safe, dismissal-aware and extras-aware preprocessing.
- [ ] Train baseline + boosted model + calibrated variant.
- [ ] Compare on Log Loss, ROC-AUC, Brier, calibration error.
- [ ] Deploy only if challenger beats champion with stability checks.

## 7) Extra Notes For Your Current Dataset

1. `over` and `ball_in_over` should not be interpreted as legal-ball count directly; wides/no-balls create extra deliveries.
2. `innings_number > 2` should be handled explicitly as super over context.
3. Sparse review/replacement fields should be treated as optional event features, not core features.
4. `winner` missing rows should not be dropped blindly; map to explicit `no_result/tie` classes where possible.

---

This document is intentionally focused on maximizing model outcome quality using your exact current CSV landscape, while keeping the path practical for hackathon delivery and scalable for production.
