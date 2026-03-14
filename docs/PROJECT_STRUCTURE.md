# Project Structure

```text
KenexAI/
  configs/
    feature_engineering.json
  data/
    cricsheet_csv/
    given_data_csv/
  artifacts/
    features/
    models/
    reports/
  docs/
    PROJECT_STRUCTURE.md
  diagrams/
  scripts/
    run_feature_engineering.py
    convert_cricsheet_json_to_csv.py
    profile_cricsheet_csv.py
    export_excalidraw_from_mcp.py
  src/
    cricketiq/
      __init__.py
      feature_engineering/
        __init__.py
        pipeline.py
      ingestion/
        __init__.py
      validation/
        __init__.py
      training/
        __init__.py
      inference/
        __init__.py
      monitoring/
        __init__.py
      utils/
        README.md
  tests/
    feature_engineering/
```

## Current Implemented Module

- `src/cricketiq/feature_engineering/pipeline.py`
  - canonical match building from `matches.csv` + `match_teams.csv`
  - delivery and innings aggregations
  - rolling team form features
  - head-to-head and venue historical features
  - match-level and match-team feature table export

## Current Artifacts

- `artifacts/features/match_level_features.csv`
- `artifacts/features/match_team_features.csv`
- `artifacts/features/feature_engineering_summary.json`
