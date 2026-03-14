# Production-Level Project Structure

## Goals

1. Clear separation between raw and processed data.
2. Separation of production code from utility scripts.
3. Predictable locations for generated artifacts.
4. Scalable module layout for model lifecycle.

## Structure

```text
KenexAI/
  configs/
    feature_engineering.json
  data/
    raw/
      t20s_json/
      cricsheet_csv_all/
      given_data_csv/
    processed/
      cricsheet_csv_men/
  artifacts/
    features/
    models/
    reports/
  docs/
    planning/
      project_plan.md
    research/
      accuracy_improvement_brainstorm.md
      dataset_comparison.md
      feature_engineering_notes.md
    PROJECT_STRUCTURE.md
    PROJECT_STRUCTURE_PRODUCTION.md
    FEATURE_ENGINEERING_COMPLETE.md
  scripts/
    run_feature_engineering.py
    data/
      convert_cricsheet_json_to_csv.py
      filter_mens_dataset.py
    dev/
      profile_cricsheet_csv.py
      export_excalidraw_from_mcp.py
  src/
    cricketiq/
      feature_engineering/
      ingestion/
      validation/
      training/
      inference/
      monitoring/
      utils/
  diagrams/
  tests/
```

## Why this is production-ready

1. Data zones prevent accidental overwrite of source data.
2. Package-first layout in `src/` supports deployment and testing.
3. Config-driven execution supports environment portability.
4. Artifacts are isolated from source for reproducible runs.
5. Documentation is split by purpose (planning vs research vs implementation).
