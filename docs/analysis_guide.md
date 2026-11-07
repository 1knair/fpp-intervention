# Analysis inputs / outputs

## Setup

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

config/settings.csv is included with project-relative defaults; place private study data at those paths or edit settings.csv

Study data and generated outputs stay out of Git

## Config

| CSV | Job |
| --- | --- |
| experiments.csv | enabled experiments, feature set, target set, model/settings |
| feature_sets.csv | feature names in each set, one per line inside the cell |
| target_sets.csv | rating names in each custom set, one per line inside the cell |
| scores.csv | scores (existing) and prediction ranges |
| settings.csv | default project-relative data/output paths and run settings |

Feature names must have extraction code; adding a name to a CSV does not create a new calculation

## Prepared inputs

Keep these under outputs/data/model_inputs unless settings.csv points model_data elsewhere:

| File | Required content |
| --- | --- |
| model_labels.csv | recording_id, participant_id, visit, split, fold, rating cols, score cols |
| drive_mapping.csv | approved drive mapping, including recording_id, participant_id, visit, record_id, scenario_id, label_status and score_field |
| event_anchors.csv | scenario landmarks for barrels/humps, including scenario_id |
| question_labels.csv | assessment-field definitions used by ANOVA |
| model_data.csv | one row per visit: IDs/splits + numeric features + ratings/scores |
| model_data_sources.json | feature/rating names and source info; regression reads its ratings list |

outputs/data/joined_visits/visit_sources.csv links each recording_id to output_path and records the source policy

Those paths point to prepared single-source or joined telemetry CSVs; the extractors use simulator headers such as Sim Time, Velocity, Acceleration, Brake Pedal Force, Signs and Events

For ANOVA, assessment_file in settings.csv points to the private assessment export with record_id, redcap_event_name and the mapped rating/score fields

Real participant mappings, telemetry, ratings and predictions are not included in this repository

## Run

```bat
.venv\Scripts\python.exe run_project.py inputs
.venv\Scripts\python.exe run_project.py regression
.venv\Scripts\python.exe run_project.py anova
```

inputs rebuilds model_data.csv, model_data_sources.json and feature_windows.csv from prepared sources; it keeps matched labels and existing participant splits

regression trains fresh models and writes a new named timestamped folder under outputs/runs:

```text
<run_name>__<timestamp>/
    config/
    experiments.json
    validation/
        metrics.csv
        rating_metrics.csv
        predictions.csv
        experiments.json
    test/
        metrics.csv
        rating_metrics.csv
        predictions.csv
        experiments.json
```

metrics.csv holds custom/existing score results; rating_metrics.csv holds individual rating results; predictions.csv holds actual/predicted/baseline values per visit

Metric cols include experiment, feature_set, target_set, features, target_variables, score, method, model, mae, baseline_mae, result, visits, people, type, target, feature_subset, rmse, baseline_rmse and r2

ANOVA writes its own all_clinician_visit_changes__<timestamp> folder with metrics.csv, clinician_scores.csv, question_fields.csv, settings.json and config copies

ANOVA compares ratings/scores across visits; it does not train feature models

## Scope

The runnable analysis starts from approved matched labels and prepared drive sources

Raw cleaning, full-motion recovery, approved drive selection and restart joining happened upstream; these three commands do not recreate every preparation step from loose raw files

Older representation/CNN code is not moved or replaced by this addition
