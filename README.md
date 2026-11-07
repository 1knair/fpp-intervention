Run this from the project root, with dependencies and the assessment data (private) in 'data/clinical_assessment.csv', and prepared telemetry/labels/mappings in 'outputs/data/':

1. Put your paths in config/; then put the features, targets, models you want for your regression test
2. Extract features:
   '.venv\Scripts\python.exe run_project.py inputs'
3. Train models and calculate metrics:
   '.venv\Scripts\python.exe run_project.py regression'
4. Compare ratings/scores across visits:
   '.venv\Scripts\python.exe run_project.py anova'

Results: 'outputs/runs/<run_name>__<timestamp>/'

Cleaning and approved drive/mapping preparation happen before these steps
