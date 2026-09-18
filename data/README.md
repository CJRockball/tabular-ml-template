# Local data

This directory contains local runtime state and is not committed to Git.

## Layout

- `raw/`: original source files acquired by the user.
- `processed/`: canonical cleaned or partitioned outputs.
- `exports/`: generated extracts and score outputs.
- `project.db`: local SQLite database for dataset metadata, runs, metrics, and artifact references.

## AI4I reference data

Place the original AI4I CSV in `data/raw/`.

The repository commits only small, deterministic test fixtures in
`tests/fixtures/`. These support offline tests and CI; they are not intended
for model performance evaluation.