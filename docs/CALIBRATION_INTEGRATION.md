# Calibration implementation and integration notes

## Status and scope

Prepared for CJRockball/tabular-ml-template, GitHub main inspected 2026-10-17.
The test configuration points to run_20261016_121745_958bec. That run's local
SQLite database, data and artifacts were not supplied to the authoring environment.
No calibration results for that run have been produced or claimed.

This archive is a source overlay, not a GitHub commit or an automatic patch.
It replaces only src/ml_template/training/calibrate.py. Back up that file or use
a new local branch before copying. Other source files in the archive are new.
Do not overwrite package __init__.py files; none are supplied.

## Install and run

1. Back up project.db. Copy src/, configs/, tests/, docs/ and integration/ into the repository root,
   preserving their relative layout. Inspect the replacement calibrate.py diff.
2. Existing pyproject.toml already declares NumPy, pandas, SciPy, scikit-learn,
   Pydantic 2, PyYAML, SQLAlchemy and PyArrow. No dependency addition is required.
3. From the repository root, run:

```bash
uv run python -m pytest -q tests/test_probability_calibrator.py tests/test_calibration_*.py
uv run python -m ml_template.training.calibrate --config configs/calibration/test_958bec.yaml --check
uv run python -m ml_template.training.calibrate --config configs/calibration/test_958bec.yaml
```

The check validates inputs and class counts, without fitting or adding calibration
records. Existing get_run() may ensure the existing training registry tables exist.
A fit creates a unique cal_* ID and two new additive SQLite tables, calibration_runs
and calibration_metrics. It does not alter model_runs, run_metrics or parent files.
The command prints the calibration ID and its artifact directory.

Optional Make integration: add this line to the root Makefile:

```make
include integration/Makefile.calibration.mk
```

Then use `make calibrate-check` and `make calibrate`. Set CALIBRATION_CONFIG=...
for other calibration YAML files. This is separate from EXP and SWEEP.

Optional real-data smoke tests (read local data; only write pytest temporary files):

```bash
ML_TEMPLATE_CALIBRATION_LOCAL=1 uv run python -m pytest -q tests/test_calibration_local.py
```

Run the full repository tests and `uv run ruff check` before merging. The sandbox
could not execute the full repository, SQLAlchemy/PyArrow integration tests, or
Ruff. See VALIDATION.md for exact validation performed.

## New files and responsibilities

- models/probability_calibrator.py: numerical mapping and plain JSON serialization.
- models/calibration_artifacts.py: parent-bound loading, hashes, status validation.
- training/calibration_config.py: separate strict calibration YAML schema.
- training/calibration_inputs.py: integrity checks, key alignment, development labels.
- evaluation/calibration_diagnostics.py: metrics, reliability bins, OOF-pair cross-fit.
- db/calibration_registry.py: additive SQLAlchemy tables and lifecycle operations.
- tests/test_calibration_*.py and test_probability_calibrator.py: portable unit,
  synthetic orchestration, registry and opt-in local-data tests.

## Statistical contract

The sigmoid is positive-slope logistic recalibration of clipped probability logits:

\[
z = \operatorname{logit}(\operatorname{clip}(p,\epsilon,1-\epsilon)),
\qquad p_{cal}=\sigma(a z+b),\qquad a\ge 10^{-8}.
\]

The objective is average unweighted binary log loss plus \(\lambda a^2/2\).
Defaults: epsilon=1e-7 and lambda=1e-6. The intercept is unpenalized. This is an
explicit constrained logistic implementation, not a claim of exact equivalence
to sklearn's built-in sigmoid fitter or to the old probability-input Platt fitter.
A slope at its floor is reported as a warning, not treated as successful
probability-quality confirmation. It can indicate weak or reversed score signal.

No class weighting, resampling, seed averaging or fold-model ensembling is used.
The final calibrator fits all development OOF score-label pairs. Its predictions
on these same pairs (p_cal_fit) are stored solely for reproducibility, not reported
as assessment metrics.

Diagnostic predictions (p_cal_crossfit) exclude each stored base-CV fold from the
calibrator fit. They do not exclude the complete base-model training/selection
process. Overlapping base training sets still induce dependencies. Therefore label
these results exploratory, not independent or nested-CV estimates. Group/time
validation will require appropriately grouped/forward calibration diagnostics;
this implementation targets the current stratified binary template.

Different diagnostic folds use different calibration mappings, which can change
pooled ranking. The one final calibrator is nondecreasing, but clipping and
floating-point saturation can create ties. Never assert exact ranking equality.

Reliability bins use quantile edges, keeping tied scores together; fewer bins than
requested are possible. Show counts and positives. Wilson intervals are pointwise
Bernoulli intervals assuming independent rows; they do not account for repeated
entities, temporal dependence, parameter uncertainty or multiple comparisons.

No probability improvement is guaranteed on new data. Do not choose calibrators,
features or parameters after seeing final-test feedback and still call it untouched.

## Data and integrity contract

The parent must have succeeded and have a valid artifact_manifest.json containing
config.json, features.json, model.ubj, oof_predictions.parquet and (when needed)
model.transformer.joblib. Missing historical artifacts cause a clear failure;
there is deliberately no positional-label or old .npy fallback.

The loader checks the parent config hash against the DB, Silver/feature-set IDs,
Silver manifest/version/data SHA-256, recomputed split ID, split assignment hash,
OOF uniqueness, exact development coverage and stored fold identity. The stored
split manifest supplies entity/target names; this does not depend on a mutable
current dataset-contract YAML. The Silver manifest roles must agree.

Only entity and target columns for development keys are returned by the filtered
PyArrow read. No holdout prediction file is opened, and no holdout label enters
fitting or diagnostics. Whole-file hash verification reads Silver bytes, including
physical data pages containing held-out rows; this is integrity verification, not
use of holdout labels. If strict physical access isolation is required, store
partitions separately before adding calibration.

If holdout_evaluated_at is already set, the fit logs a warning and records it.
This is useful for a test run but does not restore a holdout previously used for
selection. Changes to data/feature artifacts require a new parent model/calibrator.

## Artifact outputs

artifacts/calibration/<calibration_id>/ contains:
- calibrator.json: portable mapping parameters; no executable pickle.
- config.json: calibration settings and explicit parent ID.
- provenance.json: parent, dataset, split and source-file identities/hashes.
- development_predictions.parquet: key, fold, target, raw, cross-fit and fitted values.
- metrics.csv: raw and cross-fit metrics pooled and by fold.
- reliability.csv: quantile-bin support, event rates, means and Wilson intervals.
- diagnostic_notes.json: interpretation, warnings and limitations.
- environment.json, git_state.json, artifact_manifest.json.

The corresponding DB metrics use split=dev_calibration_diagnostic and separate
variant=raw / calibrated_crossfit. They are not inserted into existing run_metrics.
The current dashboard will not automatically discover these tables/files.

## Required existing-file modifications for the next integration

### training/calibrate.py -- replacement supplied

Replace the old wafer-specific script with the supplied file. Old --run-id,
--method and automatic-latest CLI usage is intentionally replaced by --config.
Do not retain both implementations under competing supported paths.

### evaluation/evaluate_run.py -- notes only; NOT modified by this archive

Keep this as the sole controlled holdout-evaluation entry point. Add an optional
--calibration-id and validate it belongs to the requested --run-id using:

```python
from ml_template.models.calibration_artifacts import load_calibrator_for_run

cal = load_calibrator_for_run(engine, calibration_id, run_id)
p_cal = cal.transform(p_hold)
```

Make this change as one coherent evaluation/registry change, not by simply adding
calibrated metrics to the existing long_form_metrics call:

1. Existing caching is keyed by run_id only. Introduce a separate evaluation
   identity including parent run, calibration ID (or raw), policy ID if present,
   and evaluator version. A raw cached result is not a calibrated cached result.
2. Prefer evaluating frozen raw and calibrated variants together on the first look.
   Reuse existing recorded raw predictions if already evaluated; record that this
   is a post-hoc calibration assessment with prior test exposure. Do not use
   --force to silently replace raw records or erase first-look history.
3. Add a separate evaluation table keyed by evaluation ID, with variant-aware
   metrics and holdout_seen_at. Do not use calibration_runs to imply that the
   calibrator itself was fitted on the holdout.
4. Existing model_run_registry.log_metrics deletes all rows for the supplied
   parent split. Calling it twice (raw then calibrated) with split='holdout' will
   destroy the first variant. Keep the original raw rows; store calibrated
   evaluation under the separate evaluation identity/table.
5. Store keyed raw and calibrated predictions under that evaluation's directory,
   not by overwriting holdout_predictions.parquet or the calibration artifact.
6. Verify model.transformer.joblib and use load_adapter so fitted group statistics
   remain identical. Never refit preprocessing at evaluation or inference time.
7. Record Brier, log loss, reliability support and operating metrics for both
   variants. Freeze method/policy before accessing holdout outcomes.

This archive intentionally does NOT introduce a second holdout evaluator or read
holdout data from the calibration command.

### evaluation/metrics.py -- notes only

Brier/AP/ROC-AUC are already implemented. Add 'log_loss' to METRIC_NAMES and
explicit dispatch in compute_metrics if making it a standard training/evaluation
metric. Its current final else branch means 'brier'; do not add a name without
updating that dispatch. Keep all existing metric keys backward compatible.
The new calibration diagnostic module already computes log loss independently,
so changing this file is not required to run the supplied calibration CLI.

### db/tracking_tables.py / central schema initialization -- notes only

The supplied calibration_registry owns two additive tables and creates them when
starting a fit. For this prototype no existing schema migration is needed. If you
want centralized schema management, move those definitions into tracking_tables
and call the relevant create_all/upgrade from its initializer; remove the duplicate
metadata definitions in calibration_registry at the same time. Do not maintain
two authoritative definitions. Consider foreign keys and a unified migration plan
when release/evaluation tables are added.

### tracking/run_artifacts.py -- no required change

Existing manifest and SHA-256 helpers are reused. Parent artifact files are never
modified. Calibration artifacts get a separate manifest before succeeded is set.
Artifacts should be treated as immutable once succeeded. The loader can detect
accidental mutation; this is not a cryptographic authenticity/signature system.

### models/model_factory.py and training/cross_validation.py -- no required change

The chosen clipped-logit probability input uses the current predict_proba and
p_oof interfaces. No raw-margin method or base-model retraining is needed.

### dashboard -- notes only

Add read-only access to calibration_runs/calibration_metrics by parent_run_id.
Show raw versus calibrated_crossfit diagnostic metrics, reliability.csv and counts;
explicitly label 'development diagnostic, not final evaluation'. Do not compare
raw parent CV metrics with calibration-fit metrics. Add final-test panels only
once the separate variant-aware evaluation identity exists. Never run fitting
inside dashboard callbacks.

### evaluation/threshold.py -- deletion candidate, NOT deleted

The inspected legacy file is a wafer-specific timestamp-printing stub. Check
imports, CLI entry points, tests and docs before deleting. Decision policy belongs
in a future real policy module, not in the calibrator.

### evaluation/reports.py / evaluation/explain.py -- no change in this bundle

Retire dataset-specific reports only after generic batch-scorecard replacement.
Keep explain.py for later base-model explanation; calibration does not replace it.

## Acceptance checklist

- Target parent exists locally and --check succeeds.
- Portable tests, synthetic workflow/registry tests and full repository tests pass.
- Optional real-run roundtrip passes for run_20261016_121745_958bec.
- A fresh cal_* record succeeds and manifest verification passes.
- Parent files and parent metric records are unchanged.
- No holdout file is loaded by the calibration-fitting path.
- Saved mapping reproduces its probabilities exactly in the same environment.
- Holdout evaluation remains disabled until variant-aware evaluation is integrated.
