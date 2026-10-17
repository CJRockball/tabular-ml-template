# Validation performed

## Results in the authoring sandbox

- Parsed all 14 Python files with ast.parse: passed.
- Portable pytest suite: 28 passed, 4 skipped, 1 warning.
- Numerical unit tests covered fitting, probability validation, ordering,
  constant/boundary scores and exact JSON roundtrip.
- Diagnostic tests covered held-fold exclusion and reliability-bin support.
- Alignment tests covered keyed joins, missing/duplicate keys, holdout rejection,
  mismatched fold membership and invalid labels.
- Config tests covered explicit test parent, invalid values and unknown keys.

## Skips and limitations

The sandbox lacks SQLAlchemy and PyArrow, and has no checked-out full repository.
Two test modules (registry and synthetic workflow) skipped at module collection
because SQLAlchemy was unavailable. Two real-parent tests skipped because they
require ML_TEMPLATE_CALIBRATION_LOCAL=1 and the local run's data/artifacts.
The 4 skip count therefore means two modules plus two tests, not four test cases.
The warning was the local_data marker being unknown in the isolated sandbox;
the target repository already registers that marker in pyproject.toml.

The real target run was not calibrated. No Brier/log-loss/holdout result for it
is included. Full repository import integration, SQLAlchemy table creation,
PyArrow filtered reads/writes, model loading and Ruff checks remain local tasks.
No network installation or substitution of missing runtime dependencies was used.

The synthetic loss-improvement unit test is intentionally in-sample and is only
a numerical smoke test, not evidence of out-of-sample model improvement.

## Captured pytest output

```text
................ss............                                           [100%]
28 passed, 4 skipped, 1 warning in 0.81s
```
