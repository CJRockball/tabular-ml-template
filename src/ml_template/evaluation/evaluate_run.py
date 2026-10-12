"""Evaluate a finished run on the holdout. This is the only code that reads it.

    python -m ml_template.evaluation.evaluate_run --run-id run_20261011_101530_ab12cd

Each run is evaluated once. A second call returns the stored metrics. ``--force``
re-evaluates and replaces them. ``model_runs.holdout_evaluated_at`` records the
first look, so the number of holdout evaluations per experiment can be counted.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from ml_template.db import model_run_registry as registry
from ml_template.evaluation.metrics import long_form_metrics
from ml_template.models.model_factory import ModelAdapter, load_adapter
from ml_template.paths import DATA_SILVER, DATA_SPLITS
from ml_template.tracking import run_artifacts
from ml_template.training.run_config import RunConfig
from ml_template.training.run_inputs import load_run_inputs

logger = logging.getLogger("ml_template.evaluation.evaluate_run")

AdapterLoader = Callable[[RunConfig, Path], ModelAdapter]


def evaluate_holdout(
    engine: Engine,
    run_id: str,
    *,
    force: bool = False,
    silver_root: Path = DATA_SILVER,
    split_root: Path = DATA_SPLITS,
    adapter_loader: AdapterLoader = load_adapter,
) -> dict[str, float | None]:
    run = registry.get_run(engine, run_id)
    if run is None:
        raise KeyError(f"unknown run_id {run_id}")
    if run["status"] != "succeeded":
        raise ValueError(f"run {run_id} has status {run['status']!r}; only succeeded runs qualify")

    if run["holdout_evaluated_at"] and not force:
        logger.info("Run %s was already evaluated; returning stored metrics", run_id)
        stored = registry.get_metrics(engine, run_id, split="holdout")
        return {m["metric"]: m["value"] for m in stored}

    run_dir = run_artifacts.resolve_stored_path(run["artifact_dir"])
    run_artifacts.verify_artifact(run_dir, "model.ubj")
    cfg = RunConfig.model_validate_json((run_dir / "config.json").read_text(encoding="utf-8"))

    inputs = load_run_inputs(cfg, silver_root=silver_root, split_root=split_root)
    key_col = inputs.contract.entity_id
    hold_keys = inputs.assignments.loc[inputs.assignments["partition"] == "holdout", key_col]
    hold_keys = hold_keys.to_numpy()

    adapter = adapter_loader(cfg, run_dir / "model.ubj")
    p_hold = adapter.predict_proba(inputs.X.loc[hold_keys])
    y_hold = inputs.y.loc[hold_keys].to_numpy()

    rows = long_form_metrics(
        y_hold, p_hold, cfg.metrics.names, cfg.metrics.alert_rates, split="holdout"
    )
    pd.DataFrame({key_col: hold_keys, "y_true": y_hold, "p_hold": p_hold}).to_parquet(
        run_dir / "holdout_predictions.parquet", index=False
    )
    registry.log_metrics(engine, run_id, rows)
    registry.mark_holdout_evaluated(engine, run_id)
    run_artifacts.write_artifact_manifest(run_dir)

    result = {r["metric"]: r["value"] for r in rows}
    logger.info("Run %s holdout: %s", run_id, {k: round(v, 4) for k, v in result.items() if v == v})
    return result


def main(argv: list[str] | None = None) -> None:
    from ml_template.db.connection import get_engine
    from ml_template.tracking.pipeline_logging import setup_logging

    p = argparse.ArgumentParser(description="Evaluate a finished run on the holdout.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--force", action="store_true", help="re-evaluate and replace stored metrics")
    args = p.parse_args(argv)

    setup_logging()
    evaluate_holdout(get_engine(), args.run_id, force=args.force)


if __name__ == "__main__":
    main()
