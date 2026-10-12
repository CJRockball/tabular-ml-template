"""Run one config or a sweep, recording everything in project.db and the run folder.

Order inside one run: write the ``running`` row, do the work, write the files,
write metrics, then set the final status. A ``succeeded`` row therefore means the
files exist. The holdout is not touched here; ``evaluation.evaluate_run`` does that.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import Engine

from ml_template.db import model_run_registry as registry
from ml_template.models.model_factory import AdapterFactory, build_adapter_factory
from ml_template.paths import ARTIFACTS_RUNS, DATA_SILVER, DATA_SPLITS, LOGS
from ml_template.tracking import run_artifacts
from ml_template.tracking.git_state import git_state
from ml_template.tracking.pipeline_logging import run_log
from ml_template.training.cross_validation import cross_validate, fit_final
from ml_template.training.run_config import RunConfig, config_hash, config_json, flatten_params
from ml_template.training.run_inputs import load_run_inputs

logger = logging.getLogger("ml_template.training.experiment_runner")

FactoryBuilder = Callable[[RunConfig], AdapterFactory]


def _train_and_record(
    cfg: RunConfig,
    engine: Engine,
    run_id: str,
    run_dir: Path,
    *,
    silver_root: Path,
    split_root: Path,
    factory_builder: FactoryBuilder,
) -> dict[str, Any]:
    run_artifacts.write_json(run_dir / "environment.json", run_artifacts.environment_info())
    (run_dir / "config.json").write_text(config_json(cfg), encoding="utf-8")

    inputs = load_run_inputs(cfg, silver_root=silver_root, split_root=split_root)
    key_col = inputs.contract.entity_id
    logger.info(
        "Run %s: %d rows, %d features, split %s",
        run_id,
        len(inputs.X),
        inputs.X.shape[1],
        inputs.split_id,
    )

    make_adapter = factory_builder(cfg)
    cv = cross_validate(
        make_adapter,
        inputs.X,
        inputs.y,
        inputs.assignments,
        key_col,
        validation_fraction=cfg.training.validation_fraction,
        seed=cfg.seed,
        metric_names=cfg.metrics.names,
        alert_rates=cfg.metrics.alert_rates,
        refit_rounds_factor=cfg.training.refit_rounds_factor,
    )
    final = fit_final(
        make_adapter, inputs.X, inputs.y, inputs.assignments, key_col, cv.n_rounds_final
    )

    final.save(run_dir / "model.ubj")
    cv.oof.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    cv.folds.to_csv(run_dir / "cv_folds.csv", index=False)
    run_artifacts.write_json(
        run_dir / "features.json",
        {
            "features": list(inputs.X.columns),
            "dtypes": {c: str(t) for c, t in inputs.X.dtypes.items()},
            "n_rounds_final": cv.n_rounds_final,
            "silver_dataset_version_id": cfg.data.silver_dataset_version_id,
            "split_id": inputs.split_id,
        },
    )

    registry.log_metrics(engine, run_id, cv.metric_rows)
    primary = next(
        r["value"]
        for r in cv.metric_rows
        if r["split"] == "cv" and r["fold"] == -1 and r["metric"] == cfg.metrics.primary
    )
    run_artifacts.write_artifact_manifest(run_dir)
    logger.info(
        "Run %s: cv %s=%.4f, final rounds=%d",
        run_id,
        cfg.metrics.primary,
        primary,
        cv.n_rounds_final,
    )
    return {"primary_value": primary, "n_rounds_final": cv.n_rounds_final}


def execute_run(
    cfg: RunConfig,
    engine: Engine,
    *,
    experiment_run_id: str,
    run_root: Path = ARTIFACTS_RUNS,
    log_root: Path = LOGS / "runs",
    silver_root: Path = DATA_SILVER,
    split_root: Path = DATA_SPLITS,
    factory_builder: FactoryBuilder = build_adapter_factory,
) -> str:
    """Train one config. Returns the run_id. Re-raises after recording a failure."""
    from ml_template.data.contract import load_dataset_contract
    from ml_template.training.run_config import resolve_path
    from ml_template.training.run_inputs import split_id_for

    contract = load_dataset_contract(resolve_path(cfg.data.dataset_config))
    run_id = registry.new_run_id("run")
    run_dir = run_artifacts.create_run_dir(run_id, run_root)
    git = git_state()
    registry.start_model_run(
        engine,
        run_id=run_id,
        experiment_run_id=experiment_run_id,
        experiment_name=cfg.experiment_name,
        config_hash=config_hash(cfg),
        params=flatten_params(cfg),
        silver_dataset_version_id=cfg.data.silver_dataset_version_id,
        feature_set_id=cfg.data.feature_set_id,
        split_id=split_id_for(cfg, contract),
        model_type=cfg.model.type,
        seed=cfg.seed,
        artifact_dir=run_artifacts.relative_to_root(run_dir),
        git_sha=git["git_sha"],
        git_dirty=git["git_dirty"],
    )

    with run_log(Path(log_root) / f"{run_id}.log"):
        try:
            outcome = _train_and_record(
                cfg,
                engine,
                run_id,
                run_dir,
                silver_root=silver_root,
                split_root=split_root,
                factory_builder=factory_builder,
            )
        except Exception as exc:
            logger.exception("Run %s failed", run_id)
            registry.finish_model_run(
                engine,
                run_id,
                succeeded=False,
                error_summary=f"{type(exc).__name__}: {exc}"[:500],
            )
            raise
        registry.finish_model_run(
            engine,
            run_id,
            succeeded=True,
            primary_metric=cfg.metrics.primary,
            primary_value=outcome["primary_value"],
            n_rounds_final=outcome["n_rounds_final"],
        )
    return run_id


def run_experiment(
    configs: list[RunConfig],
    engine: Engine,
    *,
    kind: str,
    sweep_path: str | None = None,
    fail_fast: bool = False,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Run configs under one launch (one experiment_run_id)."""
    if not configs:
        raise ValueError("no configs to run")
    names = {c.experiment_name for c in configs}
    if len(names) != 1:
        raise ValueError(f"a launch needs one experiment_name, got {sorted(names)}")
    (name,) = names

    git = git_state()
    launch_id = registry.start_launch(
        engine,
        experiment_name=name,
        kind=kind,
        n_runs=len(configs),
        sweep_path=sweep_path,
        git_sha=git["git_sha"],
        git_dirty=git["git_dirty"],
    )
    logger.info("Launch %s: %d runs for experiment %s", launch_id, len(configs), name)

    run_ids: list[str] = []
    failures: list[dict[str, str]] = []
    for cfg in configs:
        try:
            run_ids.append(execute_run(cfg, engine, experiment_run_id=launch_id, **run_kwargs))
        except Exception as exc:
            failures.append(
                {"config_hash": config_hash(cfg), "error": f"{type(exc).__name__}: {exc}"[:300]}
            )
            if fail_fast:
                break

    registry.finish_launch(
        engine,
        launch_id,
        succeeded=not failures,
        n_failed=len(failures),
        error_summary="; ".join(f["error"] for f in failures)[:500] or None,
    )
    logger.info(
        "Launch %s finished: %d succeeded, %d failed", launch_id, len(run_ids), len(failures)
    )
    return {"experiment_run_id": launch_id, "run_ids": run_ids, "failures": failures}
