"""Tables for splits, experiment launches, model runs, parameters and metrics.

Files stay the source of truth for models and predictions. These rows index them
and carry what a dashboard needs to compare runs. Timestamps are ISO UTC strings,
as in data_run_registry. Foreign keys are enforced by ``get_engine``.

Identity:
    experiment_name    human label for a question, reused across launches
    experiment_run_id  one launch of that experiment (a single run or a whole sweep)
    run_id             one model fit
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    Engine,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)

STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"

tracking_metadata = MetaData()

splits = Table(
    "splits",
    tracking_metadata,
    Column("split_id", String, primary_key=True),
    Column("silver_dataset_version_id", String, nullable=False, index=True),
    Column("path", String, nullable=False),
    Column("policy_json", String, nullable=False),
    Column("key_col", String),
    Column("target_col", String),
    Column("n_rows", Integer),
    Column("n_holdout", Integer),
    Column("assignments_sha256", String),
    Column("created_at", String),
)

experiment_launches = Table(
    "experiment_launches",
    tracking_metadata,
    Column("experiment_run_id", String, primary_key=True),
    Column("experiment_name", String, nullable=False, index=True),
    Column("kind", String, nullable=False),  # "single" or "sweep"
    Column("sweep_path", String),
    Column("n_runs", Integer, nullable=False),
    Column("n_failed", Integer),
    Column("status", String, nullable=False),
    Column("started_at", String, nullable=False),
    Column("finished_at", String),
    Column("git_sha", String),
    Column("git_dirty", Integer),
    Column("error_summary", Text),
)

model_runs = Table(
    "model_runs",
    tracking_metadata,
    Column("run_id", String, primary_key=True),
    Column(
        "experiment_run_id",
        String,
        ForeignKey("experiment_launches.experiment_run_id"),
        nullable=False,
        index=True,
    ),
    Column("experiment_name", String, nullable=False, index=True),
    Column("config_hash", String, nullable=False, index=True),
    Column("status", String, nullable=False, index=True),
    Column("started_at", String, nullable=False),
    Column("finished_at", String),
    Column("silver_dataset_version_id", String, nullable=False, index=True),
    Column("feature_set_id", String),
    Column("split_id", String, nullable=False),
    Column("model_type", String, nullable=False),
    Column("seed", Integer),
    Column("git_sha", String),
    Column("git_dirty", Integer),
    Column("primary_metric", String),
    Column("primary_value", Float),
    Column("n_rounds_final", Integer),
    Column("artifact_dir", String),
    Column("holdout_evaluated_at", String),
    Column("error_summary", Text),
)

run_params = Table(
    "run_params",
    tracking_metadata,
    Column("run_id", String, ForeignKey("model_runs.run_id"), primary_key=True),
    Column("key", String, primary_key=True),
    Column("value", Text),
)

run_metrics = Table(
    "run_metrics",
    tracking_metadata,
    Column("run_id", String, ForeignKey("model_runs.run_id"), primary_key=True),
    Column("split", String, primary_key=True),  # "cv" or "holdout"
    Column("fold", Integer, primary_key=True),  # -1 = pooled over all rows
    Column("metric", String, primary_key=True, index=True),
    Column("value", Float),  # NULL when undefined
)


def ensure_tracking_tables(engine: Engine) -> None:
    tracking_metadata.create_all(engine, checkfirst=True)
