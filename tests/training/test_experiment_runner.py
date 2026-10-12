import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

pytest.importorskip("pyarrow")
sqlalchemy = pytest.importorskip("sqlalchemy")

from ml_template.db import model_run_registry as reg  # noqa: E402
from ml_template.evaluation.evaluate_run import evaluate_holdout  # noqa: E402
from ml_template.tracking.run_artifacts import sha256_file  # noqa: E402
from ml_template.training import experiment_runner as er  # noqa: E402
from ml_template.training.run_config import RunConfig, load_config  # noqa: E402
from ml_template.training.split import make_split, save_split  # noqa: E402

SID = "silver_test_abc"
CONTRACT = """
data: {entity_id: UDI, target: Machine failure}
bronze_source_contract:
  roles:
    id: [UDI]
    feature: [Temp, Speed]
    target: [Machine failure]
    outcome: [TWF]
silver_eda_contract:
  canonical_names: {UDI: udi, Temp: temp, Speed: speed, Machine failure: machine_failure, TWF: twf}
"""


class FakeAdapter:
    max_rounds = 100

    def __init__(self):
        self.m = None

    def fit_with_validation(self, Xt, yt, Xv, yv):
        self.m = LogisticRegression(max_iter=300).fit(Xt, yt)
        return 12

    def refit(self, X, y, n_rounds):
        self.m = LogisticRegression(max_iter=300).fit(X, y)

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]

    def save(self, path):
        path.write_text("fake-model")


def fake_builder(cfg):
    return FakeAdapter


def fake_loader(cfg, path):
    adapter = FakeAdapter()
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"temp": rng.normal(size=300), "speed": rng.normal(size=300)})
    adapter.m = LogisticRegression().fit(X, (X["temp"] > 1).astype(int))
    return adapter


@pytest.fixture
def world(tmp_path):
    rng = np.random.default_rng(0)
    n = 1200
    temp, speed = rng.normal(size=n), rng.normal(size=n)
    fail = (temp + 0.5 * rng.normal(size=n) > 1.4).astype(int)
    df = pd.DataFrame(
        {
            "udi": np.arange(1, n + 1),
            "temp": temp,
            "speed": speed,
            "machine_failure": fail,
            "twf": 0,
        }
    )
    silver_root, split_root = tmp_path / "silver", tmp_path / "splits"
    out = silver_root / "ds" / SID
    out.mkdir(parents=True)
    df.to_parquet(out / "data.parquet", index=False)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "silver_dataset_version_id": SID,
                "columns": {c: str(t) for c, t in df.dtypes.items()},
                "roles": {
                    "id": ["udi"],
                    "feature": ["temp", "speed"],
                    "target": ["machine_failure"],
                    "outcome": ["twf"],
                },
                "parquet_sha256": sha256_file(out / "data.parquet"),
                "validation_passed": True,
            }
        )
    )
    (tmp_path / "contract.yaml").write_text(CONTRACT)
    (tmp_path / "xgb.yaml").write_text("type: xgb\nparams: {max_depth: 3}\n")
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "experiment_name: e2e\n"
        f"data: {{dataset_name: ds, dataset_config: {tmp_path / 'contract.yaml'}, "
        f"silver_dataset_version_id: {SID}}}\n"
        f"model_file: {tmp_path / 'xgb.yaml'}\n"
    )
    cfg = load_config(exp)
    assign = make_split(df, cfg.split, "udi", "machine_failure")
    save_split(assign, SID, cfg.split, "udi", "machine_failure", df["machine_failure"], split_root)
    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'p.db'}")
    kwargs = {
        "run_root": tmp_path / "runs",
        "log_root": tmp_path / "logs",
        "silver_root": silver_root,
        "split_root": split_root,
        "factory_builder": fake_builder,
    }
    return cfg, engine, kwargs, tmp_path


def _bad_config(cfg):
    data = {**cfg.data.model_dump(), "silver_dataset_version_id": "missing"}
    return RunConfig.model_validate({**cfg.model_dump(), "data": data})


def test_single_run_records_everything(world):
    cfg, engine, kwargs, tmp_path = world
    summary = er.run_experiment([cfg], engine, kind="single", **kwargs)
    assert not summary["failures"] and len(summary["run_ids"]) == 1
    run_id = summary["run_ids"][0]
    run = reg.get_run(engine, run_id)
    assert run["status"] == "succeeded"
    assert run["experiment_run_id"] == summary["experiment_run_id"]
    assert run["primary_metric"] == "pr_auc" and run["primary_value"] > 0
    assert run["n_rounds_final"] == 12 and run["holdout_evaluated_at"] is None
    metrics = {(m["split"], m["fold"], m["metric"]) for m in reg.get_metrics(engine, run_id)}
    assert ("cv", -1, "pr_auc") in metrics and ("cv", 0, "best_round") in metrics
    assert not any(split == "holdout" for split, _, _ in metrics)
    run_dir = tmp_path / "runs" / run_id
    for name in (
        "config.json",
        "environment.json",
        "features.json",
        "model.ubj",
        "oof_predictions.parquet",
        "cv_folds.csv",
        "artifact_manifest.json",
    ):
        assert (run_dir / name).exists(), name
    log_text = (tmp_path / "logs" / f"{run_id}.log").read_text()
    assert f"Run {run_id}" in log_text  # the run log holds this run's own messages
    assert "fold 0" not in log_text or "best_round" in log_text


def test_two_runs_have_separate_logs_and_ids(world):
    cfg, engine, kwargs, tmp_path = world
    cfg2 = RunConfig.model_validate({**cfg.model_dump(), "seed": 7})
    summary = er.run_experiment([cfg, cfg2], engine, kind="sweep", sweep_path="s.yaml", **kwargs)
    a, b = summary["run_ids"]
    assert a != b
    log_a = (tmp_path / "logs" / f"{a}.log").read_text()
    log_b = (tmp_path / "logs" / f"{b}.log").read_text()
    assert a in log_a and b not in log_a
    assert b in log_b and a not in log_b


def test_failure_is_recorded_and_sweep_continues(world):
    cfg, engine, kwargs, tmp_path = world
    bad = _bad_config(cfg)
    summary = er.run_experiment([bad, cfg], engine, kind="sweep", **kwargs)
    assert len(summary["failures"]) == 1 and len(summary["run_ids"]) == 1
    runs = reg.list_runs(engine, experiment_name="e2e")
    assert sorted(r["status"] for r in runs) == ["failed", "succeeded"]
    failed = next(r for r in runs if r["status"] == "failed")
    assert "build Silver first" in failed["error_summary"]
    assert "Traceback" in (tmp_path / "logs" / f"{failed['run_id']}.log").read_text()
    stopped = er.run_experiment([bad, cfg], engine, kind="sweep", fail_fast=True, **kwargs)
    assert stopped["run_ids"] == []


def test_holdout_is_evaluated_once_unless_forced(world):
    cfg, engine, kwargs, tmp_path = world
    run_id = er.run_experiment([cfg], engine, kind="single", **kwargs)["run_ids"][0]
    roots = {"silver_root": kwargs["silver_root"], "split_root": kwargs["split_root"]}
    first = evaluate_holdout(engine, run_id, adapter_loader=fake_loader, **roots)
    assert "pr_auc" in first and "recall_at_5pct" in first
    assert reg.get_run(engine, run_id)["holdout_evaluated_at"]
    assert (tmp_path / "runs" / run_id / "holdout_predictions.parquet").exists()
    again = evaluate_holdout(engine, run_id, adapter_loader=None, **roots)
    assert again == first
    forced = evaluate_holdout(engine, run_id, force=True, adapter_loader=fake_loader, **roots)
    assert forced.keys() == first.keys()


def test_holdout_refuses_failed_runs_and_tampered_models(world):
    cfg, engine, kwargs, tmp_path = world
    er.run_experiment([_bad_config(cfg)], engine, kind="single", **kwargs)
    failed = reg.list_runs(engine, status="failed")[0]["run_id"]
    with pytest.raises(ValueError, match="succeeded"):
        evaluate_holdout(engine, failed)
    run_id = er.run_experiment([cfg], engine, kind="single", **kwargs)["run_ids"][0]
    (tmp_path / "runs" / run_id / "model.ubj").write_text("tampered")
    roots = {"silver_root": kwargs["silver_root"], "split_root": kwargs["split_root"]}
    with pytest.raises(ValueError, match="differs"):
        evaluate_holdout(engine, run_id, **roots)
