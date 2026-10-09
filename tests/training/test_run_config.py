from pathlib import Path

import pytest
from pydantic import ValidationError

from ml_template.paths import CONFIGS
from ml_template.training import run_config as rc

MODEL = "type: xgb\nparams:\n  max_depth: 4\n  learning_rate: 0.05\n"


def _write(tmp_path: Path, extra: str = "") -> Path:
    (tmp_path / "xgb.yaml").write_text(MODEL)
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "experiment_name: t1\n"
        "data:\n  dataset_name: ds\n  dataset_config: c.yaml\n"
        "  silver_dataset_version_id: silver_x\n"
        f"model_file: {tmp_path / 'xgb.yaml'}\n" + extra
    )
    return exp


def test_load_and_override(tmp_path):
    cfg = rc.load_config(_write(tmp_path), ["model.params.max_depth=7", "seed=1"])
    assert cfg.model.params.max_depth == 7
    assert cfg.seed == 1
    assert cfg.split.n_folds == 5
    assert cfg.schema_version == 1


def test_invalid_values_rejected(tmp_path):
    exp = _write(tmp_path)
    for override in (
        "model.params.max_dept=3",  # unknown key
        "model.params.learning_rate=-1",
        "model.params.objective=reg:squarederror",
        "experiment_name=bad name",
        "metrics.primary=brier",  # not in names
        "metrics.alert_rates=[0, 0.1]",
        "split.holdout_fraction=1.5",
        "schema_version=2",
    ):
        with pytest.raises(ValidationError):
            rc.load_config(exp, [override])


def test_primary_may_be_alert_key(tmp_path):
    cfg = rc.load_config(_write(tmp_path), ["metrics.primary=recall_at_5pct"])
    assert cfg.metrics.primary == "recall_at_5pct"


def test_override_syntax_error(tmp_path):
    with pytest.raises(ValueError, match="key=value"):
        rc.load_config(_write(tmp_path), ["seed"])


def test_hash_stable_and_sensitive(tmp_path):
    exp = _write(tmp_path)
    a, b = rc.load_config(exp), rc.load_config(exp)
    c = rc.load_config(exp, ["model.params.max_depth=9"])
    assert rc.config_hash(a) == rc.config_hash(b) != rc.config_hash(c)


def test_flatten_params(tmp_path):
    flat = rc.flatten_params(rc.load_config(_write(tmp_path)))
    assert flat["model.params.max_depth"] == "4"
    assert flat["metrics.alert_rates"] == "[0.05]"
    assert flat["data.silver_dataset_version_id"] == "silver_x"


def test_expand_sweep(tmp_path):
    exp = _write(tmp_path)
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        f"base: {exp}\nexperiment_name: grid1\ngrid:\n"
        "  model.params.max_depth: [3, 5]\n  model.params.learning_rate: [0.1, 0.2, 0.3]\n"
    )
    cfgs = rc.expand_sweep(sweep)
    assert len(cfgs) == 6
    assert {c.experiment_name for c in cfgs} == {"grid1"}
    assert len({rc.config_hash(c) for c in cfgs}) == 6


def test_sweep_guards(tmp_path):
    exp = _write(tmp_path)
    big = tmp_path / "big.yaml"
    values = list(range(1, 30))
    big.write_text(
        f"base: {exp}\ngrid:\n  model.params.max_depth: {values}\n"
        f"  model.params.n_estimators: {values}\n"
    )
    with pytest.raises(ValueError, match="limit"):
        rc.expand_sweep(big)
    typo = tmp_path / "typo.yaml"
    typo.write_text(f"base: {exp}\ngrid:\n  seed: [1]\nbasee: x\n")
    with pytest.raises(ValueError, match="unknown sweep keys"):
        rc.expand_sweep(typo)
    empty = tmp_path / "empty.yaml"
    empty.write_text(f"base: {exp}\ngrid:\n  seed: []\n")
    with pytest.raises(ValueError, match="non-empty"):
        rc.expand_sweep(empty)


def test_resolve_path_falls_back_to_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert rc.resolve_path("configs/model/xgb.yaml") == CONFIGS / "model" / "xgb.yaml"


def test_repository_configs_are_valid():
    cfg = rc.load_config(CONFIGS / "experiment" / "xgb_baseline.yaml")
    assert cfg.model.params.objective == "binary:logistic"
    assert len(rc.expand_sweep(CONFIGS / "sweep" / "xgb_grid.yaml")) == 12


def test_cli_prints_hash(tmp_path, capsys):
    rc.main(["--config", str(_write(tmp_path)), "--set", "seed=3"])
    out = capsys.readouterr().out
    assert '"seed": 3' in out and "hash:" in out
