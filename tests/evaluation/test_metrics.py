import math

import numpy as np
import pytest

from ml_template.evaluation import metrics as m


def test_recall_and_precision_at_alert_rate():
    y = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    s = [0.9, 0.8, 0.1, 0.7, 0.2, 0.3, 0.4, 0.05, 0.06, 0.07]
    assert m.recall_at_alert_rate(y, s, 0.2) == pytest.approx(0.5)
    assert m.precision_at_alert_rate(y, s, 0.2) == pytest.approx(0.5)
    assert m.recall_at_alert_rate(y, s, 0.4) == pytest.approx(1.0)


def test_alert_rate_rounds_up_and_floors_at_one_row():
    y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    s = [0.9] + [0.1] * 9
    assert m.recall_at_alert_rate(y, s, 0.01) == 1.0  # k = 1, not 0
    assert m.precision_at_alert_rate(y, s, 0.15) == pytest.approx(0.5)  # k = 2


def test_tie_break_is_deterministic():
    y = [1, 0, 0, 0]
    s = [0.5, 0.5, 0.5, 0.5]
    assert m.recall_at_alert_rate(y, s, 0.25) == 1.0
    assert m.recall_at_alert_rate(y, s, 0.25) == m.recall_at_alert_rate(y, s, 0.25)


def test_pr_auc_perfect_and_single_class():
    assert m.pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert math.isnan(m.pr_auc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert math.isnan(m.roc_auc([1, 1], [0.1, 0.2]))
    assert math.isnan(m.recall_at_alert_rate([0, 0, 0], [0.1, 0.2, 0.3], 0.5))


def test_input_validation():
    with pytest.raises(ValueError):
        m.pr_auc([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        m.pr_auc([0, 1], [0.1, float("nan")])
    with pytest.raises(ValueError):
        m.pr_auc([0, 1, 1], [0.1, 0.2])
    with pytest.raises(ValueError):
        m.pr_auc([], [])
    with pytest.raises(ValueError):
        m.recall_at_alert_rate([0, 1], [0.1, 0.2], 0.0)
    with pytest.raises(ValueError):
        m.recall_at_alert_rate([0, 1], [0.1, 0.2], 1.5)
    with pytest.raises(ValueError):
        m.brier([0, 1], [0.1, 1.5])


def test_keys_and_resolved_keys():
    assert m.alert_label(0.05) == "5pct"
    assert m.alert_label(0.125) == "12.5pct"
    names = ["pr_auc", "recall_at_alert_rate", "precision_at_alert_rate"]
    assert m.resolved_keys(names, [0.05, 0.1]) == [
        "pr_auc",
        "recall_at_5pct",
        "recall_at_10pct",
        "precision_at_5pct",
        "precision_at_10pct",
    ]
    with pytest.raises(ValueError):
        m.resolved_keys(["recall_at_alert_rate"], [])
    with pytest.raises(ValueError):
        m.resolved_keys(["nope"])


def test_compute_metrics_matches_resolved_keys():
    y = [0, 1, 0, 1, 0, 0, 1, 0]
    s = [0.1, 0.9, 0.2, 0.8, 0.3, 0.15, 0.7, 0.05]
    names = ["pr_auc", "roc_auc", "brier", "recall_at_alert_rate", "precision_at_alert_rate"]
    out = m.compute_metrics(y, s, names, [0.25, 0.5])
    assert list(out) == m.resolved_keys(names, [0.25, 0.5])
    assert out["recall_at_25pct"] == pytest.approx(2 / 3)


def test_long_form_rows():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 40)
    s = rng.random(40)
    folds = np.repeat([0, 1], 20)
    rows = m.long_form_metrics(y, s, ["pr_auc"], split="cv", folds=folds)
    assert {(r["split"], r["fold"]) for r in rows} == {("cv", -1), ("cv", 0), ("cv", 1)}
    hold = m.long_form_metrics(y, s, ["pr_auc"], split="holdout")
    assert [(r["fold"], r["metric"]) for r in hold] == [(-1, "pr_auc")]
    with pytest.raises(ValueError):
        m.long_form_metrics(y, s, ["pr_auc"], split="cv", folds=[0, 1])
