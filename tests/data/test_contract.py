import pytest

from ml_template.data.contract import load_dataset_contract
from ml_template.paths import CONFIGS

YAML = """
data:
  entity_id: ID
  target: Y
roles:
  id: [ID]
  feature: [A, B]
  target: [Y]
  outcome: [F1]
silver_eda_contract:
  canonical_names: {ID: id, A: a, B: b, Y: y, F1: f1}
  canonical_dtypes: {id: int64, a: float64, b: category, y: int8, f1: int8}
"""


def _write(tmp_path, text=YAML):
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return path


def test_contract_in_canonical_names(tmp_path):
    c = load_dataset_contract(_write(tmp_path))
    assert (c.entity_id, c.target) == ("id", "y")
    assert c.features == ("a", "b")
    assert c.outcomes == ("f1",)
    assert c.dtypes == {"a": "float64", "b": "category"}
    assert set(c.excluded_from_features) == {"id", "y", "f1"}


def test_missing_canonical_name_rejected(tmp_path):
    with pytest.raises(ValueError, match="canonical name"):
        load_dataset_contract(_write(tmp_path, YAML.replace("B: b, ", "")))


def test_overlap_rejected(tmp_path):
    with pytest.raises(ValueError, match="both"):
        load_dataset_contract(_write(tmp_path, YAML.replace("outcome: [F1]", "outcome: [F1, A]")))


def test_target_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError, match="disagrees"):
        load_dataset_contract(_write(tmp_path, YAML.replace("target: Y\nroles", "target: A\nroles")))


def test_real_ai4i_contract():
    c = load_dataset_contract(CONFIGS / "ai4i_binary.yaml")
    assert c.entity_id == "udi"
    assert c.target == "machine_failure"
    assert set(c.outcomes) == {"twf", "hdf", "pwf", "osf", "rnf"}
    assert "type" in c.features and len(c.features) == 6
    assert not set(c.features) & set(c.excluded_from_features)
