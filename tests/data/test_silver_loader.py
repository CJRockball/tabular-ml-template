import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from ml_template.data.silver_loader import load_silver  # noqa: E402
from ml_template.tracking.run_artifacts import sha256_file  # noqa: E402

SID = "silver_test_abc"


def _build(tmp_path, *, roles=None, df=None, passed=True):
    out = tmp_path / "ds" / SID
    out.mkdir(parents=True)
    if df is None:
        n = 40
        df = pd.DataFrame(
            {
                "dataset_version_id": "ds_v1",
                "source_row_number": np.arange(1, n + 1),
                "udi": np.arange(1, n + 1),
                "product_id": [f"P{i}" for i in range(n)],
                "type": pd.Categorical(["L", "M"] * (n // 2)),
                "temp": np.linspace(300, 310, n),
                "machine_failure": ([0, 1] * (n // 2)),
                "twf": 0,
            }
        )
    df.to_parquet(out / "data.parquet", index=False)
    roles = roles or {
        "id": ["udi", "product_id"],
        "feature": ["type", "temp"],
        "target": ["machine_failure"],
        "outcome": ["twf"],
    }
    manifest = {
        "silver_dataset_version_id": SID,
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "roles": roles,
        "parquet_sha256": sha256_file(out / "data.parquet"),
        "validation_passed": passed,
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


def test_loads_and_exposes_roles(tmp_path):
    _build(tmp_path)
    silver = load_silver(SID, "ds", "udi", tmp_path)
    assert silver.features == ["type", "temp"]
    assert silver.target == "machine_failure"
    frame = silver.frame()
    assert frame.index.name == "udi" and frame.index.is_monotonic_increasing
    assert "source_row_number" not in silver.features


def test_hash_mismatch_detected(tmp_path):
    out = _build(tmp_path)
    df = pd.read_parquet(out / "data.parquet")
    df.loc[0, "temp"] = 999.0
    df.to_parquet(out / "data.parquet", index=False)
    with pytest.raises(ValueError, match="hash"):
        load_silver(SID, "ds", "udi", tmp_path)
    load_silver(SID, "ds", "udi", tmp_path, verify_hash=False)


def test_missing_and_unvalidated_silver(tmp_path):
    with pytest.raises(FileNotFoundError, match="build Silver"):
        load_silver(SID, "ds", "udi", tmp_path)
    _build(tmp_path, passed=False)
    with pytest.raises(ValueError, match="validation"):
        load_silver(SID, "ds", "udi", tmp_path)


def test_feature_role_overlap_rejected(tmp_path):
    _build(
        tmp_path,
        roles={
            "id": ["udi", "product_id"],
            "feature": ["type", "temp", "twf"],
            "target": ["machine_failure"],
            "outcome": ["twf"],
        },
    )
    with pytest.raises(ValueError, match="overlaps"):
        load_silver(SID, "ds", "udi", tmp_path)


def test_key_must_have_id_role(tmp_path):
    _build(tmp_path)
    with pytest.raises(ValueError, match="id role"):
        load_silver(SID, "ds", "temp", tmp_path)
