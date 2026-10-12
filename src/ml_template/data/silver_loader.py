"""Load a Silver dataset for modelling, with integrity and role checks.

Roles come from the Silver manifest, which records the roles used when that exact
version was built. Models select feature columns by explicit inclusion (the
``feature`` role). The Parquet also carries lineage columns such as
``dataset_version_id`` and ``source_row_number``; those are never features.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml_template.paths import DATA_SILVER

REQUIRED_ROLES = ("id", "feature", "target", "outcome")


@dataclass(frozen=True)
class SilverData:
    silver_dataset_version_id: str
    key_col: str
    df: pd.DataFrame
    roles: dict[str, list[str]]
    manifest: dict[str, Any]
    parquet_path: Path

    @property
    def features(self) -> list[str]:
        return list(self.roles["feature"])

    @property
    def target(self) -> str:
        return self.roles["target"][0]

    @property
    def outcomes(self) -> list[str]:
        return list(self.roles["outcome"])

    def frame(self) -> pd.DataFrame:
        """Rows indexed by the entity key and sorted by it."""
        return self.df.set_index(self.key_col).sort_index()


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_silver(
    silver_dataset_version_id: str,
    dataset_name: str,
    key_col: str,
    silver_root: Path = DATA_SILVER,
    verify_hash: bool = True,
) -> SilverData:
    out_dir = Path(silver_root) / dataset_name / silver_dataset_version_id
    manifest_path = out_dir / "manifest.json"
    parquet_path = out_dir / "data.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; build Silver first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("silver_dataset_version_id") != silver_dataset_version_id:
        raise ValueError(f"{manifest_path} belongs to a different Silver version")
    if not manifest.get("validation_passed"):
        raise ValueError(f"{silver_dataset_version_id} did not pass validation")
    if verify_hash:
        observed = _sha256_file(parquet_path)
        if observed != manifest["parquet_sha256"]:
            raise ValueError(f"{parquet_path} differs from the hash in its manifest")

    df = pd.read_parquet(parquet_path)
    if set(df.columns) != set(manifest["columns"]):
        raise ValueError(f"{parquet_path} columns differ from its manifest")

    roles = {k: list(v) for k, v in manifest["roles"].items()}
    missing = [r for r in REQUIRED_ROLES if r not in roles]
    if missing:
        raise ValueError(f"manifest has no roles {missing}")
    if len(roles["target"]) != 1:
        raise ValueError(f"expected exactly one target column, got {roles['target']}")
    if key_col not in roles["id"]:
        raise ValueError(f"key column {key_col!r} does not have the id role")

    forbidden = set(roles["id"]) | set(roles["target"]) | set(roles["outcome"])
    leaked = sorted(set(roles["feature"]) & forbidden)
    if leaked:
        raise ValueError(f"feature role overlaps id, target or outcome columns: {leaked}")

    if df[key_col].isna().any() or df[key_col].duplicated().any():
        raise ValueError(f"{key_col} must be unique and non-null")
    target = roles["target"][0]
    if not df[target].isin([0, 1]).all():
        raise ValueError(f"target {target!r} must be binary 0/1")

    return SilverData(
        silver_dataset_version_id=silver_dataset_version_id,
        key_col=key_col,
        df=df,
        roles=roles,
        manifest=manifest,
        parquet_path=parquet_path,
    )
