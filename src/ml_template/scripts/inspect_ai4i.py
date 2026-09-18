# %%
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from IPython.display import display
import pandas as pd

from ml_template.config import DATA_NAME
from ml_template.paths import DATA_RAW, ARTIFACTS

# %%
@dataclass(frozen=True)
class SourceMetadata:
    source_name: str
    source_path: str
    source_size_bytes: int
    source_sha256: str
    encoding: str
    delimiter: str

@dataclass(frozen=True)
class ObservedSchema:
    row_count: int
    column_count: int
    headers: list[str]
    pandas_inferred_dtypes: dict[str, str]
    null_counts: dict[str, int]
    flag_cardinality: dict[str, int]

@dataclass(frozen=True)
class ProposedCanonicalSchema:
    dtypes: dict[str, str]
    binary_flags: list[str]

@dataclass(frozen=True)
class SourceInspectionReport:
    report_version: int
    created_at_utc: str
    source: SourceMetadata
    observed_schema: ObservedSchema
    proposed_canonical_schema: ProposedCanonicalSchema

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

# %%
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

# %%
encoding = "utf-8"
delimiter = ","
source_path = DATA_RAW / DATA_NAME

df = pd.read_csv(
    source_path,
    encoding=encoding,
    sep=delimiter,
)

display(df.head())
print(df.info())

# %%
binary_flags = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
flag_cardinality = {name: int(df[name].nunique()) for name in binary_flags if name in df.columns}
for name, count in flag_cardinality.items():
    print(f"{name} number of unique values: {count}")

null_counts = {col: int(cnt) for col, cnt in df.isnull().sum().items()}
print("Null counts per column:\n", null_counts)

# %%
# Build proposed types: map binary flags to Int8, keep others aligned with observed types
proposed_dtypes: dict[str, str] = {}
for col, dtype in df.dtypes.items():
    if col in binary_flags:
        proposed_dtypes[col] = "Int8"
    elif col == "UDI":
        proposed_dtypes[col] = "Int64"
    else:
        proposed_dtypes[col] = str(dtype)

# %%
source_meta = SourceMetadata(
    source_name=DATA_NAME,
    source_path=str(source_path),
    source_size_bytes=source_path.stat().st_size,
    source_sha256=sha256_file(source_path),
    encoding=encoding,
    delimiter=delimiter,
)

observed_schema = ObservedSchema(
    row_count=int(df.shape[0]),
    column_count=int(df.shape[1]),
    headers=df.columns.tolist(),
    pandas_inferred_dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
    null_counts=null_counts,
    flag_cardinality=flag_cardinality,
)

proposed_schema = ProposedCanonicalSchema(
    dtypes=proposed_dtypes,
    binary_flags=binary_flags,
)

report = SourceInspectionReport(
    report_version=1,
    created_at_utc=datetime.now(timezone.utc).isoformat(),
    source=source_meta,
    observed_schema=observed_schema,
    proposed_canonical_schema=proposed_schema,
)

# %%
from ml_template.paths import ARTIFACTS

short_hash = report.source.source_sha256[:12]
artifact_path = ARTIFACTS / "inspection" / f"{source_path.stem}__sha256-{short_hash}.json"
report.save(artifact_path)
print(f"Inspection report saved to: {artifact_path}")

