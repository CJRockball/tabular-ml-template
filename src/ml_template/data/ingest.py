"""Ingest raw data files into SQLite (bronze layer).

Raw copy with lineage: adds dataset_version_id and source_row_number.
Registers every raw column in column_registry and logs ingestion event
in ingestion_log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    text,
)

from ml_template.data.schema import Role, Status
from ml_template.db.connection import get_engine
from ml_template.paths import ARTIFACTS, LOGS
from ml_template.tracking.utils import setup_logging

logger = logging.getLogger(__name__)

# Map proposed string types to SQLAlchemy column types
TYPE_MAP = {
    "Int8": Integer,
    "int8": Integer,
    "Int64": Integer,
    "int64": Integer,
    "float64": Float,
    "float32": Float,
    "str": String,
    "object": String,
    "bool": Boolean,
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def git_sha() -> str:
    """Return the current Git commit hash or 'unknown'."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def setup_db(engine: Engine, inspection_report: dict) -> None:
    """Create bronze tables if they do not already exist."""
    metadata = MetaData()
    proposed_dtypes = inspection_report.get("proposed_canonical_schema", {}).get("dtypes", {})

    # 1. Dynamically construct raw_records columns
    raw_columns = [
        Column("dataset_version_id", String, primary_key=True),
        Column("source_row_number", Integer, primary_key=True),
    ]

    for col_name, dtype_str in proposed_dtypes.items():
        sql_type = TYPE_MAP.get(dtype_str, String)
        raw_columns.append(Column(col_name, sql_type))

    Table("raw_records", metadata, *raw_columns)

    # 2. Column registry
    Table(
        "column_registry",
        metadata,
        Column("dataset_version_id", String, primary_key=True),
        Column("column_name", String, primary_key=True),
        Column("canonical_column_name", String),
        Column("source_position", Integer),
        Column("observed_parse_dtype", String),
        Column("proposed_storage_dtype", String),
        Column("semantic_role", String),
        Column("is_required", Boolean),
        Column("is_nullable", Boolean),
        Column("description", String),
        Column("created_at", DateTime),
    )

    # 3. Ingestion log
    Table(
        "ingestion_log",
        metadata,
        Column("ingest_id", Integer, primary_key=True, autoincrement=True),
        Column("dataset_version_id", String, unique=True),
        Column("source_name", String),
        Column("source_path_or_uri", String),
        Column("source_sha256", String),
        Column("inspection_report_path", String),
        Column("inspection_report_sha256", String),
        Column("row_count_read", Integer),
        Column("row_count_written", Integer),
        Column("columns_count", Integer),
        Column("encoding", String),
        Column("delimiter", String),
        Column("started_at", DateTime),
        Column("completed_at", DateTime),
        Column("status", String),
        Column("error_message", String),
        Column("git_sha", String),
    )

    metadata.create_all(engine)


def load_data(data_config: dict) -> tuple[pd.DataFrame, dict, Path, str]:
    """Load raw CSV and corresponding inspection report. Errors propagate loudly."""
    source_path = Path(data_config["data"]["source"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    data_hash = sha256_file(source_path)
    data_name = data_config["data"].get("adapter", source_path.stem)
    inspection_report_name = f"{data_name}__sha256-{data_hash[:12]}.json"
    report_path = ARTIFACTS / "inspection" / inspection_report_name

    if not report_path.is_file():
        raise FileNotFoundError(
            f"Inspection report not found at {report_path}. Run source inspection first."
        )

    with report_path.open("r", encoding="utf-8") as file:
        inspection_report = json.load(file)

    report_hash = sha256_file(report_path)
    encoding = inspection_report["source"]["encoding"]
    delimiter = inspection_report["source"]["delimiter"]

    dfX = pd.read_csv(
        source_path,
        sep=delimiter,
        encoding=encoding,
        na_values=["NaN"],
    )

    # Contract validation checks
    contract = data_config.get("bronze_source_contract", {})
    if "rows" in contract:
        assert len(dfX) == contract["rows"], (
            f"Row count mismatch: expected {contract['rows']}, got {len(dfX)}"
        )
    if "columns" in contract:
        assert dfX.shape[1] == contract["columns"], (
            f"Column count mismatch: expected {contract['columns']}, got {dfX.shape[1]}"
        )
    if "col_names" in contract:
        expected_cols = set(contract["col_names"])
        actual_cols = set(dfX.columns)
        assert actual_cols == expected_cols, (
            f"Header mismatch. Missing: {expected_cols - actual_cols}, Extra: {actual_cols - expected_cols}"
        )

    return dfX, inspection_report, report_path, report_hash


def build_registry(
    df: pd.DataFrame,
    inspection_report: dict,
    dataset_version_id: str,
    contract: dict | None = None,
) -> pd.DataFrame:
    """Register raw columns and their metadata for tracking and lineage."""
    contract = contract or {}
    observed = inspection_report.get("observed_schema", {})
    observed_dtypes = observed.get("pandas_inferred_dtypes", {})
    null_counts = observed.get("null_counts", {})

    proposed = inspection_report.get("proposed_canonical_schema", {})
    proposed_dtypes = proposed.get("dtypes", {})

    semantic_roles = contract.get("semantic_roles", {})
    canonical_names = contract.get("canonical_names", {})
    required_cols = set(contract.get("required_columns", df.columns.tolist()))
    descriptions = contract.get("column_descriptions", {})

    now_utc = datetime.now(timezone.utc)
    rows = []

    for idx, col in enumerate(df.columns):
        is_nullable = null_counts.get(col, 0) > 0
        role_val = semantic_roles.get(col, Role.UNKNOWN.value)
        if isinstance(role_val, Role):
            role_val = role_val.value

        row = {
            "dataset_version_id": dataset_version_id,
            "column_name": col,
            "canonical_column_name": canonical_names.get(col, col.lower().replace(" ", "_")),
            "source_position": idx,
            "observed_parse_dtype": observed_dtypes.get(col, str(df[col].dtype)),
            "proposed_storage_dtype": proposed_dtypes.get(col, "str"),
            "semantic_role": role_val,
            "is_required": col in required_cols,
            "is_nullable": is_nullable,
            "description": descriptions.get(col, None),
            "created_at": now_utc,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def insert_data(
    dfX: pd.DataFrame,
    registry: pd.DataFrame,
    engine: Engine,
    data_config: dict,
    inspection_report: dict,
    report_path: Path,
    report_hash: str,
    dataset_version_id: str,
    started_at: datetime,
) -> None:
    """Attach lineage attributes and insert data, registry, and log atomically."""
    # 1. Prepare raw_records by adding lineage columns
    df_to_insert = dfX.copy()
    df_to_insert.insert(0, "source_row_number", range(1, len(df_to_insert) + 1))
    df_to_insert.insert(0, "dataset_version_id", dataset_version_id)

    # 2. Build single-row ingestion log dataframe
    completed_at = datetime.now(timezone.utc)
    source_path_val = str(data_config["data"]["source"])
    source_name_val = Path(source_path_val).name

    log_entry = {
        "dataset_version_id": dataset_version_id,
        "source_name": source_name_val,
        "source_path_or_uri": source_path_val,
        "source_sha256": inspection_report["source"]["source_sha256"],
        "inspection_report_path": str(report_path),
        "inspection_report_sha256": report_hash,
        "row_count_read": len(dfX),
        "row_count_written": len(df_to_insert),
        "columns_count": dfX.shape[1],
        "encoding": inspection_report["source"]["encoding"],
        "delimiter": inspection_report["source"]["delimiter"],
        "started_at": started_at,
        "completed_at": completed_at,
        "status": Status.SUCCEEDED.value,
        "error_message": None,
        "git_sha": git_sha(),
    }
    log_df = pd.DataFrame([log_entry])

    # 3. Transactional atomic write (delete existing records for THIS version if re-running)
    with engine.begin() as conn:
        for tbl in ("raw_records", "column_registry", "ingestion_log"):
            conn.execute(
                text(f"DELETE FROM {tbl} WHERE dataset_version_id = :dvid"),
                {"dvid": dataset_version_id},
            )

        df_to_insert.to_sql("raw_records", conn, if_exists="append", index=False)
        registry.to_sql("column_registry", conn, if_exists="append", index=False)
        log_df.to_sql("ingestion_log", conn, if_exists="append", index=False)

    logger.info(
        "Successfully ingested %d rows, %d columns into raw_records (version: %s)",
        len(df_to_insert),
        dfX.shape[1],
        dataset_version_id,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ingest raw data into bronze SQLite database")
    p.add_argument(
        "--data_config",
        type=str,
        required=True,
        help="Path to YAML config file for data to be ingested",
    )
    p.add_argument(
        "--dataset_version_id",
        type=str,
        default="ds_v1",
        help="Identifier for this dataset version (default: ds_v1)",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(logfile=LOGS / "db_ingest.log")
    logger.info("[db_ingest] Started ingestion")

    started_at = datetime.now(timezone.utc)

    conf_path = Path(args.data_config)
    with conf_path.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    dfX, inspection_report, report_path, report_hash = load_data(data_config=data_config)

    engine = get_engine()
    setup_db(engine, inspection_report)

    contract = data_config.get("semantic_metadata", {})
    registry = build_registry(
        df=dfX,
        inspection_report=inspection_report,
        dataset_version_id=args.dataset_version_id,
        contract=contract,
    )

    insert_data(
        dfX=dfX,
        registry=registry,
        engine=engine,
        data_config=data_config,
        inspection_report=inspection_report,
        report_path=report_path,
        report_hash=report_hash,
        dataset_version_id=args.dataset_version_id,
        started_at=started_at,
    )
    logger.info("[db_ingest] Completed successfully")


if __name__ == "__main__":
    main()