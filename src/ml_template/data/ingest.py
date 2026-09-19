"""Ingest raw SECOM files into SQLite (bronze layer).

Raw copy only: type coercion, no value transforms. Labels stay -1/1;
encoding happens downstream. Registers every raw column in
column_registry. Convention: whoever creates a column registers it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import (
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

from ml_template.data import schema
from ml_template.paths import DATA_RAW, LOGS
from ml_template.tracking.utils import setup_logging

logger = logging.getLogger(__name__)


def file_sha256(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without parsing or loading it all."""
    hasher = hashlib.sha256()

    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def setup_db(engine: Engine) -> None:
    metadata = MetaData()
    Table(
        "sensor_readings",
        metadata,
        Column(schema.KEY_COL, Integer, primary_key=True),
        *[Column(f"{schema.SENSOR_PREFIX}{i:03d}", Float) for i in range(1, schema.N_SENSORS + 1)],
    )
    Table(
        "wafer_labels",
        metadata,
        Column(schema.KEY_COL, Integer, primary_key=True),
        Column(schema.TARGET_COL, Integer),
        Column(schema.TIME_COL, DateTime),
    )
    Table(
        "column_registry",
        metadata,
        Column("column_name", String, primary_key=True),
        Column("role", String),
        Column("status", String),
        Column("col_index", Integer),
        Column("missing_pct", Float),
        Column("derived_from", String),
        Column("notes", String),
    )
    Table(
        "ingestion_log",
        metadata,
        Column("ingest_id", Integer, primary_key=True),
        Column("load_ts", DateTime),
        Column("source_file", String),
        Column("source_sha256", String),
        Column("table_name", String),
        Column("rows_inserted", Integer),
        Column("git_sha", String),
    )
    metadata.create_all(engine)


def load_data(data_config):
    """Load raw files. Errors propagate — a failed load must die loudly."""
    try:
        source_path = Path(data_config["data"]["source"])
        data_hash = sha256_file(source_path)

    except Exception as e:
        print(f'Something went wrong getting data sha: {e}')

    data_name = data_config['data']['adapter']
    inspection_report_name = f'{data_name}__sha256-{data_hash[:12]}.json'
    report_path = (Path(f'artifacts/inspection/{inspection_report_name}'))

    try:
        with report_path.open("r", encoding="utf-8") as file:
            inspection_report = json.load(file)
    except Exception as e:
        print(f'Something went wrong while reading the inspection report. {e}')

    encoding = inspection_report['source']['encoding']
    delimiter = inspection_report['source']['delimiter']
    source_path = Path(inspection_report['source']['source_path'])

    try:
        dfX = pd.read_csv(
            source_path,
            sep=delimiter,
            encoding=encoding,
            na_values=["NaN"],
        )
    except Exception as e:
        print(f'Something went wrong while loading data. {e}')

    assert dfX.shape[0] == data_config["bronze_source_contract"]["rows"]
    assert dfX.shape[1] == data_config["bronze_source_contract"]["columns"]
    assert set(dfX.columns.to_list()) == set(data_config['bronze_source_contract']['col_names'])

    return dfX


def build_registry(dfX: pd.DataFrame) -> pd.DataFrame:
    """Register raw columns. Status changes and engineered columns are
    registered later by the modules that create them."""
    sensors = [c for c in dfX.columns if c != schema.KEY_COL]
    rows = [
        {
            "column_name": schema.KEY_COL,
            "role": schema.Role.KEY.value,
            "status": schema.Status.ACTIVE.value,
            "col_index": 0,
            "missing_pct": 0.0,
            "derived_from": None,
            "notes": "row order in raw file is chronological",
        },
        {
            "column_name": schema.TARGET_COL,
            "role": schema.Role.TARGET.value,
            "status": schema.Status.ACTIVE.value,
            "col_index": None,
            "missing_pct": 0.0,
            "derived_from": None,
            "notes": "raw -1/1; encode downstream",
        },
        {
            "column_name": schema.TIME_COL,
            "role": schema.Role.METADATA.value,
            "status": schema.Status.ACTIVE.value,
            "col_index": None,
            "missing_pct": 0.0,
            "derived_from": None,
            "notes": None,
        },
    ]
    miss = dfX[sensors].isna().mean()
    rows += [
        {
            "column_name": s,
            "role": schema.Role.FEATURE_RAW.value,
            "status": schema.Status.ACTIVE.value,
            "col_index": int(s[1:]),
            "missing_pct": float(miss[s]),
            "derived_from": None,
            "notes": None,
        }
        for s in sensors
    ]
    return pd.DataFrame(rows)


def insert_data(dfX, dfy, registry, engine: Engine, data_dir=DATA_RAW) -> None:
    log = pd.DataFrame(
        [
            {
                "load_ts": datetime.now(UTC),
                "source_file": "secom.data",
                "source_sha256": file_sha256(data_dir / "secom.data"),
                "table_name": "sensor_readings",
                "rows_inserted": len(dfX),
                "git_sha": git_sha(),
            },
            {
                "load_ts": datetime.now(UTC),
                "source_file": "secom_labels.data",
                "source_sha256": file_sha256(data_dir / "secom_labels.data"),
                "table_name": "wafer_labels",
                "rows_inserted": len(dfy),
                "git_sha": git_sha(),
            },
        ]
    )
    with engine.begin() as conn:  # one transaction: all-or-nothing
        for table in ("sensor_readings", "wafer_labels", "column_registry"):
            conn.execute(text(f"DELETE FROM {table}"))
        dfX.to_sql("sensor_readings", conn, if_exists="append", index=False)
        dfy.to_sql("wafer_labels", conn, if_exists="append", index=False)
        registry.to_sql("column_registry", conn, if_exists="append", index=False)
        log.to_sql("ingestion_log", conn, if_exists="append", index=False)
    logger.info(
        "ingested %d readings, %d labels, %d registry rows",
        len(dfX),
        len(dfy),
        len(registry),
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Load input data contract")
    p.add_argument(
        "--data_config",
        type=str,
        default=None,
        help="name of config file for data to be ingested",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:

    args = parse_args(argv)

    global logger
    logger = setup_logging(logfile=LOGS / "db_ingest.log")
    logger.info("[db_ingest] start")

    #engine = get_engine()
    #setup_db(engine)

    conf_path = Path(args.data_config)
    with conf_path.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)
    dfX = load_data(data_config=data_config)

    # registry = build_registry(dfX)
    # insert_data(dfX, dfy, registry, engine)
    logger.info("[db_ingest] done")


if __name__ == "__main__":
    main()
