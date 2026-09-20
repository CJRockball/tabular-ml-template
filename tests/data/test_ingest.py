"""Integration and contract tests for the bronze layer ingestion module.

Tests verify bronze table schema creation, contract validation, idempotency,
lineage registration in column_registry, and audit tracking in ingestion_log.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml
from sqlalchemy import create_engine, inspect, text

from ml_template.data.ingest import (
    build_registry,
    insert_data,
    load_data,
    setup_db,
    sha256_file,
)
from ml_template.data.schema import Status
from ml_template.paths import ARTIFACTS, CONFIGS


@pytest.fixture(scope="session")
def base_config_path() -> Path:
    """Resolve the default AI4I binary configuration file."""
    candidate = CONFIGS / "ai4i_binary.yaml"
    if not candidate.is_file():
        candidate = Path("configs/ai4i_binary.yaml")
    if not candidate.is_file():
        pytest.skip(f"Config file not found at {candidate}")
    return candidate


@pytest.fixture(scope="session")
def base_config(base_config_path: Path) -> dict:
    """Load configuration dictionary."""
    with base_config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@pytest.fixture
def test_engine(tmp_path: Path):
    """Provide an isolated, ephemeral SQLite database for test execution."""
    db_file = tmp_path / "test_bronze.db"
    engine = create_engine(f"sqlite:///{db_file}")
    yield engine
    engine.dispose()


@pytest.fixture
def mock_env(tmp_path: Path, base_config: dict):
    """Set up an isolated fixture file, inspection report, and config copy."""
    cfg = copy.deepcopy(base_config)

    # 1. Point source to sample fixture
    fixture_src = Path("tests/fixtures/ai4i_small.csv")
    if not fixture_src.is_file():
        fixture_src = Path("data/raw/ai4i2020.csv")

    test_source_csv = tmp_path / "source.csv"
    test_source_csv.write_bytes(fixture_src.read_bytes())
    cfg["data"]["source"] = str(test_source_csv)

    src_hash = sha256_file(test_source_csv)
    adapter_name = cfg["data"].get("adapter", test_source_csv.stem)
    report_filename = f"{adapter_name}__sha256-{src_hash[:12]}.json"

    # 2. Mock inspection report
    df_sample = pd.read_csv(test_source_csv)
    report_dir = ARTIFACTS / "inspection"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / report_filename

    inspection_data = {
        "report_version": 1,
        "created_at_utc": datetime.now().isoformat(),
        "source": {
            "source_name": test_source_csv.name,
            "source_path": str(test_source_csv),
            "source_sha256": src_hash,
            "encoding": "utf-8",
            "delimiter": ",",
        },
        "observed_schema": {
            "row_count": len(df_sample),
            "column_count": df_sample.shape[1],
            "headers": list(df_sample.columns),
            "pandas_inferred_dtypes": {col: str(df_sample[col].dtype) for col in df_sample.columns},
            "null_counts": {col: int(df_sample[col].isna().sum()) for col in df_sample.columns},
        },
        "proposed_canonical_schema": {
            "dtypes": {
                col: ("int64" if "int" in str(df_sample[col].dtype) else "float64" if "float" in str(df_sample[col].dtype) else "str")
                for col in df_sample.columns
            }
        },
    }

    report_file.write_text(json.dumps(inspection_data, indent=2), encoding="utf-8")

    # 3. Synchronize bronze contract expectations to the fixture
    if "bronze_source_contract" in cfg:
        cfg["bronze_source_contract"]["rows"] = len(df_sample)
        cfg["bronze_source_contract"]["columns"] = df_sample.shape[1]
        cfg["bronze_source_contract"]["col_names"] = list(df_sample.columns)

    yield {
        "config": cfg,
        "source_path": test_source_csv,
        "report_path": report_file,
        "report_data": inspection_data,
        "report_hash": sha256_file(report_file),
    }

    # Clean up generated inspection report artifact
    if report_file.exists():
        report_file.unlink()


def test_bronze_tables_creation(test_engine, mock_env):
    """Verify setup_db creates raw_records, column_registry, and ingestion_log."""
    setup_db(test_engine, mock_env["report_data"])
    inspector = inspect(test_engine)
    tables = set(inspector.get_table_names())
    expected = {"raw_records", "column_registry", "ingestion_log"}
    assert expected.issubset(tables), f"Missing bronze tables: {expected - tables}"


def test_full_ingest_pipeline_execution(test_engine, mock_env):
    """Verify complete ingestion workflow, database writes, and lineage record creation."""
    cfg = mock_env["config"]
    df_raw, report_data, report_path, report_hash = load_data(cfg)

    setup_db(test_engine, report_data)

    contract = cfg.get("semantic_metadata", {})
    version_id = "test_v1"
    started_at = datetime.now()

    registry = build_registry(
        df=df_raw,
        inspection_report=report_data,
        dataset_version_id=version_id,
        contract=contract,
    )

    insert_data(
        dfX=df_raw,
        registry=registry,
        engine=test_engine,
        data_config=cfg,
        inspection_report=report_data,
        report_path=report_path,
        report_hash=report_hash,
        dataset_version_id=version_id,
        started_at=started_at,
    )

    # 1. Assert raw_records contents and lineage columns
    with test_engine.connect() as conn:
        raw_count = conn.execute(
            text("SELECT COUNT(*) FROM raw_records WHERE dataset_version_id = :v"),
            {"v": version_id},
        ).scalar()
        assert raw_count == len(df_raw)

        sample = pd.read_sql(
            text("SELECT * FROM raw_records WHERE dataset_version_id = :v LIMIT 5"),
            conn,
            params={"v": version_id},
        )
        assert "source_row_number" in sample.columns
        assert list(sample["source_row_number"]) == [1, 2, 3, 4, 5]

        # 2. Assert column_registry completeness
        reg_count = conn.execute(
            text("SELECT COUNT(*) FROM column_registry WHERE dataset_version_id = :v"),
            {"v": version_id},
        ).scalar()
        assert reg_count == df_raw.shape[1]

        # 3. Assert ingestion_log entry details
        log_row = conn.execute(
            text("SELECT * FROM ingestion_log WHERE dataset_version_id = :v"),
            {"v": version_id},
        ).mappings().fetchone()
        assert log_row is not None
        assert log_row["status"] == Status.SUCCEEDED.value
        assert log_row["row_count_read"] == len(df_raw)
        assert log_row["row_count_written"] == len(df_raw)
        assert log_row["columns_count"] == df_raw.shape[1]
        assert log_row["source_sha256"] == mock_env["report_data"]["source"]["source_sha256"]


def test_contract_validation_row_mismatch_fails(mock_env):
    """Ensure row count assertion in load_data triggers on contract violation."""
    cfg = mock_env["config"]
    cfg["bronze_source_contract"] = cfg.get("bronze_source_contract", {})
    cfg["bronze_source_contract"]["rows"] = 999_999  # deliberate discrepancy

    with pytest.raises(AssertionError, match="Row count mismatch"):
        load_data(cfg)


def test_contract_validation_column_mismatch_fails(mock_env):
    """Ensure header/column count assertions in load_data trigger on missing column."""
    cfg = mock_env["config"]
    cfg["bronze_source_contract"] = cfg.get("bronze_source_contract", {})
    cfg["bronze_source_contract"]["col_names"] = ["Invalid_Col_1", "Invalid_Col_2"]

    with pytest.raises(AssertionError, match="Header mismatch|Column count mismatch"):
        load_data(cfg)


def test_ingest_idempotency(test_engine, mock_env):
    """Ensure re-running the same dataset_version_id overwrites cleanly without duplication."""
    cfg = mock_env["config"]
    df_raw, report_data, report_path, report_hash = load_data(cfg)
    setup_db(test_engine, report_data)

    version_id = "idempotent_v1"
    registry = build_registry(df_raw, report_data, version_id, cfg.get("semantic_metadata", {}))

    for _ in range(2):
        insert_data(
            dfX=df_raw,
            registry=registry,
            engine=test_engine,
            data_config=cfg,
            inspection_report=report_data,
            report_path=report_path,
            report_hash=report_hash,
            dataset_version_id=version_id,
            started_at=datetime.now(),
        )

    with test_engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM raw_records WHERE dataset_version_id = :v"),
            {"v": version_id},
        ).scalar()
        log_count = conn.execute(
            text("SELECT COUNT(*) FROM ingestion_log WHERE dataset_version_id = :v"),
            {"v": version_id},
        ).scalar()

        assert count == len(df_raw), "Rows were duplicated instead of overwritten"
        assert log_count == 1, "Log entries were duplicated"


def test_source_file_immutability(mock_env):
    """Verify that source CSV remains unaltered and byte-identical after loading."""
    initial_hash = sha256_file(mock_env["source_path"])
    load_data(mock_env["config"])
    post_hash = sha256_file(mock_env["source_path"])
    assert initial_hash == post_hash, "Ingest modified the raw source file in place"
