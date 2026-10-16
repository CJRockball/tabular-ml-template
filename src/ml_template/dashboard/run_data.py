"""Loaders for the single-run page. Reads project.db and run folders; no Dash or Plotly."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text

from ml_template.paths import ARTIFACTS_RUNS, DATA_SILVER

RUN_COLUMNS = (
    "run_id, experiment_name, experiment_run_id, seed, feature_set_id, model_type, silver_dataset_version_id, "
    "primary_metric, primary_value, n_rounds_final, n_features, n_rows, duration_s, "
    "started_at, finished_at, holdout_evaluated_at, artifact_dir, split_id, git_sha, git_dirty"
)


def _query(engine: Engine, sql: str, **params) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def list_runs(
    engine: Engine, experiment_name: str | None = None, status: str = "succeeded"
) -> pd.DataFrame:
    """Runs, newest first. ISO UTC strings sort correctly as text."""
    sql = f"SELECT {RUN_COLUMNS} FROM model_runs WHERE status = :status"
    params: dict = {"status": status}
    if experiment_name:
        sql += " AND experiment_name = :name"
        params["name"] = experiment_name
    return _query(engine, sql + " ORDER BY started_at DESC", **params)


def latest_run_id(engine: Engine, experiment_name: str | None = None) -> str | None:
    runs = list_runs(engine, experiment_name)
    return None if runs.empty else str(runs["run_id"].iloc[0])


def run_header(engine: Engine, run_id: str) -> dict:
    df = _query(engine, f"SELECT {RUN_COLUMNS} FROM model_runs WHERE run_id = :r", r=run_id)
    if df.empty:
        raise KeyError(f"unknown run_id: {run_id}")
    return df.iloc[0].to_dict()


def load_params(engine: Engine, run_id: str) -> dict[str, str]:
    df = _query(engine, "SELECT key, value FROM run_params WHERE run_id = :r", r=run_id)
    return dict(zip(df["key"], df["value"], strict=True))


def load_metrics(engine: Engine, run_id: str) -> pd.DataFrame:
    """Columns: split, fold, metric, value. Fold -1 is pooled."""
    return _query(
        engine,
        "SELECT split, fold, metric, value FROM run_metrics WHERE run_id = :r",
        r=run_id,
    )


def load_importance(engine: Engine, run_id: str, fold: int = -1) -> pd.DataFrame:
    """Columns: method, feature, value, std. Fold -1 is the mean over folds."""
    return _query(
        engine,
        "SELECT method, feature, value, std FROM run_feature_importance "
        "WHERE run_id = :r AND fold = :f ORDER BY value DESC",
        r=run_id,
        f=fold,
    )


def load_labels(engine: Engine, header: dict, silver_root: Path = DATA_SILVER) -> pd.Series | None:
    """Target indexed by entity key, from the run's Silver parquet. Key and target names
    come from the splits table, so nothing here is dataset specific."""
    split = _query(
        engine,
        "SELECT key_col, target_col FROM splits WHERE split_id = :s",
        s=header["split_id"],
    )
    if split.empty:
        return None
    key, target = split.iloc[0]["key_col"], split.iloc[0]["target_col"]
    if not key or not target:
        return None
    silver_id = header["silver_dataset_version_id"]
    matches = sorted(Path(silver_root).glob(f"*/{silver_id}/data.parquet"))
    if not matches:
        return None
    df = pd.read_parquet(matches[0], columns=[key, target])
    return df.set_index(key)[target].astype(int)


def load_oof_with_labels(
    engine: Engine, header: dict, folder: Path, silver_root: Path = DATA_SILVER
) -> pd.DataFrame | None:
    """OOF predictions plus a y_true column; None if predictions or labels are missing."""
    oof = load_oof(folder)
    labels = load_labels(engine, header, silver_root)
    if oof is None or labels is None or labels.index.name not in oof.columns:
        return None
    out = oof.copy()
    out["y_true"] = labels.reindex(out[labels.index.name]).to_numpy()
    return out.dropna(subset=["y_true"]).astype({"y_true": int})


def run_dir(header: dict, base: Path = ARTIFACTS_RUNS) -> Path:
    stored = header.get("artifact_dir")
    if stored and Path(stored).is_dir():
        return Path(stored)
    return base / str(header["run_id"])


def read_artifact(folder: Path, name: str) -> pd.DataFrame | None:
    """CSV or Parquet by suffix; None when the file was not written for this run."""
    path = folder / name
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_learning_curves(folder: Path) -> pd.DataFrame | None:
    """Columns: fold, round, dataset, metric, value, best_round."""
    return read_artifact(folder, "learning_curves.parquet")


def load_cv_folds(folder: Path) -> pd.DataFrame | None:
    return read_artifact(folder, "cv_folds.csv")


def load_error_sample(folder: Path) -> pd.DataFrame | None:
    return read_artifact(folder, "error_sample.csv")


def load_oof(folder: Path) -> pd.DataFrame | None:
    return read_artifact(folder, "oof_predictions.parquet")


def load_config(folder: Path) -> dict | None:
    path = folder / "config.json"
    return json.loads(path.read_text()) if path.exists() else None
