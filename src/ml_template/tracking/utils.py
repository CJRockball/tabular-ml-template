"""Logging configuration and utilities for pipeline tracking."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from ml_template.paths import LOGS

DEFAULT_APP_LOG = LOGS / "app" / "pipeline.log"
BASE_LOGGER_NAME = "ml_template"


def setup_logging(
    level: int = logging.INFO,
    logfile: str | Path | None = None,
    run_logfile: str | Path | None = None,
) -> logging.Logger:
    """Configure unified logging for the entire ml_template package hierarchy.

    Attaches console and file handlers to the top-level 'ml_template' logger
    so messages from any submodule (data.ingest, training, etc.) flow into:
      1. Standard output (console)
      2. The centralized overview log (logs/app/pipeline.log by default)
      3. An optional run-scoped log (logs/runs/<run_id>.log)

    Sub-modules should simply invoke `logger = logging.getLogger(__name__)`.
    """
    base_logger = logging.getLogger(BASE_LOGGER_NAME)
    base_logger.setLevel(level)
    base_logger.propagate = False  # Avoid duplicate printing to the Python root logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Console stream handler (only add once)
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in base_logger.handlers
    )
    if not has_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(level)
        base_logger.addHandler(ch)

    # Collect existing FileHandler target paths
    existing_files = {
        Path(h.baseFilename).resolve()
        for h in base_logger.handlers
        if isinstance(h, logging.FileHandler)
    }

    # 2. Centralized pipeline overview log
    target_app_log = Path(logfile or DEFAULT_APP_LOG).resolve()
    target_app_log.parent.mkdir(parents=True, exist_ok=True)
    if target_app_log not in existing_files:
        app_fh = logging.FileHandler(target_app_log, mode="a", encoding="utf-8")
        app_fh.setFormatter(fmt)
        app_fh.setLevel(level)
        base_logger.addHandler(app_fh)
        existing_files.add(target_app_log)

    # 3. Optional run-specific log
    if run_logfile is not None:
        target_run_log = Path(run_logfile).resolve()
        target_run_log.parent.mkdir(parents=True, exist_ok=True)
        if target_run_log not in existing_files:
            run_fh = logging.FileHandler(target_run_log, mode="a", encoding="utf-8")
            run_fh.setFormatter(fmt)
            run_fh.setLevel(level)
            base_logger.addHandler(run_fh)

    return base_logger


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
