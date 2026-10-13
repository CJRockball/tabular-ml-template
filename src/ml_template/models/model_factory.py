"""Build model adapters from a run config. The only place that knows model types.

Adding a neural network later means one new adapter module and one branch here.
Training and evaluation code depend only on the ``ModelAdapter`` interface.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ml_template.training.run_config import RunConfig


class ModelAdapter(Protocol):
    max_rounds: int

    def fit_with_validation(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> int: ...

    def refit(self, X: pd.DataFrame, y: pd.Series, n_rounds: int) -> None: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    def save(self, path: Path) -> None: ...


AdapterFactory = Callable[[], ModelAdapter]


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Give text columns the category dtype once, on the full frame.

    Done before any slicing so every fold shares one category mapping.
    """
    out = X.copy()
    for col in out.columns:
        series = out[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            out[col] = series.astype("category")
    return out


def build_adapter_factory(cfg: RunConfig) -> AdapterFactory:
    """Return a callable that makes a fresh, unfitted adapter for this config."""
    if cfg.model.type == "xgb":
        from ml_template.models.xgb_classifier import XGBAdapter

        return lambda: XGBAdapter(cfg.model.params, cfg.seed)
    raise ValueError(f"unknown model type {cfg.model.type!r}")


def load_adapter(cfg: RunConfig, path: Path) -> ModelAdapter:
    if cfg.model.type == "xgb":
        from ml_template.models.xgb_classifier import XGBAdapter

        return XGBAdapter.load(cfg.model.params, cfg.seed, path)
    raise ValueError(f"unknown model type {cfg.model.type!r}")
