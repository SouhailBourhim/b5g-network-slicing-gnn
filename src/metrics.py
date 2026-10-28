"""Evaluation metrics, shared by every model of the project.

All models predict y = log10(delay [s]). On that scale:
  MAE_log = mean |y - y_hat|      -> 0.1 means "off by a factor 10^0.1 = 1.26 on average"
  MSE_log = mean (y - y_hat)^2    -> penalises large misses (e.g. a starved flow predicted fast)
  RMSE_log = sqrt(MSE_log)
An error in log10 is a *relative* error on the delay (see L1: 8 orders of magnitude).
MdAPE = median |d - d_hat| / d on the delay in seconds, for a reading in percent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_log_true, y_log_pred) -> dict:
    """MAE / MSE / RMSE on log10(delay) and MdAPE on the delay in seconds."""
    y = np.asarray(y_log_true, dtype=float)
    p = np.asarray(y_log_pred, dtype=float)
    err = p - y
    ape = np.abs(10.0 ** p - 10.0 ** y) / 10.0 ** y          # relative error on seconds
    return {"n": int(y.size),
            "mae_log": float(np.mean(np.abs(err))),
            "mse_log": float(np.mean(err ** 2)),
            "rmse_log": float(np.sqrt(np.mean(err ** 2))),
            "mdape_pct": float(np.median(ape) * 100)}


def metrics_by_group(df: pd.DataFrame, groups: list[str], y_col: str = "y", pred_col: str = "pred",
                     add_all: bool = True) -> pd.DataFrame:
    """One row of metrics per group (e.g. per slice type), plus an 'ALL' row."""
    rows = []
    if add_all:
        rows.append({**{g: "ALL" for g in groups}, **regression_metrics(df[y_col], df[pred_col])})
    for key, part in df.groupby(groups, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(groups, key)), **regression_metrics(part[y_col], part[pred_col])})
    return pd.DataFrame(rows)
