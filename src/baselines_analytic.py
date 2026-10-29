"""Analytic baselines (volet B): queueing theory only, nothing learned (except the optional calibration).

The per-hop estimates are already computed from the INPUTS in src/features.py and
summed along the path in flow_features:
    analytic_mm1        = log10(mm1_sum)          M/M/1, rho clipped at 0.99 (FIFO link view)
    analytic_mm1k_link  = log10(mm1k_link_sum)    M/M/1/K, finite buffer, FIFO link view
    analytic_mm1k_wfq   = log10(mm1k_queue_sum)   M/M/1/K, WFQ guaranteed-rate view
    analytic_mm1k_link_cal = a + b * log10(mm1k_link_sum), (a, b) fitted on TRAIN only

Usage: python -m src.baselines_analytic
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.contracts import save_predictions
from src.splits import load_splits

RAW_MODELS = {"analytic_mm1": "mm1_sum",
              "analytic_mm1k_link": "mm1k_link_sum",
              "analytic_mm1k_wfq": "mm1k_queue_sum"}
EPS = 1e-12   # floor before log10: an estimate of exactly 0 would give -inf


def load_frame() -> pd.DataFrame:
    cols = ["scenario_id", "flow_id", "slice_type", "log_delay", "has_delay", *RAW_MODELS.values()]
    f = pd.read_parquet(PROCESSED_DIR / "flow_features.parquet", columns=cols)
    f = f[f.has_delay].merge(load_splits()[["scenario_id", "split"]], on="scenario_id")
    for name, col in RAW_MODELS.items():
        f[name] = np.log10(np.maximum(f[col].to_numpy(), EPS))
    return f


def calibrate(f: pd.DataFrame, x_col: str) -> tuple[float, float]:
    """Least squares  log_delay = a + b * x  on TRAIN only (returns a, b)."""
    tr = f[f.split == "train"]
    b, a = np.polyfit(tr[x_col].to_numpy(), tr.log_delay.to_numpy(), deg=1)
    return float(a), float(b)


def main():
    f = load_frame()
    out = f[f.split != "train"]
    for name in RAW_MODELS:
        p = save_predictions(out.assign(pred_log10_delay=out[name]), name, seed=0)
        mae = (out[name] - out.log_delay).abs().groupby(out.split).mean().round(3).to_dict()
        print(f"{name:24s} MAE_log {mae}  -> {p}")
    a, b = calibrate(f, "analytic_mm1k_link")
    cal = a + b * out["analytic_mm1k_link"]
    save_predictions(out.assign(pred_log10_delay=cal), "analytic_mm1k_link_cal", seed=0)
    mae = (cal - out.log_delay).abs().groupby(out.split).mean().round(3).to_dict()
    print(f"analytic_mm1k_link_cal   MAE_log {mae}  (a={a:.3f}, b={b:.3f}, fitted on train)")


if __name__ == "__main__":
    main()
