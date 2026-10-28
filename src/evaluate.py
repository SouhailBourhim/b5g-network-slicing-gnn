"""Single results table for every model (L6).

Reads results/preds/*.parquet (+ events if present) and writes results/metrics_all.csv:
one row per (model, seed, split, subset, slice_type), subset in {all, event, non_event}.
Every number of the report's comparison table comes from this file.

Usage: python -m src.evaluate
"""
from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR
from src.contracts import load_predictions, load_events
from src.metrics import metrics_by_group
from src.splits import load_splits


def evaluation_frame() -> pd.DataFrame:
    """Predictions joined with targets, slice type, split and event flag."""
    preds = load_predictions()
    if preds.empty:
        raise SystemExit("no prediction file in results/preds/")
    truth = pd.read_parquet(PROCESSED_DIR / "flow_features.parquet",
                            columns=["scenario_id", "flow_id", "slice_type", "log_delay", "has_delay"])
    truth = truth[truth.has_delay].rename(columns={"log_delay": "y"})
    df = preds.merge(truth, on=["scenario_id", "flow_id"], how="inner") \
              .merge(load_splits()[["scenario_id", "split"]], on="scenario_id")
    df = df.rename(columns={"pred_log10_delay": "pred"})
    ev = load_events()
    if ev is not None:
        df = df.merge(ev[["scenario_id", "flow_id", "is_event"]], on=["scenario_id", "flow_id"], how="left")
    return df


def evaluate() -> pd.DataFrame:
    df = evaluation_frame()
    rows = []
    subsets = {"all": df}
    if "is_event" in df:
        subsets["event"] = df[df.is_event == True]       # noqa: E712 (NaN-safe)
        subsets["non_event"] = df[df.is_event == False]  # noqa: E712
    for name, part in subsets.items():
        m = metrics_by_group(part, ["model", "seed", "split", "slice_type"], add_all=False)
        m_all = metrics_by_group(part, ["model", "seed", "split"], add_all=False).assign(slice_type="ALL")
        rows.append(pd.concat([m_all, m]).assign(subset=name))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(RESULTS_DIR / "metrics_all.csv", index=False)
    return out


if __name__ == "__main__":
    res = evaluate()
    view = res[(res.subset == "all") & (res.slice_type == "ALL")]
    print(view.groupby(["model", "split"])[["mae_log", "mse_log"]].agg(["mean", "std"]).round(4).to_string())
