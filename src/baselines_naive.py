"""Naive baseline: predict, for each flow, the TRAIN median of log10(delay) of its slice type.

It is the floor every model must beat, and the reference example of the prediction
contract (src/contracts.py). Usage: python -m src.baselines_naive
"""
import pandas as pd

from src.config import PROCESSED_DIR
from src.contracts import save_predictions
from src.splits import load_splits


def main():
    f = pd.read_parquet(PROCESSED_DIR / "flow_features.parquet",
                        columns=["scenario_id", "flow_id", "slice_type", "log_delay", "has_delay"])
    f = f[f.has_delay].merge(load_splits()[["scenario_id", "split"]], on="scenario_id")
    median_by_slice = f[f.split == "train"].groupby("slice_type").log_delay.median()   # fitted on train only
    f["pred_log10_delay"] = f.slice_type.map(median_by_slice)
    print("train medians (log10 s):", median_by_slice.round(3).to_dict())
    print("written:", save_predictions(f[f.split != "train"], "naive_median_slice", seed=0))


if __name__ == "__main__":
    main()
