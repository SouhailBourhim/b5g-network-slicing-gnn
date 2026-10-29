"""Load-shift protocol (volet B, B4): train on LOW-load scenarios, test on MID and HIGH load.

Load of a scenario = scen_rho_max (max link utilisation, from inputs; already in splits.parquet).
Cutoffs Q1, Q2 = terciles of scen_rho_max over the ORIGINAL TRAIN scenarios only.
test_unseen_topo scenarios are not used here (the topology shift stays a separate experiment).

  shift_train     original train, load <= Q1        fit everything here (scalers, weights, ...)
  shift_val       original val,   load <= Q1        early stopping, hyper-parameters (low load only!)
  shift_test_id   original test,  load <= Q1        in-distribution reference
  shift_test_mid  train/val/test, Q1 < load <= Q2   mild shift
  shift_test_high train/val/test, load > Q2         strong shift
mid/high scenarios of any original split are legal test sets here: a shift model never trains on them.

Usage: python -m src.shift   -> results/shift_splits.csv, results/shift_summary.json, results/L4_shift_baselines.csv
Model owners: train on shift_train, select on shift_val, score with evaluate_shift(df) (df has y, pred, scenario_id).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR
from src.metrics import metrics_by_group
from src.splits import load_splits

SHIFT_SPLITS = ["shift_train", "shift_val", "shift_test_id", "shift_test_mid", "shift_test_high"]


def make_shift_splits() -> tuple[pd.DataFrame, tuple[float, float]]:
    s = load_splits()
    q1, q2 = s[s.split == "train"].scen_rho_max.quantile([1 / 3, 2 / 3]).tolist()   # train only
    pool = s[s.split.isin(["train", "val", "test"])].copy()
    lab = pd.Series(np.nan, index=pool.index, dtype=object)
    low, mid, high = pool.scen_rho_max <= q1, pool.scen_rho_max.between(q1, q2, inclusive="right"), pool.scen_rho_max > q2
    lab[low & (pool.split == "train")] = "shift_train"
    lab[low & (pool.split == "val")] = "shift_val"
    lab[low & (pool.split == "test")] = "shift_test_id"
    lab[mid] = "shift_test_mid"
    lab[high] = "shift_test_high"
    pool["shift_split"] = lab
    out = pool[["scenario_id", "topology", "scen_rho_max", "split", "shift_split"]].dropna(subset=["shift_split"]).copy()
    # load shift only? flag scenarios whose topology also appears in shift_train (filter on it to isolate load)
    out["topology_in_shift_train"] = out.topology.isin(set(out[out.shift_split == "shift_train"].topology))
    return out, (q1, q2)


def load_shift_splits() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "shift_splits.csv")


def evaluate_shift(df: pd.DataFrame) -> pd.DataFrame:
    """df: scenario_id, y (log10 delay), pred. Returns MAE/MSE per shift split."""
    d = df.drop(columns=["shift_split"], errors="ignore").merge(
        load_shift_splits()[["scenario_id", "shift_split"]], on="scenario_id")
    return metrics_by_group(d, ["shift_split"], add_all=False).set_index("shift_split").reindex(SHIFT_SPLITS).reset_index()


def baselines_demo(sh: pd.DataFrame) -> pd.DataFrame:
    """Naive median and calibrated M/M/1/K fitted on shift_train only, plus parameter-free M/M/1."""
    f = pd.read_parquet(PROCESSED_DIR / "flow_features.parquet",
                        columns=["scenario_id", "flow_id", "slice_type", "log_delay", "has_delay", "mm1_sum", "mm1k_link_sum"])
    f = f[f.has_delay].merge(sh[["scenario_id", "shift_split"]], on="scenario_id")
    tr = f[f.shift_split == "shift_train"]
    med = tr.groupby("slice_type").log_delay.median()
    f["naive_median_slice"] = f.slice_type.map(med)
    f["analytic_mm1"] = np.log10(np.maximum(f.mm1_sum, 1e-12))
    x = np.log10(np.maximum(f.mm1k_link_sum, 1e-12))
    b, a = np.polyfit(x[f.shift_split == "shift_train"], tr.log_delay, deg=1)
    f["analytic_mm1k_link_cal"] = a + b * x
    rows = []
    for m in ["naive_median_slice", "analytic_mm1", "analytic_mm1k_link_cal"]:
        r = evaluate_shift(f.assign(y=f.log_delay, pred=f[m]))
        rows.append(r.assign(model=m))
    out = pd.concat(rows)
    out.attrs["calibration"] = (float(a), float(b))
    return out


def main():
    sh, (q1, q2) = make_shift_splits()
    sh.to_csv(RESULTS_DIR / "shift_splits.csv", index=False)
    fl = pd.read_parquet(PROCESSED_DIR / "flow_features.parquet", columns=["scenario_id"])
    nflows = fl.scenario_id.value_counts()
    sh["n_flows"] = sh.scenario_id.map(nflows)
    g = sh.groupby("shift_split").agg(n_scenarios=("scenario_id", "size"), n_flows=("n_flows", "sum"),
                                      n_topologies=("topology", "nunique"),
                                      rho_min=("scen_rho_max", "min"), rho_median=("scen_rho_max", "median"),
                                      rho_max=("scen_rho_max", "max")).reindex(SHIFT_SPLITS).round(3)
    summary = {"Q1": round(q1, 4), "Q2": round(q2, 4), "groups": g.reset_index().to_dict(orient="records")}
    (RESULTS_DIR / "shift_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"Q1 = {q1:.3f}, Q2 = {q2:.3f}"); print(g.to_string())
    # topologies of shift_test_* seen in shift_train? (load shift should not also be a topology shift)
    seen = set(sh[sh.shift_split == "shift_train"].topology)
    for k in ["shift_test_mid", "shift_test_high"]:
        t = sh[sh.shift_split == k]
        print(f"{k}: {100 * t.topology.isin(seen).mean():.1f}% of scenarios on a topology present in shift_train")
    res = baselines_demo(sh)
    res.to_csv(RESULTS_DIR / "L4_shift_baselines.csv", index=False)
    print("calibration (a, b) on shift_train:", res.attrs["calibration"])
    print(res.pivot(index="model", columns="shift_split", values="mae_log")[SHIFT_SPLITS].round(3).to_string())
    print(res.pivot(index="model", columns="shift_split", values="mse_log")[SHIFT_SPLITS].round(3).to_string())


if __name__ == "__main__":
    main()
