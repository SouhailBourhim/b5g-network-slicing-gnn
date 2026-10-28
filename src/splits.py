"""Frozen data splits shared by every model (see JOURNAL D2.1-D2.3).

Unit of split = the scenario (never the flow): flows of one scenario share links,
so a random flow split would put near-duplicates in train and test.

  1. test_unseen_topo : ~15 % of the topologies, drawn inside each size quartile
                        (S/M/L/XL by number of core nodes) -> all their scenarios.
  2. train/val/test   : the remaining scenarios, 70/15/15, grouped by scenario and
                        stratified by load (tercile of the scenario's max rho).

Usage: python -m src.splits      -> data/processed/splits.parquet, results/splits.csv,
                                     results/splits_summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR, SEED

SPLITS = ["train", "val", "test", "test_unseen_topo"]


def scenario_table(processed_dir=PROCESSED_DIR) -> pd.DataFrame:
    """One row per scenario with the two stratification variables."""
    scen = pd.read_parquet(processed_dir / "scenarios.parquet",
                           columns=["scenario_id", "topology", "n_core_nodes", "n_flows"])
    rho = pd.read_parquet(processed_dir / "flow_features.parquet", columns=["scenario_id", "rho_link_max"])
    scen = scen.merge(rho.groupby("scenario_id").rho_link_max.max().rename("scen_rho_max").reset_index(),
                      on="scenario_id")
    return scen


def holdout_topologies(scen: pd.DataFrame, frac: float = 0.15, seed: int = SEED) -> list[str]:
    """Pick ~frac of the topologies inside each size quartile (small to very large)."""
    topo = scen.groupby("topology").n_core_nodes.first().reset_index()
    topo["size_bin"] = pd.qcut(topo.n_core_nodes, 4, labels=["S", "M", "L", "XL"])
    rng = np.random.default_rng(seed)
    held = []
    for _, part in topo.sort_values("topology").groupby("size_bin", observed=True):
        k = max(1, round(frac * len(part)))
        held += rng.choice(part.topology.to_numpy(), size=k, replace=False).tolist()
    return sorted(held)


def grouped_split(scen: pd.DataFrame, fracs=(0.70, 0.15, 0.15), seed: int = SEED) -> pd.Series:
    """Assign whole scenarios to train/val/test, stratified by load tercile."""
    load_bin = pd.qcut(scen.scen_rho_max, 3, labels=["low", "mid", "high"])
    rng = np.random.default_rng(seed)
    out = pd.Series(index=scen.index, dtype=object)
    for _, idx in scen.groupby(load_bin, observed=True).groups.items():   # one load tercile at a time
        idx = rng.permutation(np.sort(np.asarray(idx)))                    # sorted first -> reproducible
        n_tr = round(fracs[0] * len(idx))
        n_va = round(fracs[1] * len(idx))
        out[idx[:n_tr]] = "train"
        out[idx[n_tr:n_tr + n_va]] = "val"
        out[idx[n_tr + n_va:]] = "test"
    return out


def make_splits(seed: int = SEED, holdout_frac: float = 0.15, processed_dir=PROCESSED_DIR) -> pd.DataFrame:
    scen = scenario_table(processed_dir).sort_values("scenario_id").reset_index(drop=True)
    held = holdout_topologies(scen, holdout_frac, seed)
    scen["split"] = "test_unseen_topo"
    pool = ~scen.topology.isin(held)
    scen.loc[pool, "split"] = grouped_split(scen[pool], seed=seed)
    return scen[["scenario_id", "topology", "n_core_nodes", "scen_rho_max", "split"]]


def load_splits(processed_dir=PROCESSED_DIR) -> pd.DataFrame:
    """What every model uses: scenario_id -> split."""
    return pd.read_parquet(processed_dir / "splits.parquet")


def summarize(splits: pd.DataFrame, processed_dir=PROCESSED_DIR) -> dict:
    f = pd.read_parquet(processed_dir / "flow_features.parquet", columns=["scenario_id", "slice_type"])
    f = f.merge(splits[["scenario_id", "split"]], on="scenario_id")
    return {
        "n_scenarios": splits.split.value_counts().reindex(SPLITS).to_dict(),
        "frac_scenarios": (splits.split.value_counts(normalize=True).reindex(SPLITS)).round(4).to_dict(),
        "n_topologies": splits.groupby("split").topology.nunique().reindex(SPLITS).to_dict(),
        "heldout_topologies": sorted(splits[splits.split == "test_unseen_topo"].topology.unique().tolist()),
        "n_flows_by_slice": f.groupby(["split", "slice_type"]).size().unstack().reindex(SPLITS).to_dict(orient="index"),
        "scen_rho_max_median": splits.groupby("split").scen_rho_max.median().reindex(SPLITS).round(3).to_dict(),
        "core_nodes_median": splits.groupby("split").n_core_nodes.median().reindex(SPLITS).to_dict(),
    }


def fit_scaler_on_train(X: pd.DataFrame, is_train: np.ndarray):
    """StandardScaler fitted on the training rows only (no information from val/test)."""
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit(X[is_train])


def main():
    splits = make_splits()
    splits.to_parquet(PROCESSED_DIR / "splits.parquet", index=False)
    RESULTS_DIR.mkdir(exist_ok=True)
    splits.to_csv(RESULTS_DIR / "splits.csv", index=False)
    s = summarize(splits)
    with open(RESULTS_DIR / "splits_summary.json", "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=1, default=str)
    print(json.dumps({k: s[k] for k in ("n_scenarios", "frac_scenarios", "n_topologies", "scen_rho_max_median")}, indent=1))


if __name__ == "__main__":
    main()
