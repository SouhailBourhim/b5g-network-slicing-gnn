"""Heterogeneous graph per scenario (L5, part 1), RouteNet-style.

Nodes
  flow : one per flow   -> x_flow  (slice type, provisioning, traffic, path length)
  link : one per directed link used by >= 1 flow -> x_link (capacity, load, competition)
Edges
  (flow, traverses, link) and its reverse (link, carries, flow), one per hop.
  edge_attr = what belongs to the PAIR (flow, link): position on the path and the
  WFQ queue the flow uses on that link (guaranteed share, queue load, default queue).

Everything comes from INPUTS (features.py); targets are y = log10(delay) on flow nodes,
with y_mask = False for the 9 flows without a delivered packet.
Continuous features are standardized with statistics computed on TRAIN scenarios only.

Usage: python -m src.graphs     -> data/processed/graphs/{train,val,test,test_unseen_topo}.pt + norm.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.config import PROCESSED_DIR, GRAPHS_DIR
from src.splits import load_splits, SPLITS

FLOW_FEATS = ["is_eMBB", "is_URLLC", "is_mMTC", "delta", "prov_ratio",
              "log_rate", "log_pkt_size", "path_len"]
LINK_FEATS = ["log_capacity", "rho_link", "log_n_flows_link", "log_levels_qos", "log_n_queues_used",
              "log_mm1k_link", "is_access", "is_core", "is_egress"]
EDGE_FEATS = ["hop_rel", "log_wfq_share", "log_rho_queue", "is_default_queue",
              "log_n_flows_queue", "log_mm1k_queue"]
BINARY = {"is_eMBB", "is_URLLC", "is_mMTC", "is_access", "is_core", "is_egress", "is_default_queue"}


# --------------------------------------------------------------------------
# 1. Feature tables (one row per flow / link / hop), still in pandas
# --------------------------------------------------------------------------
def node_and_edge_tables(processed_dir=PROCESSED_DIR):
    f = pd.read_parquet(processed_dir / "flow_features.parquet",
                        columns=["scenario_id", "flow_id", "is_eMBB", "is_URLLC", "is_mMTC", "delta",
                                 "prov_ratio", "rate_bps", "mean_pkt_size", "path_len", "log_delay", "has_delay"])
    f["log_rate"] = np.log10(f.rate_bps)
    f["log_pkt_size"] = np.log10(f.mean_pkt_size)

    L = pd.read_parquet(processed_dir / "link_features.parquet",
                        columns=["scenario_id", "link_id", "src_type", "dst_type", "capacity_bps", "levels_qos",
                                 "n_flows_link", "rho_link", "mm1k_link", "n_queues_used"])
    L["log_capacity"] = np.log10(L.capacity_bps)
    L["log_n_flows_link"] = np.log10(L.n_flows_link)
    L["log_levels_qos"] = np.log10(L.levels_qos)
    L["log_n_queues_used"] = np.log10(L.n_queues_used)
    L["log_mm1k_link"] = np.log10(L.mm1k_link)
    L["is_access"] = L.src_type.str.startswith("origin").astype(np.int8)       # source -> antenna
    L["is_egress"] = L.dst_type.str.startswith("destination").astype(np.int8)  # towards a destination
    L["is_core"] = ((L.is_access == 0) & (L.is_egress == 0)).astype(np.int8)

    h = pd.read_parquet(processed_dir / "hop_features.parquet",
                        columns=["scenario_id", "flow_id", "hop", "link_id", "is_default_queue", "wfq_share",
                                 "rho_queue", "n_flows_queue", "mm1k_queue"])
    h = h.merge(f[["scenario_id", "flow_id", "path_len"]], on=["scenario_id", "flow_id"])
    h["hop_rel"] = h.hop / h.path_len                                   # 0 = first hop
    h["log_wfq_share"] = np.log10(h.wfq_share)
    h["log_rho_queue"] = np.log10(h.rho_queue.clip(1e-4, 1e6))
    h["log_n_flows_queue"] = np.log10(h.n_flows_queue)
    h["log_mm1k_queue"] = np.log10(h.mm1k_queue)
    h["is_default_queue"] = h.is_default_queue.astype(np.int8)

    # local link index inside each scenario (0..n_links-1), in link_id order
    L = L.sort_values(["scenario_id", "link_id"]).reset_index(drop=True)
    L["link_local"] = L.groupby("scenario_id").cumcount()
    h = h.merge(L[["scenario_id", "link_id", "link_local"]], on=["scenario_id", "link_id"], how="left")
    assert h.link_local.notna().all(), "a hop points to a link missing from link_features"
    f = f.sort_values(["scenario_id", "flow_id"]).reset_index(drop=True)
    h = h.sort_values(["scenario_id", "flow_id", "hop"]).reset_index(drop=True)
    return f, L, h


def fit_normalizer(f, L, h, train_ids) -> dict:
    """Mean/std of every continuous feature, on TRAIN scenarios only."""
    stats = {}
    for df, cols in ((f, FLOW_FEATS), (L, LINK_FEATS), (h, EDGE_FEATS)):
        tr = df[df.scenario_id.isin(train_ids)]
        for c in cols:
            if c in BINARY:
                stats[c] = (0.0, 1.0)                                   # leave 0/1 untouched
            else:
                stats[c] = (float(tr[c].mean()), float(tr[c].std()) or 1.0)
    return stats


def apply_normalizer(df: pd.DataFrame, cols: list[str], stats: dict) -> np.ndarray:
    m = np.array([stats[c][0] for c in cols], dtype=np.float32)
    s = np.array([stats[c][1] for c in cols], dtype=np.float32)
    return ((df[cols].to_numpy(dtype=np.float32) - m) / s).astype(np.float32)


# --------------------------------------------------------------------------
# 2. Cut the big arrays into one HeteroData per scenario
# --------------------------------------------------------------------------
def offsets(sorted_ids: np.ndarray, wanted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start/end row of each wanted scenario in an array sorted by scenario_id."""
    return np.searchsorted(sorted_ids, wanted, "left"), np.searchsorted(sorted_ids, wanted, "right")


def build_graphs(f, L, h, stats, scenario_ids) -> list[HeteroData]:
    xf, xl, ea = apply_normalizer(f, FLOW_FEATS, stats), apply_normalizer(L, LINK_FEATS, stats), \
        apply_normalizer(h, EDGE_FEATS, stats)
    y = f.log_delay.fillna(0.0).to_numpy(np.float32)
    ymask = f.has_delay.to_numpy(bool)
    fid = f.flow_id.to_numpy(np.int64)
    h_flow, h_link = h.flow_id.to_numpy(np.int64), h.link_local.to_numpy(np.int64)
    fa, fb = offsets(f.scenario_id.to_numpy(), scenario_ids)
    la, lb = offsets(L.scenario_id.to_numpy(), scenario_ids)
    ha, hb = offsets(h.scenario_id.to_numpy(), scenario_ids)
    # IMPORTANT: torch.tensor() COPIES the slice. torch.from_numpy(a[i:j]) would be a
    # view on the whole array, and torch.save would then write the whole array once
    # per graph (a first version produced a 330 GB file this way).
    graphs = []
    for k, sid in enumerate(scenario_ids):
        g = HeteroData()
        g["flow"].x = torch.tensor(xf[fa[k]:fb[k]])
        g["flow"].y = torch.tensor(y[fa[k]:fb[k]])
        g["flow"].y_mask = torch.tensor(ymask[fa[k]:fb[k]])
        g["flow"].flow_id = torch.tensor(fid[fa[k]:fb[k]])
        g["link"].x = torch.tensor(xl[la[k]:lb[k]])
        # flow_id is 0..n_flows-1 inside a scenario, so it is directly the local flow index
        ei = torch.tensor(np.stack([h_flow[ha[k]:hb[k]], h_link[ha[k]:hb[k]]]))
        e = torch.tensor(ea[ha[k]:hb[k]])
        g["flow", "traverses", "link"].edge_index = ei
        g["flow", "traverses", "link"].edge_attr = e
        g["link", "carries", "flow"].edge_index = ei.flip(0)
        g["link", "carries", "flow"].edge_attr = e
        g.scenario_id = int(sid)
        graphs.append(g)
    return graphs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-scenarios", type=int, default=None, help="per split, for quick tests")
    ap.add_argument("--out-dir", default=str(GRAPHS_DIR))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits()
    f, L, h = node_and_edge_tables()
    # sanity: flow_id is a contiguous 0..n-1 index inside each scenario (used as local index)
    assert (f.groupby("scenario_id").flow_id.agg(lambda s: (s.to_numpy() == np.arange(len(s))).all())).all()
    train_ids = splits[splits.split == "train"].scenario_id.to_numpy()
    stats = fit_normalizer(f, L, h, train_ids)
    with open(out_dir / "norm.json", "w") as fh:
        json.dump({"stats": stats, "flow_feats": FLOW_FEATS, "link_feats": LINK_FEATS, "edge_feats": EDGE_FEATS,
                   "fitted_on": "train"}, fh, indent=1)
    for s in SPLITS:
        ids = np.sort(splits[splits.split == s].scenario_id.to_numpy())[: a.max_scenarios]
        t1 = time.time()
        graphs = build_graphs(f, L, h, stats, ids)
        torch.save(graphs, out_dir / f"{s}.pt")
        n_f = sum(g["flow"].num_nodes for g in graphs)
        n_l = sum(g["link"].num_nodes for g in graphs)
        size_mb = (out_dir / f"{s}.pt").stat().st_size / 1e6
        print(f"{s:17s}: {len(graphs):5d} graphs, {n_f:>9,} flow nodes, {n_l:>9,} link nodes, "
              f"{size_mb:,.0f} MB, {time.time() - t1:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


def load_graphs(split: str) -> list[HeteroData]:
    return torch.load(GRAPHS_DIR / f"{split}.pt", weights_only=False)


if __name__ == "__main__":
    main()
