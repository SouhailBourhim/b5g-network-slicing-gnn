"""Input-only features (never computed from simulation outputs).

Three levels:
  * hop  : one row per (flow, link traversed)  -> loads, utilizations, queueing terms
  * link : one row per directed link used by >= 1 flow -> link-node features of the GNN
  * flow : one row per flow -> flat features for Ridge / GBM / MLP + targets

Key quantities (all from the generator parameters, capacities, routing, WFQ):
  rate_bps        offered load of a flow (pkts/s x mean packet size)
  rho_link        sum of rates on the link / capacity            (FIFO view)
  rho_queue       sum of rates in the flow's WFQ queue / (share x capacity)
                  (guaranteed-rate view: WFQ guarantees share x C to a queue)
  mm1_*           M/M/1 sojourn time per hop, rho clipped at RHO_CLIP
  mm1k_*          M/M/1/K sojourn time per hop (finite buffer K = queue size + 1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, SLICE_TYPES

RHO_CLIP = 0.99   # M/M/1 is undefined for rho >= 1: clip just below saturation


# --------------------------------------------------------------------------
# Queueing formulas
# --------------------------------------------------------------------------
def mm1_sojourn(rho: np.ndarray, service_time: np.ndarray, rho_clip: float = RHO_CLIP) -> np.ndarray:
    """M/M/1 mean time in system  T = S / (1 - rho), with rho clipped below 1."""
    rho = np.clip(rho, 0.0, rho_clip)
    return service_time / (1.0 - rho)


def mm1k_sojourn(rho: np.ndarray, service_time: np.ndarray, K: np.ndarray) -> np.ndarray:
    """M/M/1/K mean time in system (K = max packets in the system).

    Stationary law: P_n = rho^n (1-rho) / (1-rho^(K+1)),  n = 0..K
      L   = sum n P_n   = x/(1-x) - (K+1) x^(K+1) / (1-x^(K+1))   with x = rho (< 1)
      P_K = blocking probability (packet arrives to a full buffer -> lost)
      T   = L / lambda_eff,  lambda_eff = lambda (1 - P_K),  lambda = rho / S   (Little)
    For rho > 1 the law is the mirror image of the one for 1/rho (P_n <-> P_{K-n}),
    so L(rho) = K - L(1/rho) and P_K(rho) = P_0(1/rho): no overflow, no special case.
    Valid for any rho >= 0: the finite buffer bounds the delay (~K x S); traffic
    in excess becomes loss instead of delay.
    """
    rho = np.maximum(np.asarray(rho, dtype=float), 1e-12)
    K = np.broadcast_to(np.asarray(K, dtype=float), rho.shape)
    S = np.asarray(service_time, dtype=float)
    above = rho > 1.0
    x = np.where(above, 1.0 / rho, rho)                  # x in (0, 1]
    at1 = x > 1.0 - 1e-6                                 # rho ~ 1: closed form is 0/0
    x = np.where(at1, 0.5, x)                            # dummy value, replaced below
    xK1 = x ** (K + 1)
    L_x = x / (1 - x) - (K + 1) * xK1 / (1 - xK1)        # mean number in system for load x
    P0_x = (1 - x) / (1 - xK1)                           # P(empty) for load x
    PK_x = P0_x * x ** K                                 # P(full) for load x
    L_x = np.where(at1, K / 2, L_x)                      # limits at rho = 1: uniform law
    P0_x = np.where(at1, 1 / (K + 1), P0_x)
    PK_x = np.where(at1, 1 / (K + 1), PK_x)
    L = np.where(above, K - L_x, L_x)
    PK = np.where(above, P0_x, PK_x)
    lam_eff = rho / S * (1 - PK)
    return L / lam_eff


# --------------------------------------------------------------------------
# Hop level
# --------------------------------------------------------------------------
def hop_table(flows: pd.DataFrame, flow_links: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """One row per (flow, hop) with link/queue loads computed from inputs."""
    h = flow_links.merge(flows[["scenario_id", "flow_id", "rate_bps", "avg_pkts_lambda", "mean_pkt_size"]],
                         on=["scenario_id", "flow_id"], how="left")
    h = h.merge(links[["scenario_id", "link_id", "capacity_bps", "src_type", "dst_type"]],
                on=["scenario_id", "link_id"], how="left")
    # The default (last) WFQ queue gets weight 100 - sum(reservations), which is
    # negative when reservations exceed capacity (rare). A negative guaranteed
    # rate is meaningless: floor at 1e-5, the share of the simulator's minimum
    # weight 0.001 (out of 100). Raw value kept for traceability.
    h["wfq_share_raw"] = h["wfq_share"]
    h["wfq_share"] = h["wfq_share"].clip(lower=1e-5)
    link_key = ["scenario_id", "link_id"]
    queue_key = ["scenario_id", "link_id", "queue"]
    g_link, g_queue = h.groupby(link_key), h.groupby(queue_key)

    h["load_link"] = g_link["rate_bps"].transform("sum")
    h["pkts_link"] = g_link["avg_pkts_lambda"].transform("sum")
    h["n_flows_link"] = g_link["rate_bps"].transform("size")
    h["rho_link"] = h["load_link"] / h["capacity_bps"]
    h["load_queue"] = g_queue["rate_bps"].transform("sum")
    h["n_flows_queue"] = g_queue["rate_bps"].transform("size")
    h["rho_queue"] = h["load_queue"] / (h["wfq_share"] * h["capacity_bps"])

    # Mean service time of a packet on the link (load-weighted packet size / C)
    mean_size_link = h["load_link"] / h["pkts_link"]
    h["service_time_link"] = mean_size_link / h["capacity_bps"]
    h["tx_time_own"] = h["mean_pkt_size"] / h["capacity_bps"]   # own packet transmission
    K = h["queue_size_pkts"].to_numpy() + 1
    h["mm1_link"] = mm1_sojourn(h["rho_link"].to_numpy(), h["service_time_link"].to_numpy())
    h["mm1k_link"] = mm1k_sojourn(h["rho_link"].to_numpy(), h["service_time_link"].to_numpy(), K)
    # WFQ guaranteed-rate view: the queue is served at share x C
    mean_size_q = h["load_queue"] / g_queue["avg_pkts_lambda"].transform("sum")
    st_q = mean_size_q / (h["wfq_share"] * h["capacity_bps"])
    h["mm1k_queue"] = mm1k_sojourn(h["rho_queue"].to_numpy(), st_q.to_numpy(), K)
    return h


def link_table(h: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Link-level input features for links used by at least one flow."""
    agg = (h.groupby(["scenario_id", "link_id"])
             .agg(load_link=("load_link", "first"), n_flows_link=("n_flows_link", "first"),
                  rho_link=("rho_link", "first"), mm1k_link=("mm1k_link", "first"),
                  n_queues_used=("queue", "nunique"))
             .reset_index())
    out = links.merge(agg, on=["scenario_id", "link_id"], how="inner")
    return out


# --------------------------------------------------------------------------
# Flow level (flat feature vector)
# --------------------------------------------------------------------------
def flow_table(flows: pd.DataFrame, h: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    """Flat per-flow features (aggregates over the path) + targets."""
    g = h.groupby(["scenario_id", "flow_id"])
    agg = g.agg(rho_link_max=("rho_link", "max"), rho_link_mean=("rho_link", "mean"),
                rho_link_sum=("rho_link", "sum"),
                rho_queue_max=("rho_queue", "max"), rho_queue_mean=("rho_queue", "mean"),
                wfq_share_min=("wfq_share", "min"), wfq_share_mean=("wfq_share", "mean"),
                n_default_queue_hops=("is_default_queue", "sum"),
                cap_min=("capacity_bps", "min"), cap_mean=("capacity_bps", "mean"),
                n_flows_link_max=("n_flows_link", "max"), n_flows_link_mean=("n_flows_link", "mean"),
                tx_time_sum=("tx_time_own", "sum"),
                mm1_sum=("mm1_link", "sum"), mm1k_link_sum=("mm1k_link", "sum"),
                mm1k_queue_sum=("mm1k_queue", "sum"), mm1k_queue_max=("mm1k_queue", "max"),
                n_hops_rho_gt1=("rho_link", lambda x: (x > 1).sum()))
    # number of competing flows on the busiest link of the path
    idx = h.groupby(["scenario_id", "flow_id"])["rho_link"].idxmax()
    agg["n_flows_busiest_link"] = h.loc[idx, ["scenario_id", "flow_id", "n_flows_link"]] \
        .set_index(["scenario_id", "flow_id"])["n_flows_link"]
    f = flows.merge(agg.reset_index(), on=["scenario_id", "flow_id"], how="left")
    f = f.merge(scenarios[["scenario_id", "topology", "n_core_nodes", "n_flows", "n_slices"]],
                on="scenario_id", how="left")
    # provisioning ratio of the slice: reserved / real offered rate (see JOURNAL D1.2)
    f["prov_ratio"] = (0.85 + 0.3 * f["delta"]) * f["nominal_bw"] / f["rate_bps"]
    for t in SLICE_TYPES:
        f[f"is_{t}"] = (f["slice_type"] == t).astype(np.int8)
    # targets. delay == -1 is the simulator's sentinel for "no packet delivered"
    # (100% loss): the delay is undefined, so the flow is kept for the loss
    # target but flagged out of the delay regression.
    f["loss"] = f["pkts_drop"] / f["pkts_gen"]
    f["has_delay"] = f["delay"] > 0
    f["log_delay"] = np.where(f["has_delay"], np.log10(f["delay"].where(f["has_delay"], 1.0)), np.nan)
    return f


# Feature groups (used by the flat models and by the ablation study)
FEATURE_GROUPS = {
    "slice":       ["is_eMBB", "is_URLLC", "is_mMTC", "delta", "prov_ratio"],
    "traffic":     ["rate_bps", "avg_pkts_lambda", "mean_pkt_size"],
    "path":        ["path_len"],
    "capacity":    ["cap_min", "cap_mean", "tx_time_sum"],
    "utilization": ["rho_link_max", "rho_link_mean", "rho_link_sum", "n_hops_rho_gt1",
                    "n_flows_link_max", "n_flows_link_mean", "n_flows_busiest_link"],
    "wfq":         ["wfq_share_min", "wfq_share_mean", "rho_queue_max", "rho_queue_mean",
                    "n_default_queue_hops"],
    "queueing":    ["mm1_sum", "mm1k_link_sum", "mm1k_queue_sum", "mm1k_queue_max"],
    "scenario":    ["n_core_nodes", "n_flows", "n_slices"],
}
ALL_FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]


def load_tables(processed_dir=PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    return {k: pd.read_parquet(processed_dir / f"{k}.parquet")
            for k in ("scenarios", "links", "flows", "flow_links")}


def build_all(processed_dir=PROCESSED_DIR, save: bool = True) -> dict[str, pd.DataFrame]:
    """Compute hop / link / flow feature tables and cache them as parquet."""
    t = load_tables(processed_dir)
    h = hop_table(t["flows"], t["flow_links"], t["links"])
    lk = link_table(h, t["links"])
    f = flow_table(t["flows"], h, t["scenarios"])
    if save:
        h.to_parquet(processed_dir / "hop_features.parquet", index=False)
        lk.to_parquet(processed_dir / "link_features.parquet", index=False)
        f.drop(columns=["path"]).to_parquet(processed_dir / "flow_features.parquet", index=False)
    return {"hops": h, "links": lk, "flows": f, "scenarios": t["scenarios"]}


if __name__ == "__main__":
    import time
    t0 = time.time()
    out = build_all()
    print({k: v.shape for k, v in out.items()}, f"{time.time() - t0:.0f}s")
