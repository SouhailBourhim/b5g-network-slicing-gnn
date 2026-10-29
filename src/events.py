"""Event definition (volet B, L4). An event is a STATE of the network, not an instant:
the dataset has no time axis (one steady-state snapshot per scenario).

Both rules use INPUTS only (rates, capacities, routing, WFQ weights) - never delay or loss.

  congestion : rho_link_max >= TAU            (some link on the path is overloaded)
  starvation : some hop has WFQ share < STARVE_SHARE on a link with rho_link >= TAU
               (the flow's queue is guaranteed almost nothing on a saturated link)
  event_type : none | congestion | starvation | both ;  is_event = event_type != 'none'

TAU is chosen on TRAIN only (see results/L4_tau_choice.csv). Sensitivity columns
is_event_tau_<t> recompute the whole rule for each t in TAU_SENSITIVITY.

Usage: python -m src.events
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR
from src.contracts import save_events
from src.splits import load_splits

TAU = 1.05                                   # main threshold (elbow of the median delay on train)
TAU_SENSITIVITY = [0.9, 1.0, 1.05, 1.1, 2.0]  # 2.0 = "severe" tier (about top 15 % of flows)
STARVE_SHARE = 1e-3


def tau_col(t: float) -> str:
    return "is_event_tau_" + f"{t:g}".replace(".", "_")


def starvation_by_tau(hops: pd.DataFrame, taus) -> pd.DataFrame:
    """Per flow: does it have a hop with share < STARVE_SHARE on a link with rho_link >= tau?"""
    low = hops[hops.wfq_share < STARVE_SHARE]
    out = {}
    for t in taus:
        hit = low[low.rho_link >= t].groupby(["scenario_id", "flow_id"]).size()
        out[t] = hit
    return pd.DataFrame(out).notna()   # index = flows with at least one low-share hop


def build_events() -> pd.DataFrame:
    f = pd.read_parquet(PROCESSED_DIR / "flow_features.parquet",
                        columns=["scenario_id", "flow_id", "rho_link_max"])
    h = pd.read_parquet(PROCESSED_DIR / "hop_features.parquet",
                        columns=["scenario_id", "flow_id", "wfq_share", "rho_link"])
    st = starvation_by_tau(h, sorted(set(TAU_SENSITIVITY + [TAU])))
    f = f.merge(st.reset_index(), on=["scenario_id", "flow_id"], how="left")
    taus = sorted(set(TAU_SENSITIVITY + [TAU]))
    for t in taus:
        f[t] = f[t].fillna(False).astype(bool)
    cong = f.rho_link_max >= TAU
    starv = f[TAU]
    f["event_type"] = np.select([cong & starv, cong, starv], ["both", "congestion", "starvation"], "none")
    f["is_event"] = f.event_type != "none"
    for t in TAU_SENSITIVITY:
        f[tau_col(t)] = (f.rho_link_max >= t) | f[t]
    keep = ["scenario_id", "flow_id", "is_event", "event_type"] + [tau_col(t) for t in TAU_SENSITIVITY]
    return f[keep].astype({"scenario_id": "int32", "flow_id": "int32"})


def coverage(ev: pd.DataFrame) -> pd.DataFrame:
    """% of flows and of scenarios in event, by split and slice type."""
    fl = pd.read_parquet(PROCESSED_DIR / "flow_features.parquet", columns=["scenario_id", "flow_id", "slice_type"])
    d = ev.merge(fl, on=["scenario_id", "flow_id"]).merge(load_splits()[["scenario_id", "split"]], on="scenario_id")
    rows = []
    for keys, g in [(("all", s), x) for s, x in d.groupby("split")] + \
                   [((sl, s), x) for (s, sl), x in d.groupby(["split", "slice_type"])]:
        rows.append(dict(slice=keys[0], split=keys[1], n_flows=len(g), pct_flows_event=100 * g.is_event.mean(),
                         pct_congestion=100 * g.event_type.isin(["congestion", "both"]).mean(),
                         pct_starvation=100 * g.event_type.isin(["starvation", "both"]).mean(),
                         pct_scen_with_event=100 * g.groupby("scenario_id").is_event.any().mean()))
    return pd.DataFrame(rows).round(1)


def main():
    ev = build_events()
    print("written:", save_events(ev))
    print(ev.event_type.value_counts(normalize=True).round(3).to_dict())
    cov = coverage(ev)
    cov.to_csv(RESULTS_DIR / "L4_coverage.csv", index=False)
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
