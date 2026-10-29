"""Visual check of the event definition on small scenarios (volet B, B3).

Links are coloured by rho_link (from inputs). A black cross marks a starved hop (WFQ share < 1e-3 on a
link with rho >= TAU). The blue path is one flow in event (a starved flow if any), drawn only to show what
the rule picks; delays are NOT used to define anything.

Usage: python -m src.events_viz   -> reports/figures/L4_01_scenarios.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import PROCESSED_DIR
from src.events import STARVE_SHARE, TAU
from src.plotting import NEUTRAL, savefig, set_style

# (scenario_id, panel title). All from TRAIN so the visual check never looks at test data.
CASES = [(4618, "Aucun événement"),
         (5792, "Congestion simple"),
         (290, "Famine"),
         (2294, "Famine locale")]
BOUNDS = [0, 0.5, 0.9, TAU, 1.5, 2.5, 1e9]
COLORS = ["#fee8c8", "#fdd49e", "#fdbb84", "#e34a33", "#b30000", "#7f0000"]
NORM = mcolors.BoundaryNorm(BOUNDS, len(COLORS))
CMAP = mcolors.ListedColormap(COLORS)
FLOW_BLUE = "#2a78d6"


def load_case(sid: int):
    lk = pd.read_parquet(PROCESSED_DIR / "link_features.parquet", filters=[("scenario_id", "==", sid)])
    hp = pd.read_parquet(PROCESSED_DIR / "hop_features.parquet", filters=[("scenario_id", "==", sid)])
    fl = pd.read_parquet(PROCESSED_DIR / "flows.parquet", columns=["scenario_id", "flow_id", "path"],
                         filters=[("scenario_id", "==", sid)])
    ev = pd.read_parquet(PROCESSED_DIR / "events.parquet", filters=[("scenario_id", "==", sid)])
    return lk, hp, fl, ev


def draw_case(ax, sid: int, title: str):
    lk, hp, fl, ev = load_case(sid)
    G = nx.Graph()
    G.add_edges_from(zip(lk.src, lk.dst))
    pos = nx.spring_layout(G, seed=3, k=1.2 / np.sqrt(max(len(G), 1)), iterations=200)
    ntype = {}
    for r in lk.itertuples():
        ntype[r.src], ntype[r.dst] = r.src_type, r.dst_type
    for n, (x, y) in pos.items():
        ax.scatter(x, y, s=70 if ntype.get(n) == "antenna" else 90, c="white", edgecolors=NEUTRAL, linewidths=1.0,
                   marker="^" if ntype.get(n) == "antenna" else "o", zorder=3)
    # links: coloured by rho, the two directions drawn as opposite arcs
    mid = {}
    for r in lk.itertuples():
        x0, y0 = pos[r.src]; x1, y1 = pos[r.dst]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=CMAP(NORM(r.rho_link)), lw=2.2 if r.rho_link >= TAU else 1.3,
                                    shrinkA=6, shrinkB=6, connectionstyle="arc3,rad=0.15", mutation_scale=9))
        # midpoint of matplotlib's arc3 curve (rad = 0.15): chord midpoint + rad/2 * (dy, -dx)
        mid[r.link_id] = ((x0 + x1) / 2 + 0.075 * (y1 - y0), (y0 + y1) / 2 - 0.075 * (x1 - x0))
    # starved hops
    st = hp[(hp.wfq_share < STARVE_SHARE) & (hp.rho_link >= TAU)].link_id.unique()
    for lid in st:
        ax.scatter(*mid[lid], marker="X", s=70, c="black", zorder=5)
    # one flow in event, preferring a starved one
    evf = ev[ev.is_event]
    starved_flows = set(hp[(hp.wfq_share < STARVE_SHARE) & (hp.rho_link >= TAU)].flow_id)
    pick = [f for f in evf.flow_id if f in starved_flows] or list(evf.flow_id)
    if pick:
        path = fl.set_index("flow_id").loc[pick[0], "path"]
        for a, b in zip(path[:-1], path[1:]):
            if a in pos and b in pos:
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=FLOW_BLUE, lw=5, alpha=0.28, zorder=1,
                        solid_capstyle="round")
    topo = pd.read_parquet(PROCESSED_DIR / "scenarios.parquet", columns=["scenario_id", "topology"]).set_index(
        "scenario_id").loc[sid, "topology"]
    ax.set_title(f"{title}\nscénario {sid} ({topo}) : {len(ev)} flux, {int(ev.is_event.sum())} en événement, "
                 f"ρ max {lk.rho_link.max():.2f}", fontsize=9.5)
    ax.set_axis_off(); ax.grid(False)


def main():
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (sid, title) in zip(axes.ravel(), CASES):
        draw_case(ax, sid, title)
    labels = ["ρ < 0,5", "0,5 – 0,9", f"0,9 – {TAU}", f"{TAU} – 1,5 (congestion)", "1,5 – 2,5", "≥ 2,5"]
    handles = [Line2D([0], [0], color=c, lw=3) for c in COLORS]
    handles += [Line2D([0], [0], marker="X", color="w", markerfacecolor="black", markersize=9),
                Line2D([0], [0], color=FLOW_BLUE, lw=5, alpha=0.4),
                Line2D([0], [0], marker="^", color="w", markeredgecolor=NEUTRAL, markerfacecolor="white", markersize=8),
                Line2D([0], [0], marker="o", color="w", markeredgecolor=NEUTRAL, markerfacecolor="white", markersize=8)]
    labels += ["saut affamé (part WFQ < 1e-3, ρ ≥ τ)", "chemin d'un flux en événement", "antenne", "nœud interne"]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    print("written:", savefig(fig, "L4_01_scenarios"))


if __name__ == "__main__":
    main()
