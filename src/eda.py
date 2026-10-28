"""Exploration helpers (L1): data checks and the figures of the report."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import SLICE_TYPES
from src.plotting import SLICE_COLORS, NEUTRAL, savefig


# --------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------
def quality_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Types, missing values and duplicates for every table (one row per column)."""
    keys = {"scenarios": ["scenario_id"], "links": ["scenario_id", "link_id"],
            "flows": ["scenario_id", "flow_id"], "flow_links": ["scenario_id", "flow_id", "hop"]}
    rows = []
    for name, df in tables.items():
        n_dup = int(df.duplicated(subset=keys[name]).sum()) if name in keys else -1
        for c in df.columns:
            if c == "path":
                continue
            rows.append({"table": name, "column": c, "dtype": str(df[c].dtype),
                         "n_missing": int(df[c].isna().sum()), "n_rows": len(df),
                         "n_duplicate_keys_in_table": n_dup})
    return pd.DataFrame(rows)


def impossible_values(f: pd.DataFrame) -> dict:
    """Count physically impossible or sentinel values in the flow table."""
    return {
        "delay_sentinel_-1 (no packet delivered)": int((f["delay"] == -1).sum()),
        "delay_negative_other": int(((f["delay"] < 0) & (f["delay"] != -1)).sum()),
        "jitter_negative_other": int(((f["jitter"] < 0) & (f["delay"] != -1)).sum()),
        "loss_gt_1": int((f["loss"] > 1 + 1e-9).sum()),
        "loss_negative": int((f["loss"] < 0).sum()),
        "pkts_gen_zero": int((f["pkts_gen"] <= 0).sum()),
        "rate_bps_nonpositive": int((f["rate_bps"] <= 0).sum()),
        "nan_in_targets": int(f[["delay", "jitter", "pkts_drop"]].isna().sum().sum()),
        "percentiles_not_sorted": int((f[f.has_delay][["p10", "p20", "p50", "p80", "p90"]].diff(axis=1).iloc[:, 1:] < -1e-12).any(axis=1).sum()),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_inventory(scen: pd.DataFrame):
    """Topology sizes, flows and slices per scenario."""
    topo = scen.groupby("topology").agg(n_core_nodes=("n_core_nodes", "first"),
                                        n_scen=("scenario_id", "size"))
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
    ax[0].hist(topo["n_core_nodes"], bins=30, color=NEUTRAL)
    ax[0].set(title="Taille des topologies", xlabel="nœuds du cœur (inner+antenna)", ylabel="nb de topologies")
    ax[1].hist(topo["n_scen"], bins=30, color=NEUTRAL)
    ax[1].set(title="Scénarios par topologie", xlabel="nb de scénarios", ylabel="nb de topologies")
    ax[2].hist(scen["n_flows"], bins=40, color=NEUTRAL)
    ax[2].set(title="Flux par scénario", xlabel="nb de flux", ylabel="nb de scénarios")
    for t in SLICE_TYPES:
        ax[3].hist(scen[f"n_slices_{t}"], bins=np.arange(0, scen[[f"n_slices_{s}" for s in SLICE_TYPES]].max().max() + 2) - 0.5,
                   histtype="step", lw=2, color=SLICE_COLORS[t], label=t)
    ax[3].set(title="Slices par scénario", xlabel="nb de slices du type", ylabel="nb de scénarios")
    ax[3].legend()
    fig.tight_layout()
    return fig


def fig_targets(f: pd.DataFrame):
    """Distribution of delay, jitter and loss per slice type (log scales)."""
    d = f[f.has_delay]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    bins = np.linspace(np.log10(d.delay.min()), np.log10(d.delay.max()), 80)
    for t in SLICE_TYPES:
        x = d[d.slice_type == t]
        ax[0].hist(np.log10(x.delay), bins=bins, histtype="step", lw=2, density=True, color=SLICE_COLORS[t], label=t)
        jit = x.jitter[x.jitter > 0]
        ax[1].hist(np.log10(jit), bins=80, histtype="step", lw=2, density=True, color=SLICE_COLORS[t], label=t)
        ax[2].hist(f[f.slice_type == t].loss, bins=50, histtype="step", lw=2, density=True, color=SLICE_COLORS[t], label=t)
    ax[0].set(title="Délai moyen par flux", xlabel="log10(délai [s])", ylabel="densité")
    ax[1].set(title="Gigue (variance du délai)", xlabel="log10(gigue [s²])", ylabel="densité")
    ax[2].set(title="Taux de perte par flux", xlabel="pertes / paquets générés", ylabel="densité (échelle log)", yscale="log")
    ax[0].legend()
    fig.tight_layout()
    return fig


def binned_quantiles(x: pd.Series, y: pd.Series, bins) -> pd.DataFrame:
    """Median and inter-quartile range of y in bins of x (robust to heavy tails)."""
    cat = pd.cut(x, bins)
    g = y.groupby(cat, observed=True)
    out = pd.DataFrame({"q25": g.quantile(0.25), "q50": g.median(), "q75": g.quantile(0.75), "n": g.size()})
    out["x"] = [iv.mid for iv in out.index]
    return out[out.n >= 30]


def fig_delay_vs(f: pd.DataFrame, col: str, bins, xlabel: str, title: str, logx: bool = False):
    """Median delay (+ IQR band) as a function of an input feature, per slice."""
    d = f[f.has_delay]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for t in SLICE_TYPES:
        x = d[d.slice_type == t]
        b = binned_quantiles(x[col], x.delay, bins)
        ax.plot(b.x, b.q50, color=SLICE_COLORS[t], label=t, marker="o", ms=4)
        ax.fill_between(b.x, b.q25, b.q75, color=SLICE_COLORS[t], alpha=0.15, lw=0)
    ax.set(yscale="log", xlabel=xlabel, ylabel="délai [s] (médiane, IQR)", title=title)
    if logx:
        ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_rho_validation(links: pd.DataFrame):
    """Input-side utilization min(rho,1) vs utilization measured by the simulator."""
    L = links[links.meas_util.notna()]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    hb = ax[0].hexbin(np.minimum(L.rho_link, 1.0), L.meas_util, gridsize=60, bins="log", cmap="Blues", mincnt=1)
    ax[0].plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls="--")
    ax[0].set(xlabel="min(ρ calculé depuis les entrées, 1)", ylabel="utilisation mesurée (simulateur)",
              title="Validation de ρ (liens utilisés)")
    fig.colorbar(hb, ax=ax[0], label="nb de liens (log)")
    ax[1].hist(L.rho_link, bins=np.linspace(0, min(L.rho_link.quantile(0.999), 4), 80), color=NEUTRAL)
    ax[1].axvline(1, color="#e34948", lw=1.5, ls="--")
    ax[1].set(xlabel="ρ_lien = charge offerte / capacité", ylabel="nb de liens", title="Charge offerte des liens", yscale="log")
    fig.tight_layout()
    return fig


def fig_correlations(f: pd.DataFrame, features: list[str]):
    """Spearman correlation of each input feature with log10(delay)."""
    d = f[f.has_delay]
    corr = d[features + ["log_delay"]].corr(method="spearman")["log_delay"].drop("log_delay").sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 0.24 * len(corr) + 1))
    ax.barh(corr.index, corr.values, color=[("#2a78d6" if v > 0 else "#eb6834") for v in corr.values], height=0.7)
    ax.axvline(0, color=NEUTRAL, lw=0.8)
    ax.set(xlabel="corrélation de Spearman avec log10(délai)", title="Variables d'entrée vs délai")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig, corr


def fig_corr_matrix(f: pd.DataFrame, features: list[str]):
    """Spearman correlation between input features (redundancy check)."""
    c = f[features].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), features, rotation=90, fontsize=7)
    ax.set_yticks(range(len(features)), features, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Spearman")
    ax.set_title("Corrélations entre variables d'entrée")
    fig.tight_layout()
    return fig
