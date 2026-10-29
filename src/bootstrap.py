"""Intervalles de confiance par bootstrap sur les SCENARIOS (L6, C4).

Les flux d'un scenario partagent des liens : on reechantillonne donc les scenarios,
pas les flux. Poids multinomiaux identiques pour tous les modeles d'un groupe
(differences appariees). Les predictions sont moyennees sur les graines.

Usage : python -m src.bootstrap [--B 1000] [--ref lgbm] [--all-models]
Sorties : results/bootstrap_ci.csv, results/bootstrap_diff.csv, results/bootstrap_load_matched.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR, SEED
from src.contracts import load_events, load_predictions
from src.splits import load_splits


def flow_table() -> pd.DataFrame:
    f = pd.read_parquet(PROCESSED_DIR / "flow_features.parquet",
                        columns=["scenario_id", "flow_id", "has_delay", "log_delay", "slice_type"])
    f = f[f.has_delay].drop(columns="has_delay")
    f = f.merge(load_splits()[["scenario_id", "split", "scen_rho_max"]], on="scenario_id")
    ev = load_events()
    if ev is not None:
        f = f.merge(ev[["scenario_id", "flow_id", "is_event"]], on=["scenario_id", "flow_id"], how="left")
        f["is_event"] = f["is_event"].astype("float")      # 1 event, 0 non-event, NaN inconnu
    return f


def per_scenario_stats(preds: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Sommes d'erreurs par (modele, split, scenario, type de slice, sous-ensemble), moyennees sur graines."""
    p = preds.merge(f, on=["scenario_id", "flow_id"], how="inner")
    err = p["pred_log10_delay"].to_numpy(dtype=np.float64) - p["log_delay"].to_numpy(dtype=np.float64)
    p["ae"], p["se"] = np.abs(err), err ** 2
    subsets = {"all": np.ones(len(p), bool)}
    if "is_event" in p.columns and p["is_event"].notna().any():
        subsets["event"] = (p["is_event"] == 1).to_numpy()
        subsets["nonevent"] = (p["is_event"] == 0).to_numpy()
    else:
        print("events.parquet absent : seuls les sous-ensembles 'all' sont calcules")
    out = []
    for sub, ms in subsets.items():
        for sl in ["ALL"] + sorted(p["slice_type"].unique()):
            m = ms if sl == "ALL" else ms & (p["slice_type"] == sl).to_numpy()
            g = (p[m].groupby(["model", "seed", "split", "scenario_id"], observed=True)
                     .agg(ae=("ae", "sum"), se=("se", "sum"), n=("ae", "size")).reset_index())
            g = (g.drop(columns="seed").groupby(["model", "split", "scenario_id"], observed=True)
                  .mean().reset_index())
            out.append(g.assign(slice_type=sl, subset=sub))
    return pd.concat(out, ignore_index=True)


def _pivots(g: pd.DataFrame):
    A = g.pivot(index="scenario_id", columns="model", values="ae").fillna(0.0)
    E = g.pivot(index="scenario_id", columns="model", values="se").fillna(0.0).reindex(A.index)
    N = g.pivot(index="scenario_id", columns="model", values="n").fillna(0.0).reindex(A.index)
    return A, E, N


def _weights(S: int, B: int, seed: int) -> np.ndarray:
    """Ligne 0 = poids 1 (estimation ponctuelle), lignes 1..B = tirages bootstrap."""
    rng = np.random.default_rng(seed)
    W = rng.multinomial(S, np.full(S, 1.0 / S), size=B).astype(np.float64)
    return np.vstack([np.ones((1, S)), W])


def bootstrap_groups(stats: pd.DataFrame, B: int, ref: str, min_scen: int = 20):
    ci_rows, diff_rows = [], []
    for (split, sl, sub), g in stats.groupby(["split", "slice_type", "subset"], observed=True):
        A, E, N = _pivots(g)
        S = len(A)
        if S < min_scen:
            continue
        W = _weights(S, B, SEED)
        models = list(A.columns)
        with np.errstate(divide="ignore", invalid="ignore"):
            den = W @ N.to_numpy()
            res = {"mae": (W @ A.to_numpy()) / den, "mse": (W @ E.to_numpy()) / den}
        meta = dict(split=split, slice_type=sl, subset=sub, n_scenarios=S)
        for metric, M in res.items():
            for j, m in enumerate(models):
                lo, hi = np.nanpercentile(M[1:, j], [2.5, 97.5])
                ci_rows.append({**meta, "model": m, "metric": metric, "value": M[0, j], "ci_lo": lo, "ci_hi": hi})
            if ref in models:
                jr = models.index(ref)
                for j, m in enumerate(models):
                    if j == jr:
                        continue
                    d = M[:, j] - M[:, jr]
                    lo, hi = np.nanpercentile(d[1:], [2.5, 97.5])
                    diff_rows.append({**meta, "model": m, "ref": ref, "metric": metric, "diff": d[0],
                                      "ci_lo": lo, "ci_hi": hi, "p_better": float(np.mean(d[1:] < 0))})
    return pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)


def load_matched(stats: pd.DataFrame, scen: pd.DataFrame, B: int, n_bins: int = 5) -> pd.DataFrame:
    """MAE 'topologies inedites' vs 'test' a charge egale (ponderation par classes de scen_rho_max du test)."""
    s = stats[(stats.slice_type == "ALL") & (stats.subset == "all")].merge(
        scen[["scenario_id", "scen_rho_max"]], on="scenario_id")
    edges = np.quantile(scen.loc[scen.split == "test", "scen_rho_max"], np.linspace(0, 1, n_bins + 1)[1:-1])
    s["bin"] = np.digitize(s["scen_rho_max"], edges)
    t = s[s.split == "test"]
    t = t[t.model == t.model.iloc[0]]
    pi = t.groupby("bin").n.sum().reindex(range(n_bins)).fillna(0).to_numpy()
    pi = pi / pi.sum()

    raw, matched = {}, {}
    for k, split in enumerate(["test", "test_unseen_topo"]):
        g = s[s.split == split]
        A, _, N = _pivots(g)
        bins = g.drop_duplicates("scenario_id").set_index("scenario_id")["bin"].reindex(A.index).to_numpy()
        onehot = np.eye(n_bins)[bins]
        W = _weights(len(A), B, SEED + 1 + k)           # echantillons independants entre les deux splits
        for j, m in enumerate(A.columns):
            a, n = A.iloc[:, j].to_numpy()[:, None], N.iloc[:, j].to_numpy()[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                num_k = W @ (a * onehot)
                den_k = W @ (n * onehot)
                mae_k = num_k / den_k
                valid = den_k > 0
                w = np.where(valid, pi, 0.0)
                matched[(split, m)] = (np.where(valid, mae_k, 0.0) * w).sum(1) / w.sum(1)
                raw[(split, m)] = (W @ a[:, 0]) / (W @ n[:, 0])
    rows = []
    for m in sorted({m for _, m in raw}):
        for label, d in [("brut", raw), ("charge_egale", matched)]:
            te, un = d[("test", m)], d[("test_unseen_topo", m)]
            diff = un - te
            rows.append(dict(model=m, comparaison=label, mae_test=te[0], mae_unseen=un[0],
                             diff_unseen_moins_test=diff[0],
                             ci_lo=np.nanpercentile(diff[1:], 2.5), ci_hi=np.nanpercentile(diff[1:], 97.5)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--ref", default="lgbm", help="modele de reference pour les differences")
    ap.add_argument("--all-models", action="store_true", help="inclure aussi les ablations lgbm_no_*")
    args = ap.parse_args()

    preds = load_predictions()
    if not args.all_models:
        preds = preds[~preds.model.str.startswith("lgbm_no_")]
    preds["model"] = preds["model"].astype("category")
    print("modeles :", sorted(preds.model.unique()),
          "| graines :", preds.groupby("model", observed=True).seed.nunique().to_dict())

    f = flow_table()
    stats = per_scenario_stats(preds, f)
    ci, diff = bootstrap_groups(stats, args.B, args.ref)
    scen = load_splits()[["scenario_id", "split", "scen_rho_max"]]
    lm = load_matched(stats, scen, args.B)

    RESULTS_DIR.mkdir(exist_ok=True)
    ci.to_csv(RESULTS_DIR / "bootstrap_ci.csv", index=False)
    diff.to_csv(RESULTS_DIR / "bootstrap_diff.csv", index=False)
    lm.to_csv(RESULTS_DIR / "bootstrap_load_matched.csv", index=False)

    pd.set_option("display.width", 160)
    head = ci[(ci.slice_type == "ALL") & (ci["subset"] == "all") & (ci.metric == "mae")]
    print("\nMAE_log [IC 95 % bootstrap scenarios], toutes slices :")
    print(head.round(4).drop(columns=["slice_type", "subset", "metric"]).to_string(index=False))
    print(f"\nDifferences (modele - {args.ref}), MAE_log, toutes slices :")
    print(diff[(diff.slice_type == "ALL") & (diff["subset"] == "all") & (diff.metric == "mae")]
          .round(4).drop(columns=["slice_type", "subset", "metric"]).to_string(index=False))
    print("\nTopologies inedites - test, brut vs a charge egale :")
    print(lm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()