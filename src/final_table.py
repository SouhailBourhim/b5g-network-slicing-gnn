"""Tableau final (L6, C4) : modele x {test, topologies inedites} x {tout, evenements, hors evenements}
x type de slice. MAE et MSE en moyenne +- ecart-type sur les graines, plus l'IC 95 % bootstrap
(sur les scenarios) pour la MAE.

Usage : python -m src.final_table [--include-ablations]
Sorties : results/table_final_long.csv, results/table_final.md
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR

SPLITS = ["test", "test_unseen_topo"]
SUBSETS = ["all", "event", "non_event"]
SLICES = ["ALL", "URLLC", "eMBB", "mMTC"]


def fmt(m: float, s: float) -> str:
    if np.isnan(m):
        return ""
    return f"{m:.3f}" if np.isnan(s) else f"{m:.3f} ± {s:.3f}"


def md(df: pd.DataFrame, title: str) -> str:
    cols = [df.index.name or "modele"] + [str(c) for c in df.columns]
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-ablations", action="store_true")
    args = ap.parse_args()

    d = pd.read_csv(RESULTS_DIR / "metrics_all.csv")
    d = d[d.split.isin(SPLITS)]
    if not args.include_ablations:
        d = d[~d.model.str.startswith("lgbm_no_")]

    g = (d.groupby(["model", "split", "subset", "slice_type"])
          .agg(n_seeds=("seed", "nunique"), n_flows=("n", "first"),
               mae=("mae_log", "mean"), mae_std=("mae_log", "std"),
               mse=("mse_log", "mean"), mse_std=("mse_log", "std"))
          .reset_index())

    # IC bootstrap (scenarios) pour la MAE
    ci = pd.read_csv(RESULTS_DIR / "bootstrap_ci.csv")
    ci = ci[ci.metric == "mae"].copy()
    ci["subset"] = ci["subset"].replace({"nonevent": "non_event"})
    ci = ci[["model", "split", "subset", "slice_type", "ci_lo", "ci_hi"]]
    g = g.merge(ci, on=["model", "split", "subset", "slice_type"], how="left")

    g["mae_txt"] = [fmt(m, s) for m, s in zip(g.mae, g.mae_std)]
    g["mse_txt"] = [fmt(m, s) for m, s in zip(g.mse, g.mse_std)]
    g["mae_ic"] = [f"{m:.3f} [{lo:.3f} ; {hi:.3f}]" if not np.isnan(lo) else fmt(m, s)
                   for m, s, lo, hi in zip(g.mae, g.mae_std, g.ci_lo, g.ci_hi)]
    g.to_csv(RESULTS_DIR / "table_final_long.csv", index=False)

    order = (g[(g.split == "test") & (g.subset == "all") & (g.slice_type == "ALL")]
             .sort_values("mae").model.tolist())

    def wide(sub: pd.DataFrame, columns: list[str], value: str) -> pd.DataFrame:
        w = sub.pivot(index="model", columns=columns, values=value).reindex(order)
        w.columns = [" / ".join(c) if isinstance(c, tuple) else c for c in w.columns]
        return w.fillna("")

    all_slices = g[g.slice_type == "ALL"]
    t1 = wide(all_slices, ["split", "subset"], "mae_ic")
    t2 = wide(all_slices, ["split", "subset"], "mse_txt")
    t3 = wide(g[(g.split == "test") & (g.subset == "all")], ["slice_type"], "mae_txt")[SLICES]
    t4 = wide(g[(g.split == "test") & (g.subset == "event")], ["slice_type"], "mae_txt")[SLICES]

    out = "\n".join([
        md(t1, "MAE_log [IC 95 % bootstrap sur les scenarios], toutes slices"),
        md(t2, "MSE_log (moyenne ± ecart-type sur graines), toutes slices"),
        md(t3, "MAE_log par type de slice, test, tous les flux"),
        md(t4, "MAE_log par type de slice, test, evenements uniquement"),
        "Remarque : seul lgbm a 5 graines ; les autres modeles ont une seule graine (pas d'ecart-type).\n"])
    (RESULTS_DIR / "table_final.md").write_text(out, encoding="utf-8")
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(t1.to_string()); print(); print(t3.to_string()); print(); print(t4.to_string())


if __name__ == "__main__":
    main()