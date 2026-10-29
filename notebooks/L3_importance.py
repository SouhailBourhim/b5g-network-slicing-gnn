"""L3 - Importance des variables : gain vs SHAP, SHAP par groupe, ablations.

Usage : python -m notebooks.L3_importance
Sorties : figures/L3_02_shap_*.png, figures/L3_03_ablation.png, results/importance_shap_gain.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.baselines_ml import LGBM_MAX_TREES, LGBM_PARAMS, LGBM_PATIENCE, load_data
from src.config import RESULTS_DIR, SEED
from src.features import ALL_FEATURES, FEATURE_GROUPS

N_SHAP = 20_000
FIG = Path("figures")


def main():
    FIG.mkdir(exist_ok=True)
    feats = ALL_FEATURES
    df = load_data(feats)
    is_tr = (df.split == "train").to_numpy()
    is_va = (df.split == "val").to_numpy()
    y = df.log_delay.to_numpy()
    X = df[feats].replace([np.inf, -np.inf], np.nan).astype(np.float32)

    # 1) même modèle que lgbm graine 0
    dtr = lgb.Dataset(X[is_tr], y[is_tr])
    dva = lgb.Dataset(X[is_va], y[is_va], reference=dtr)
    booster = lgb.train({**LGBM_PARAMS, "seed": SEED + 0}, dtr, num_boost_round=LGBM_MAX_TREES,
                        valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(LGBM_PATIENCE, verbose=False)])
    print("arbres retenus :", booster.best_iteration)

    # 2) SHAP sur 20 000 flux de val (tirage fixe)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(np.where(is_va)[0], size=N_SHAP, replace=False)
    Xs = X.iloc[idx]
    t0 = time.time()
    sv = shap.TreeExplainer(booster).shap_values(Xs, num_iteration=booster.best_iteration) \
        if False else shap.TreeExplainer(booster).shap_values(Xs)
    print(f"SHAP : {time.time() - t0:.0f}s")

    shap_imp = pd.Series(np.abs(sv).mean(axis=0), index=feats)
    gain = pd.Series(booster.feature_importance("gain"), index=feats)
    gain = 100 * gain / gain.sum()
    shap_pct = 100 * shap_imp / shap_imp.sum()
    tab = pd.DataFrame({"gain_pct": gain, "shap_pct": shap_pct})
    tab["rang_gain"] = tab.gain_pct.rank(ascending=False).astype(int)
    tab["rang_shap"] = tab.shap_pct.rank(ascending=False).astype(int)
    tab = tab.sort_values("shap_pct", ascending=False).round(2)
    tab.to_csv(RESULTS_DIR / "importance_shap_gain.csv")
    print(tab.head(12).to_string())

    # 3) figures
    top = tab.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.shap_pct, color="tab:blue", label="SHAP (|valeur| moyenne)")
    ax.barh(top.index, top.gain_pct, height=0.35, color="tab:orange", label="gain LightGBM")
    ax.set_xlabel("% de l'importance totale"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "L3_02_shap_vs_gain.png", dpi=130); plt.close(fig)

    plt.figure()
    shap.summary_plot(sv, Xs, max_display=15, show=False)
    plt.tight_layout(); plt.savefig(FIG / "L3_02_shap_beeswarm.png", dpi=130); plt.close()

    # 4) SHAP par groupe
    grp = {g: shap_pct[cols].sum() for g, cols in FEATURE_GROUPS.items()}
    grp = pd.Series(grp).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(grp.index, grp.values); ax.set_xlabel("% SHAP total par groupe"); fig.tight_layout()
    fig.savefig(FIG / "L3_02_shap_groupes.png", dpi=130); plt.close(fig)
    print("\nSHAP par groupe (%):\n", grp.sort_values(ascending=False).round(1).to_string())

    # 5) ablation : perte de MAE_log sur val vs lgbm complet (graine 0)
    m = pd.read_csv(RESULTS_DIR / "metrics_all.csv")
    m = m[(m.slice_type == "ALL") & (m["subset"] == "all") & (m.split == "val") & (m.seed == 0)]
    base = m[m.model == "lgbm"].mae_log.iloc[0]
    ab = m[m.model.str.startswith("lgbm_no_")].set_index("model").mae_log - base
    ab.index = ab.index.str.replace("lgbm_no_", "")
    ab = ab.sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(ab.index, ab.values, color="tab:red")
    ax.axvline(0.002, ls="--", c="gray"); ax.text(0.002, 0, " écart-type\n entre graines", va="bottom", fontsize=8)
    ax.set_xlabel("Δ MAE_log sur val (groupe retiré − modèle complet)"); fig.tight_layout()
    fig.savefig(FIG / "L3_03_ablation.png", dpi=130); plt.close(fig)
    print("\nAblation ΔMAE_log (val):\n", ab.round(4).to_string())


if __name__ == "__main__":
    main()
    