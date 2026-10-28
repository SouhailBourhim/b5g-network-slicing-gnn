# %% [markdown]
# # L1 — Exploration du jeu de données « network slicing » (Farreras et al., 2024)
#
# **Objectif.** Comprendre ce que contient réellement le jeu de données avant de formuler le problème d'apprentissage :
# combien de scénarios et de topologies, quelles variables, quelle qualité, et comment la cible (délai par flux)
# dépend des entrées.
#
# **Rappel du vocabulaire.**
# - *Scénario* = une simulation complète : une topologie, des slices admises, du trafic, un routage → des mesures.
# - *Flux* = un trafic d'un nœud source (un équipement d'une slice) vers une destination ; c'est l'unité de prédiction.
# - *Lien* = un lien orienté du graphe, avec une capacité en bits/s.
# - *ρ (rho)* = charge offerte / capacité. Si ρ ≥ 1, le lien reçoit plus qu'il ne peut transmettre.
#
# Le parsing brut (`src/parsing.py`, validé contre l'API officielle par `tests/test_parsing.py`) a produit 4 tables :
# `scenarios`, `links`, `flows`, `flow_links` (un saut d'un flux sur un lien). `src/features.py` calcule ensuite,
# **uniquement à partir des entrées**, les charges, utilisations et estimations de files d'attente.

# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR, PROCESSED_DIR, SLICE_TYPES
from src.features import build_all, load_tables, ALL_FEATURES, FEATURE_GROUPS
from src.eda import (quality_report, impossible_values, fig_inventory, fig_targets, fig_delay_vs,
                     fig_rho_validation, fig_correlations, fig_corr_matrix)
from src.plotting import set_style, savefig
from src.variables import save_variables_table

set_style()
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
summary = {}   # every number quoted in the report is written to results/eda_summary.json

# %% [markdown]
# ## 1. Inventaire de l'archive

# %%
raw = load_tables()
out = build_all()                       # features calculées depuis les entrées (cache parquet)
scen, f, links, hops = out["scenarios"], out["flows"], out["links"], out["hops"]

n_tars = scen.tar_file.nunique()
inv = {
    "n_tarballs": int(n_tars),
    "n_scenarios_ok": int(len(scen)),
    "n_scenarios_skipped_not_ok": int(scen.groupby("tar_file").n_skipped_in_tar.first().sum()),
    "n_topologies": int(scen.topology.nunique()),
    "n_flows": int(len(f)),
    "n_hops": int(len(hops)),
    "n_links_used": int(len(links)),
    "flows_per_slice_type": f.slice_type.value_counts().to_dict(),
}
summary["inventory"] = inv
print(json.dumps(inv, indent=1))

# %%
desc = scen[["n_nodes", "n_core_nodes", "n_core_links", "n_links", "n_flows", "n_slices",
             "n_slices_eMBB", "n_slices_URLLC", "n_slices_mMTC", "sim_wallclock_s"]].describe().T
summary["scenario_describe"] = desc[["mean", "min", "50%", "max"]].round(2).to_dict(orient="index")
desc

# %%
topo = scen.groupby("topology").agg(n_scen=("scenario_id", "size"), n_core_nodes=("n_core_nodes", "first"),
                                    n_core_links=("n_core_links", "first"))
summary["topologies"] = {"n": int(len(topo)), "scen_per_topo_min": int(topo.n_scen.min()),
                         "scen_per_topo_median": float(topo.n_scen.median()), "scen_per_topo_max": int(topo.n_scen.max()),
                         "core_nodes_min": int(topo.n_core_nodes.min()), "core_nodes_max": int(topo.n_core_nodes.max())}
print(summary["topologies"])
fig = fig_inventory(scen); savefig(fig, "L1_01_inventaire"); plt.show()

# %% [markdown]
# **Structure d'un scénario.** Chaque graphe contient : un *cœur* issu de Topology Zoo (nœuds `inner`),
# des nœuds `antenna` (points d'accès radio), un nœud feuille `origin_*` par source de trafic accroché à une antenne,
# et des nœuds `destination_*` par type de slice. Le grand nombre de nœuds (`n_nodes`) vient donc surtout des
# sources mMTC (une par capteur), pas de la taille du cœur.

# %%
types = raw["links"].groupby(["src_type", "dst_type"]).size().sort_values(ascending=False).head(12)
types

# %% [markdown]
# ## 2. Qualité des données : types, manquants, doublons, valeurs impossibles

# %%
qr = quality_report(raw)
qr.to_csv(RESULTS_DIR / "quality_report.csv", index=False)
print("Colonnes avec valeurs manquantes :")
print(qr[qr.n_missing > 0][["table", "column", "n_missing", "n_rows"]].to_string(index=False))
print("\nClés dupliquées par table :", qr.groupby("table").n_duplicate_keys_in_table.first().to_dict())

# %% [markdown]
# Les seules valeurs manquantes attendues sont `meas_*` (statistiques mesurées) sur les liens que le simulateur
# ne rapporte pas, et `exp_max_factor` si un flux n'est pas poissonien. Ce sont des sorties ou des constantes, pas des entrées.

# %%
iv = impossible_values(f)
summary["impossible_values"] = iv
iv

# %%
const = {c: f[c].unique().tolist()[:5] for c in ["time_dist", "size_dist", "exp_max_factor"]}
const["max_avg_lambda"] = scen.max_avg_lambda.unique().tolist()[:5]
const["sched_policy"] = raw["links"].sched_policy.unique().tolist()
const["queue_size_pkts"] = sorted(hops.queue_size_pkts.unique().tolist())[:10]
const["rate_bps_by_slice"] = f.groupby("slice_type").rate_bps.unique().apply(lambda v: sorted(v.tolist())[:5]).to_dict()
summary["constants"] = {k: [str(x) for x in v] if isinstance(v, list) else {kk: [str(x) for x in vv] for kk, vv in v.items()} for k, v in const.items()}
const

# %% [markdown]
# **Vérification du débit réel (JOURNAL D0.3).** Le débit offert est calculé depuis les paramètres du générateur
# (`AvgPktsLambda × taille moyenne`). On le compare au débit généré mesuré par le simulateur.

# %%
ratio = f.avg_bw_kbps * 1000 / f.rate_bps
nominal_ratio = (f.rate_bps / f.nominal_bw).groupby(f.slice_type).median()
summary["rate_check"] = {"measured_over_rate_median": float(ratio.median()),
                         "measured_over_rate_p01": float(ratio.quantile(0.01)),
                         "measured_over_rate_p99": float(ratio.quantile(0.99)),
                         "rate_over_nominal_by_slice": nominal_ratio.round(4).to_dict()}
summary["rate_check"]

# %% [markdown]
# **Le routage simulé diffère souvent du chemin déclaré dans `slices_K.json` (JOURNAL D1.1).**
# Le parseur utilise `routing_K.txt`. Preuve : l'utilisation calculée avec ces chemins reproduit l'utilisation
# mesurée par le simulateur.

# %%
Lm = links[links.meas_util.notna()]
summary["routing_check"] = {
    "frac_flows_path_equal_slice_file": float(f.path_matches_slice_file.mean()),
    "frac_scenarios_all_paths_equal": float(f.groupby("scenario_id").path_matches_slice_file.all().mean()),
    "corr_min_rho1_vs_meas_util": float(np.corrcoef(np.minimum(Lm.rho_link, 1), Lm.meas_util)[0, 1]),
    "mae_min_rho1_vs_meas_util": float(np.abs(np.minimum(Lm.rho_link, 1) - Lm.meas_util).mean()),
    "frac_used_links_rho_gt_1": float((links.rho_link > 1).mean()),
}
summary["routing_check"]

# %%
fig = fig_rho_validation(links); savefig(fig, "L1_02_validation_rho"); plt.show()

# %% [markdown]
# ## 3. Distribution des cibles

# %%
d = f[f.has_delay]
tq = d.groupby("slice_type").delay.describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).round(6)
summary["delay_by_slice"] = tq.to_dict(orient="index")
summary["delay_global"] = {"min": float(d.delay.min()), "median": float(d.delay.median()), "max": float(d.delay.max()),
                           "max_over_min_orders_of_magnitude": float(np.log10(d.delay.max() / d.delay.min())),
                           "skew_raw": float(d.delay.skew()), "skew_log10": float(np.log10(d.delay).skew()),
                           "frac_delay_gt_1s": float((d.delay > 1).mean())}
summary["loss_by_slice"] = f.groupby("slice_type").loss.describe(percentiles=[0.5, 0.9]).round(4).to_dict(orient="index")
print(json.dumps(summary["delay_global"], indent=1))
tq

# %%
fig = fig_targets(f); savefig(fig, "L1_03_distributions_cibles"); plt.show()

# %% [markdown]
# ## 4. Délai en fonction des entrées

# %%
fig = fig_delay_vs(f, "path_len", np.arange(0.5, 16.5, 1), "longueur du chemin (nb de liens)",
                   "Délai vs longueur du chemin"); savefig(fig, "L1_04_delai_vs_longueur"); plt.show()

# %%
fig = fig_delay_vs(f, "rho_link_max", np.linspace(0, 3, 31), "ρ max sur le chemin (charge offerte / capacité)",
                   "Délai vs utilisation du lien le plus chargé"); savefig(fig, "L1_05_delai_vs_rho"); plt.show()

# %%
fig = fig_delay_vs(f, "wfq_share_min", np.logspace(-5, 0, 26), "part WFQ minimale garantie sur le chemin",
                   "Délai vs ressource WFQ allouée", logx=True); savefig(fig, "L1_06_delai_vs_wfq"); plt.show()

# %% [markdown]
# **Flux « affamés ».** Les délais supérieurs à la seconde correspondent à des flux dont la file WFQ, sur au moins
# un port, a un poids quasi nul (la slice n'a pas de réservation sur ce port car le routage simulé s'écarte du chemin
# prévu par l'admission). Vérification :

# %%
starv = pd.crosstab(f.wfq_share_min < 1e-3, pd.cut(f.delay, [-2, 0, 0.01, 0.1, 1, 1e3],
                                                   labels=["aucun paquet", "<10 ms", "10-100 ms", "0,1-1 s", ">1 s"]))
starv.index = ["part min ≥ 1e-3", "part min < 1e-3"]
summary["starvation_crosstab"] = starv.to_dict()
starv

# %% [markdown]
# Une part quasi nulle ne suffit pas : WFQ est *conservatif* (il redistribue la capacité inutilisée par les autres
# files). Une file de poids ~0 n'est donc affamée que si le lien est **aussi** saturé. On croise les deux conditions
# au niveau du saut :

# %%
hh = hops[["scenario_id", "flow_id", "wfq_share", "rho_link"]].copy()
hh["part_quasi_nulle"] = hh.wfq_share < 1e-3
hh["part_quasi_nulle_ET_lien_sature"] = hh.part_quasi_nulle & (hh.rho_link >= 1)
anyhop = hh.groupby(["scenario_id", "flow_id"])[["part_quasi_nulle", "part_quasi_nulle_ET_lien_sature"]].any().reset_index()
ff = f[["scenario_id", "flow_id", "delay"]].merge(anyhop)
starv2 = pd.crosstab([ff.part_quasi_nulle, ff.part_quasi_nulle_ET_lien_sature],
                     pd.cut(ff.delay, [-2, 0, 0.01, 0.1, 1, 1e4], labels=["aucun paquet", "<10 ms", "10-100 ms", "0,1-1 s", ">1 s"]))
summary["starvation_x_saturation"] = {f"part<1e-3={a}, part<1e-3 & rho>=1={b}": row.to_dict() for (a, b), row in starv2.iterrows()}
del hh, anyhop, ff
starv2

# %% [markdown]
# ## 5. Corrélations

# %%
fig, corr = fig_correlations(f, ALL_FEATURES); savefig(fig, "L1_07_correlations_cible"); plt.show()
summary["spearman_with_log_delay"] = corr.round(3).to_dict()

# %%
fig = fig_corr_matrix(f.sample(min(len(f), 200_000), random_state=0), ALL_FEATURES)
savefig(fig, "L1_08_correlations_entrees"); plt.show()

# %% [markdown]
# ## 6. Dictionnaire des variables (livrable L1)

# %%
vt = save_variables_table()
vt

# %%
with open(RESULTS_DIR / "eda_summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=1, ensure_ascii=False, default=str)
print("écrit : results/eda_summary.json")

# %% [markdown]
# ## 7. Ce qu'il faut retenir (chiffres : `results/eda_summary.json`)
#
# 1. **Volume.** 7 127 scénarios valides (773 écartés car la simulation n'a pas atteint l'état `OK`), 162 topologies,
#    1,58 M flux, 7,4 M sauts flux→lien. Assez de topologies (41 scénarios par topologie en médiane) pour un test
#    sur **topologies inédites**.
# 2. **Déséquilibre.** 75 % des flux sont mMTC, 21 % URLLC, 4 % eMBB → les métriques seront toujours rapportées
#    **par type de slice**, sinon mMTC domine tout.
# 3. **Cible très asymétrique.** Le délai couvre 8 ordres de grandeur (1e-5 s à ~1 900 s), asymétrie 38 en brut contre
#    0,99 en log10 → on prédit **log10(délai)**.
# 4. **Le profil de trafic est constant par type de slice** (débit, lois, file de 32 paquets) : la variabilité du délai
#    vient de la **topologie, du routage, de la compétition entre flux et de l'allocation WFQ**.
# 5. **ρ calculé depuis les entrées est fiable** (corrélation 0,984 avec l'utilisation mesurée) et le délai montre un
#    **coude net à ρ ≈ 1** (figure L1_05) : base du seuil d'événement en L4.
# 6. **Flux affamés.** ~2,9 % des flux ont un délai > 1 s ; presque tous ont une file WFQ de poids quasi nul **sur un lien
#    saturé**. Mais beaucoup de flux dans la même situation restent rapides : aucune règle simple ne sépare les deux,
#    il faut un modèle qui voit avec qui chaque file est en compétition → argument pour le GNN.
