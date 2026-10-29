# 06 — Comparaison, discussion, limites (L6)

Document commun aux trois volets. Chiffres : `results/table_final.md`, `results/bootstrap_*.csv`,

> **Mise à jour D3.16** : une 2e graine de `gnn_gatv2` est arrivée après la rédaction de ce fichier (MAE_log test moyenne sur 2 graines : **0,135 ± 0,0102**, contre 0,128 à 1 graine). Les conclusions (H1, H3) ne changent pas, seules les décimales bougent légèrement. Chiffres à jour : `notebooks/L5_gnn.py`/`L6_comparaison.py` (recalculent depuis `results/`) et `JOURNAL.md` D3.16.
`results/metrics_all.csv`. Détail : `notebooks/L6_comparaison.py` (+ `.ipynb` exécuté), figure `L6_01_erreur_vs_rho.png`.

## Ce qu'on a fait

On a réuni les prédictions des trois volets dans un seul tableau (`src/final_table.py`), répondu explicitement
aux quatre hypothèses de L2 avec des intervalles de confiance plutôt que des points isolés, tracé l'erreur de
chaque modèle en fonction de la charge sur tout le spectre de ρ (pas seulement au seuil événement/non-événement),
et rassemblé les deux ablations du projet (GNN K=0, LightGBM par groupe) pour vérifier qu'elles racontent la même
histoire.

## Pourquoi ces choix

| Choix | Alternative rejetée | Pourquoi |
|---|---|---|
| Comparer le GNN au **meilleur** modèle plat (`lgbm`), pas au plus faible | Comparer seulement à `mlp_flat` | `lgbm` reçoit plus d'information (agrégats du chemin) ; le battre lui est une preuve plus forte que battre un MLP sur les seules variables du flux. |
| Bootstrap **apparié** (mêmes tirages de scénarios pour les deux modèles comparés) | Deux IC calculés séparément, puis comparés | Resserre l'intervalle de la différence : la variance due au tirage de scénarios s'annule dans la soustraction quand elle est partagée. |
| Repondération par classe de charge pour comparer test et topologies inédites | Comparer les MAE brutes | Les topologies inédites sont moins chargées (D2.1) ; sans repondération, on confond « topologie nouvelle » et « charge plus faible ». |
| `event_type` (none / congestion / both) plutôt que `is_event` binaire pour H4 | Binaire événement/non-événement | La famine implique la congestion (D4.3), donc le binaire mélange « juste chargé » et « chargé et affamé » dans la même case `event` ; `event_type` les sépare. |

## Ce que montrent les résultats

**Le tableau final** (`results/table_final.md`) réunit 10 modèles × {test, topologies inédites} × {tout, événement,
non-événement} × type de slice, chacun avec un intervalle de confiance à 95 %.

### H1 — un GNN qui voit les liens partagés bat un modèle plat aux mêmes entrées, surtout à forte charge

**Confirmée.** `gnn_gatv2` : 0,128 [0,118 ; 0,138] contre `lgbm` 0,352 [0,326 ; 0,379] — différence appariée
−0,2243 [−0,2484 ; −0,2023], le GNN meilleur dans 100 % des 1000 rééchantillonnages. Deux mesures complémentaires :
`mlp_flat` → `gnn_gatv2` isole le passage de messages à information égale (× 3,1) ; `lgbm` → `gnn_gatv2` compare à
la meilleure agrégation manuelle possible (× 2,75). Les deux confirment H1 ; la seconde est la plus dure à réfuter.

### H2 — tous les modèles se dégradent sur les événements, les modèles plats davantage

**Vraie uniquement au palier sévère.** Au seuil principal (τ = 1,05, 83 % des flux en événement), un modèle qui
ignore la charge (naïf, `gnn_k0`) se retrouve calibré sur la majorité événementielle et *s'améliore* sur elle —
un artefact de proportion, pas une propriété du modèle. Au palier sévère (τ = 2,0, 16 % des flux), presque tous les
modèles se dégradent, naïf et `gnn_k0` compris (MAE_log test, reste → sévère) : naïf 0,770 → 1,098, `gnn_k0`
0,737 → 1,078, `analytic_mm1` 0,585 → 0,983, `mlp_flat` 0,373 → 0,561, `lgbm` 0,325 → 0,491, `gnn_gatv2`
**0,124 → 0,145**. Le GNN se dégrade 8 fois moins que `lgbm`.

### H3 — le GNN généralise mieux aux topologies inédites

**Infirmée.** L'écart brut « inédites − test » est négatif pour tous les modèles, GNN compris (−0,0125
[−0,0273 ; +0,0022]), mais tombe à −0,0039 [−0,0189 ; +0,0119] une fois la charge égalisée par classes de ρ — un
intervalle qui contient 0. `gnn_k0` montre le mécanisme en clair : significatif en brut (−0,0574
[−0,1141 ; −0,0044]), nul à charge égale (+0,0007 [−0,0592 ; +0,0545]). Conclusion : le GNN **transfère sans
perte** vers des topologies inédites (comportement attendu d'un modèle à poids partagés), mais rien ne montre qu'il
y généralise *mieux* qu'un modèle plat — l'avantage apparent était un effet de charge, pas de structure.

### H4 — les flux affamés sont là où les modèles plats échouent le plus

**Confirmée, sans ambiguïté.** MAE_log test par `event_type` : sur `none` (34 632 flux) le GNN fait 1,9× mieux que
`lgbm` ; sur `congestion` (148 514 flux), 2,8× mieux ; sur **`both`** (congestion + famine, 18 807 flux, 9,3 %),
**3,3× mieux** (0,315 contre 1,026). L'avantage relatif du GNN croît avec la sévérité — exactement le régime où une
décision d'admission de slice coûte le plus cher à se tromper. En absolu, le GNN se dégrade aussi sur `both`
(0,106 → 0,315) : il n'est pas parfait, seulement le moins mauvais, et nettement.

### Analyse d'erreur (figure `L6_01_erreur_vs_rho.png`)

Sur tout le spectre de ρ, pas seulement au seuil binaire : le GNN reste la courbe la plus basse à toute charge, les
deux modèles aveugles à la charge (`naive`, `gnn_k0`) sont les plus hauts et quasi plats (ils n'ont par construction
aucune information pour réagir à ρ), `lgbm` et `mlp_flat` se dégradent progressivement après le coude de τ = 1,05.

### Les deux ablations du projet convergent

| Groupe retiré (LightGBM) | Δ MAE_log test |
|---|---|
| `utilization` (ρ, concurrence) | **+0,058** |
| `wfq` (part garantie) | **+0,034** |
| `queueing`, `slice` | +0,006 à +0,009 |
| `capacity`, `path`, `scenario`, `traffic` | dans le bruit (±0,005) |

Même conclusion que le GNN (volet A, D3.12) : ce qui porte le signal, c'est la **charge et la garantie de
service**, pas le profil de trafic brut. Les deux volets, avec des méthodes indépendantes (réentraînement sans
groupe vs K = 0), arrivent à la même hiérarchie de variables — un recoupement qui renforce la confiance dans le
résultat plus que chaque ablation seule.

## Questions probables du jury

**Le GNN gagne partout : pourquoi s'embêter avec les baselines plates ?**
Parce qu'elles bornent l'affirmation. Sans `lgbm`, on ne saurait pas si le gain du GNN vient du graphe ou du fait
qu'il est simplement un modèle plus gros ; sans `analytic_mm1`, on ne saurait pas si un modèle appris apporte quoi
que ce soit par rapport à une formule de physique des files sans aucune donnée. Chaque baseline élimine une
explication alternative.

**Le gain est-il le même partout, ou concentré quelque part ?**
Concentré, et c'est la réponse la plus intéressante : il croît avec la sévérité (× 1,9 sur les flux calmes, × 3,3
sur les flux affamés). Un gain uniforme aurait été moins convaincant qu'un gain qui se renforce justement là où la
décision d'admission de slice est la plus risquée.

**Pourquoi H3 est-elle infirmée alors que le GNN a la meilleure architecture pour généraliser (poids partagés) ?**
Avoir la bonne architecture pour généraliser ne garantit pas un avantage *mesuré* sur ce jeu précis : ici, l'écart
apparent entre test et topologies inédites était presque entièrement un effet de charge (24 topologies tenues à
l'écart, en moyenne moins chargées). Le GNN généralise sans perte — ce qui confirme l'architecture — mais ne
généralise pas *mieux* que des modèles sans cette propriété, parce que le problème qui expliquait l'écart apparent
(la charge) n'a rien à voir avec l'architecture.

**Quel est le résultat le plus fragile du projet ?**
H1 et H4 reposent sur un bootstrap à 1000 répétitions avec des intervalles qui excluent clairement 0 : solides.
H3 et la régression inattendue de M/M/1/K (volet B) sont les deux résultats les moins expliqués — on les rapporte
tels quels plutôt que de forcer une explication non vérifiée.

## Le code à connaître

### `bootstrap_groups` (`src/bootstrap.py`, volet C) — la différence appariée

```python
W = _weights(S, B, SEED)                              # memes tirages de scenarios pour tous les modeles
den = W @ N.to_numpy(); res["mae"] = (W @ A.to_numpy()) / den
if ref in models:
    d = M[:, j] - M[:, jr]                             # difference calculee tirage par tirage
    lo, hi = np.nanpercentile(d[1:], [2.5, 97.5])
```
La même matrice de poids `W` sert à tous les modèles : la différence `modèle − référence` est calculée **à
l'intérieur** de chaque tirage bootstrap, pas entre deux intervalles séparés. C'est ce qui permet à l'intervalle de
la différence d'exclure 0 alors que les deux intervalles individuels (gnn_gatv2 et lgbm) ne se chevauchent même pas
visuellement de toute façon — mais pour des cas plus proches, l'appariement est ce qui tranche.

### `fig_error_vs_rho` (`src/eda.py`) — une courbe par modèle, pas par slice

```python
for m, c in zip(models, colors):
    d = frame[frame.model == m]
    b = binned_quantiles(d.rho_link_max, d.abs_err, bins)   # meme fonction que les figures L1, par groupe "modele"
    ax.plot(b.x, b.q50, color=c, label=m, marker="o", ms=3)
```
Réutilise `binned_quantiles` (déjà écrite pour L1, `fig_delay_vs`) en groupant par modèle au lieu de type de slice
— un seul outil partagé pour deux questions différentes (« le délai dépend-il de ρ ? » en L1, « l'erreur
dépend-elle de ρ ? » ici).
