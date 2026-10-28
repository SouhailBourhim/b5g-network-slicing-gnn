# 02 — Formulation du problème et protocole d'évaluation (L2)

Les chiffres viennent de `results/leakage_experiment.csv`, `results/leakage_summary.json`, `results/splits_summary.json`
et `results/eda_summary.json`.

## Ce qu'on a fait

On a fixé la tâche : prédire, pour chaque flux d'une configuration réseau, son **délai moyen** (en log10).
On a ensuite **prouvé par une expérience** qu'il ne faut pas découper les données flux par flux : les flux d'un
même scénario se ressemblent, et un modèle évalué sur des « frères » de ses flux d'entraînement paraît bien meilleur
qu'il ne l'est. Enfin on a gelé les découpages : un jeu de test de **topologies jamais vues**, puis train / val / test
**par scénario** sur le reste. Tout le monde dans l'équipe utilise ces mêmes fichiers.

**Définitions utiles.**
- *Fuite de données* : de l'information du jeu de test qui se retrouve, sous une forme indirecte, dans l'entraînement.
- *Modèle « what-if »* : un modèle qui répond à « que se passe-t-il si je choisis cette configuration ? » sans
  lancer la simulation.
- *Split groupé* : on découpe par groupes entiers (ici les scénarios), jamais à l'intérieur d'un groupe.

## Pourquoi ces choix

| Choix | Alternative rejetée | Pourquoi (chiffre ou principe) |
|---|---|---|
| Cible = délai par flux, en log10 | Délai brut | Le délai couvre 8,3 ordres de grandeur ; asymétrie 38 en brut, 0,99 en log10 (L1). Erreur en log = erreur relative. |
| Unité de découpage = **scénario** | Flux tirés au hasard | Expérience de fuite : l'erreur annoncée est 3,15 fois trop faible avec le split par flux (ci-dessous). |
| Topologies inédites découpées **en premier** | Les tirer après le split des scénarios | Sinon une topologie du test inédit pourrait apparaître dans le train (fuite par la topologie). |
| Topologies inédites tirées dans chaque quartile de taille | Tirage uniforme | 162 topologies : un tirage uniforme pouvait ne retenir que des petits réseaux. Là on garde des petits et des grands (figure `L2_02`). |
| 70/15/15 stratifié par charge (tercile de ρ max) | Tirage simple | Garantit que les scénarios congestionnés sont dans train, val et test en même proportion (ρ max médian : 1,63 / 1,61 / 1,65). |
| Scaler/encodeur ajustés sur train seulement | Ajuster sur tout | Sinon la moyenne et l'écart-type du test « fuient » dans l'entraînement. |
| Métriques sur log10(délai) + MdAPE en % + par type de slice | MAE sur le délai brut ; une seule moyenne | mMTC fait 75 % des flux : une moyenne globale cacherait les erreurs sur eMBB et URLLC. |

## Ce que montrent les résultats

**L'expérience de fuite** (figure `L2_01_fuite.png`). Même modèle (LightGBM, paramètres fixes), mêmes variables,
1 500 scénarios, 3 graines. 20 % des scénarios sont mis de côté et jamais utilisés par aucun schéma (E, environ 64 000 flux).

| Schéma | Erreur estimée (MAE sur log10) | Erreur réelle sur E | Facteur |
|---|---|---|---|
| A : split aléatoire des flux | 0,127 ± 0,005 | 0,399 ± 0,013 | **3,15 ×** |
| B : split des scénarios | 0,358 ± 0,024 | 0,410 ± 0,017 | 1,14 × |

Avec A, 100 % des flux de test ont des frères dans l'entraînement (0 % avec B). Un étudiant qui publierait l'erreur
du schéma A annoncerait 0,13 alors que le modèle fait en réalité 0,40 sur de nouvelles configurations. Pour B,
l'estimation reste un peu optimiste (écart 0,05) : c'est de l'ordre de deux écarts-types entre graines, donc de la
variance entre scénarios plus que de la fuite. Un écart de 0,1 en log10 = erreur typique d'un facteur 1,26 sur le délai.

**Les découpages** (figure `L2_02_splits.png`) :

| Split | Scénarios | Topologies | ρ max médian |
|---|---|---|---|
| train | 4 233 | 137 | 1,63 |
| val | 907 | 127 | 1,61 |
| test | 907 | 134 | 1,65 |
| test_unseen_topo | 1 080 | 24 (jamais vues) | **1,35** |

Chaque slice est présente dans chaque split.

**Limite à garder en tête.** Les topologies inédites sont moins chargées (1,35 contre 1,63). En L6, une différence
d'erreur sur ce test pourra venir de la topologie nouvelle **ou** de la charge plus faible ; on comparera donc
aussi à charge égale (par tranche de ρ max).

## Hypothèses

- **H1** — Un GNN voyant les liens et files partagés bat un modèle plat aux mêmes entrées, surtout à forte charge.
- **H2** — Tous les modèles se dégradent sur les événements, les modèles plats davantage.
- **H3** — Le GNN généralise mieux aux topologies inédites.
- **H4** — Les flux affamés sont là où les modèles plats échouent le plus.

On ne dit pas qu'elles sont vraies : L6 les confirme ou les infirme avec les chiffres, y compris si le résultat est
décevant pour le GNN.

## Questions probables du jury

**Pourquoi ne pas splitter aléatoirement ?**
Parce que les flux d'un scénario partagent des liens et des files : ils sont quasi dupliqués. Mesuré : avec un split
aléatoire par flux l'erreur annoncée est 0,13, la vraie 0,40 (3,15 fois plus). Avec un split par scénario, 0,36 annoncée
contre 0,41 réelle.

**Pourquoi découper les topologies avant les scénarios ?**
Si on découpait d'abord les scénarios, un scénario du train et un du test inédit pourraient avoir la même topologie ;
le test ne serait plus « inédit ». On les isole en premier, puis on découpe le reste.

**Qu'est-ce que votre test « topologies inédites » mesure vraiment ?**
La généralisation à un réseau jamais vu. Mais ces 24 topologies sont moins chargées (1,35 contre 1,63), donc on
compare aussi à charge égale avant de conclure.

**Pourquoi ce n'est pas du « data snooping » de choisir les topologies inédites avec une graine ?**
Elles sont tirées au hasard (graine 42) dans chaque quartile de taille, avant tout entraînement, et fixées dans
`results/splits.csv`. On ne les a pas choisies en regardant les résultats.

**Pourquoi un modèle « what-if » plutôt que lancer le simulateur ?**
Une simulation prend en moyenne 10,8 h (médiane 9,4 h) ; un orchestrateur doit répondre à « si j'admets cette slice,
les SLA tiennent-ils ? » en millisecondes, pour explorer beaucoup de configurations.

## Le code à connaître

### 1. `holdout_topologies` et `make_splits` (`src/splits.py`) — le découpage

```python
topo["size_bin"] = pd.qcut(topo.n_core_nodes, 4, labels=["S", "M", "L", "XL"])   # 4 quartiles de taille
for _, part in topo.groupby("size_bin"):
    k = max(1, round(frac * len(part)))                       # 15 % de chaque quartile
    held += rng.choice(part.topology.to_numpy(), size=k, replace=False).tolist()

scen["split"] = "test_unseen_topo"                            # par défaut : topologie inédite
pool = ~scen.topology.isin(held)                              # scénarios dont la topologie n'est PAS retenue
scen.loc[pool, "split"] = grouped_split(scen[pool], seed=seed)  # train/val/test sur ceux-là
```
On retient des **topologies** (pas des scénarios) en premier ; leurs scénarios vont tous dans `test_unseen_topo`.
Le reste est découpé par scénario. Une graine fixe rend le résultat identique à chaque exécution (testé).

### 2. `run` (`src/leakage.py`) — l'expérience de fuite

```python
E = df[df.scenario_id.isin(scen[:n_new])]                     # scénarios jamais vus par A ni B
rest = df[df.scenario_id.isin(scen[n_new:])]
is_test_a = rng.random(len(rest)) < 0.2                       # A : 20 % des FLUX au hasard
A_train, A_test = rest[~is_test_a], rest[is_test_a]
test_scen = set(rest_scen[: int(0.2 * len(rest_scen))])       # B : 20 % des SCÉNARIOS entiers
in_b = rest.scenario_id.isin(test_scen)
B_train, B_test = rest[~in_b], rest[in_b]
leak_a = A_test.scenario_id.isin(A_train.scenario_id).mean()  # fuite mesurée : 1,0 pour A, 0,0 pour B
```
Même modèle, mêmes données ; seul le découpage change. On mesure deux erreurs par schéma : sur son propre test (ce
qu'il annoncerait) et sur E (ce qui arriverait en vrai). Le rapport entre les deux est le facteur d'optimisme.

### 3. `regression_metrics` (`src/metrics.py`)

```python
err = p - y                                                    # erreur en log10
ape = np.abs(10.0 ** p - 10.0 ** y) / 10.0 ** y               # erreur relative sur le délai en secondes
return {"mae_log": np.mean(np.abs(err)), "mse_log": np.mean(err ** 2), ..., "mdape_pct": np.median(ape) * 100}
```
Un `mae_log` de 0,1 se lit « le modèle se trompe en moyenne d'un facteur 10^0,1 = 1,26 ». `mse_log` pénalise plus
les grosses erreurs, typiquement un flux affamé prédit rapide.
