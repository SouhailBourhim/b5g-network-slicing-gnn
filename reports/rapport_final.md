# AI-Based Dynamic Resource Allocation for B5G Network Slicing

INPT Rabat, Smart-ICT. Jeu de données : Farreras, Paillissé, Fàbrega, Vilà — *Generation of a network slicing
dataset: The foundations for AI-based B5G resource management*, Data in Brief 55 (2024) 110738,
Zenodo 10.5281/zenodo.10610616 (v2.2.1, CC BY 4.0).

> Tous les chiffres de ce rapport viennent d'un fichier de `results/`. Le détail méthodologique, les questions de
> jury et le code expliqué ligne par ligne sont dans `docs/explications/01` à `06`. Les décisions datées, avec leur
> raison chiffrée et l'alternative rejetée, sont dans `JOURNAL.md`.

## Résumé

À partir de 7 127 simulations paquet par paquet de réseaux de transport (topologie + capacités + routage + slices
eMBB/URLLC/mMTC + trafic), on apprend un modèle « what-if » qui prédit le délai de chaque flux sans relancer la
simulation (10,8 h en moyenne). On compare une baseline naïve, des formules de files d'attente (M/M/1, M/M/1/K),
deux modèles plats (Ridge, LightGBM) et un GNN hétérogène flux/lien. Le GNN atteint une MAE_log de **0,128** sur
test, contre **0,352** pour le meilleur modèle plat (LightGBM) et **0,822** pour la baseline naïve — un gain
significatif (intervalle de confiance bootstrap qui exclut 0) qui **croît avec la sévérité de la charge** : × 1,9
sur les flux calmes, **× 3,3 sur les flux affamés**. En revanche, rien ne montre que le GNN généralise mieux à des
topologies inédites une fois la charge égalisée.

## 1. Introduction

**Le problème.** Avant d'admettre une nouvelle slice ou de changer un routage, un orchestrateur B5G doit savoir si
les slices déjà admises respecteront encore leur SLA. Le simuler prend des heures ; un modèle appris répond en
millisecondes. C'est la motivation du jeu de données et des modèles de performance par graphe (RouteNet,
GNNetSlice).

**La question.** Un GNN qui voit explicitement les liens et les files partagés entre flux apporte-t-il un gain
mesurable sur un modèle plat recevant les mêmes informations résumées à la main — et si oui, dans quelles
conditions ?

**Organisation du projet.** Trois volets en parallèle : **A** (GNN et modèles neuronaux), **B** (réseau,
baselines analytiques, événements), **C** (données, formulation, ML classique). Répartition complète :
`docs/plan_de_travail_equipe.docx`.

## 2. L1 — Exploration des données

Parsing validé contre l'API officielle (`tests/test_parsing.py`) : **7 127 scénarios** valides, **162 topologies**
(Internet Topology Zoo), **1 576 669 flux**, **7 446 192 sauts** flux→lien. Une simulation coûte en moyenne **10,8 h**
(médiane 9,4 h) — le motif du modèle « what-if ».

Le délai couvre **8,3 ordres de grandeur** (1e-5 s à ~1 900 s), asymétrie 38 en brut contre 0,99 en log10 : la cible
est donc **log10(délai)** (figure `L1_03_distributions_cibles.png`). Le délai médian explose d'un facteur ~10 entre
ρ_max = 0,95 et 1,05 (figure `L1_05_delai_vs_rho.png`) — le coude qui fixera le seuil d'événement en L4. Les
champs de trafic (loi d'inter-arrivée, taille de paquets) sont **constants par type de slice** une fois le débit
réel recalculé depuis les paramètres du générateur (D0.3) : la variabilité du délai vient de la topologie, du
routage et de la compétition entre flux, pas du profil de trafic individuel — un argument de poids en faveur d'un
modèle relationnel. 9 flux sans paquet reçu sont exclus de la régression (délai non défini).

Détail complet, dictionnaire des variables et justification entrée/cible/exclue : `docs/explications/01_exploration.md`,
`reports/variables.md`.

## 3. L2 — Formulation et protocole d'évaluation

**Tâche** : régression de log10(délai moyen) par flux. **Hypothèses** (tranchées en §7) : H1 (le GNN bat un modèle
plat à mêmes entrées, surtout à forte charge), H2 (tous les modèles se dégradent sur les événements), H3 (le GNN
généralise mieux aux topologies inédites), H4 (les flux affamés sont où les modèles plats échouent le plus).

**La fuite de données, mesurée et non seulement argumentée.** Les flux d'un même scénario partagent des liens et
des files : un split aléatoire par flux place des quasi-jumeaux de part et d'autre. Expérience contrôlée (même
modèle, mêmes variables, 3 graines, `src/leakage.py`) :

| Schéma de split | Erreur annoncée | Erreur réelle (scénarios jamais vus) | Facteur |
|---|---|---|---|
| Aléatoire par flux | 0,127 ± 0,005 | 0,399 ± 0,013 | **× 3,15** |
| Par scénario | 0,358 ± 0,024 | 0,410 ± 0,017 | × 1,14 |

**Découpage gelé** (`src/splits.py`, graine 42) : 24 topologies (~15 % par quartile de taille) tenues à l'écart en
premier → `test_unseen_topo` (1 080 scénarios) ; le reste réparti 70/15/15 par scénario, stratifié par tercile de
charge → train (4 233), val (907), test (907). **Limite à garder en tête** : les topologies inédites sont moins
chargées (ρ_max médian 1,35 contre 1,63) — reprise en §7 pour H3.

Détail, code et questions de jury : `docs/explications/02_problematique.md`.

## 4. L3 — Baselines tabulaires et importance des variables

Deux modèles plats sur `ALL_FEATURES` (27 variables, aucune issue de la simulation) : **Ridge** (régularisation
choisie sur une grille complète, val) et **LightGBM** (arrêt précoce sur val, 5 graines). MAE_log test :

| Modèle | val | test | topologies inédites |
|---|---|---|---|
| naive_median_slice (plancher) | 0,815 | 0,822 | 0,790 |
| Ridge | 0,499 | 0,510 [0,481 ; 0,538] | 0,453 [0,424 ; 0,481] |
| **LightGBM** (5 graines) | 0,348 ± 0,002 | **0,352 [0,326 ; 0,379]** | 0,321 [0,294 ; 0,350] |

LightGBM bat Ridge d'un facteur 1,45 : le délai dépend d'effets de seuil (près de ρ = 1) et d'interactions qu'un
modèle linéaire ne représente pas nativement. La grille d'α de Ridge est quasiment plate (variation en 5ᵉ décimale
de α = 1e-3 à 10) : avec 934 542 lignes de train pour 27 variables, la régularisation n'a simplement pas de rôle à
jouer dans cette plage — la limite de Ridge est sa forme linéaire, pas son réglage.

**Importance des variables** (gain LightGBM et SHAP, figure `L3_02_shap_vs_gain.png`) : les deux méthodes
indépendantes s'accordent sur le top 3 — `rho_link_max` (36 % du gain, 21 % du SHAP), puis les termes de file
M/M/1/K. **Ablation par groupe** (réentraînement sans le groupe, figure `L3_03_ablation.png`) : retirer
`utilization` coûte **+0,058** de MAE_log, `wfq` **+0,034** — très au-dessus du bruit entre graines (0,0019) ; les
groupes `capacity`, `path`, `scenario`, `traffic` sont dans le bruit. **Ce qui porte le signal, c'est la charge et
la garantie de service, pas le trafic brut** — une conclusion retrouvée indépendamment par le GNN (§6).

Détail, fonctions expliquées, questions de jury : `docs/explications/03_baselines.md`.

## 5. L4 — Baselines analytiques et événements

**Baselines physiques**, sans apprentissage, calculées depuis les entrées (ρ, capacités, WFQ) :

| Modèle | MAE_log test |
|---|---|
| `analytic_mm1` (M/M/1, ρ écrêté à 0,99) | **0,649** |
| `analytic_mm1k_link_cal` (M/M/1/K calibré sur train) | 0,672 |
| `analytic_mm1k_link` (M/M/1/K, buffer fini) | 0,839 |
| `analytic_mm1k_wfq` (vue débit garanti WFQ) | 1,141 |

M/M/1 simple bat les variantes plus réalistes en moyenne — une régression de M/M/1/K non entièrement expliquée
(limite assumée, §7).

**Événements**, définis depuis les seules entrées (jamais depuis le délai, pour rester utilisables en prédiction) :
congestion = ρ_max du chemin ≥ τ ; famine = part WFQ < 1e-3 sur un lien à ρ ≥ τ. Seuil **τ = 1,05**, choisi sur
train au coude de la courbe délai médian vs ρ (saut de +0,742 en log10 entre (1,0 ; 1,05] et (1,05 ; 1,1], contre
+0,134 juste avant). Conséquence inattendue : **82,9 % des flux de test sont « en événement »** — ce jeu simule
surtout des réseaux chargés, ce n'est pas un seuil truqué. Un palier « sévère » (τ = 2,0, 16 % des flux) sert de
comparaison plus discriminante en §7.

Détail, vérification visuelle, questions de jury : `docs/explications/04_evenements.md`.

## 6. L5 — Représentation en graphe et GNN

**Le graphe.** Un `HeteroData` par scénario : nœuds **flux** (type de slice, δ, débit, taille de paquet, longueur
du chemin) et **lien** (capacité, ρ, nb de flux, M/M/1/K) ; arêtes **flux↔lien** (un par saut), avec en attribut ce
qui appartient au **couple** — la part WFQ garantie à ce flux sur ce lien précis. Figure `L5_01_exemple_graphe.png` :
un scénario où 7 flux partagent le même lien saturé, le couplage que l'approche cible.

**Le modèle.** `HeteroConv` alternant liens←flux puis flux←liens pendant **K = 3** itérations (résidu + LayerNorm),
puis un MLP de lecture. Deux convolutions : **GATv2** (attention par voisin, `edge_dim`, 868 481 paramètres) et
EdgeSAGE (moyenne, message voit l'arête). K = 3 fait voir à un flux les liens de ses concurrents directs — la
congestion qui se propage. GCN est écarté : normalisation homogène par le seul degré, incapable de lire un
attribut d'arête ni de distinguer flux et lien.

**Entraînement** (`src/train_gnn.py`, GPU) : MSE sur log10(délai), AdamW, arrêt précoce sur la MAE_log de val. 76
époques (141 min), meilleure époque 61. Train et val descendent ensemble jusqu'à ~36, puis val plafonne pendant que
train continue — surapprentissage visible mais contrôlé (figure `L5_02_courbes_apprentissage.png`).

**Résultat et les deux ablations** :

| Modèle | val | test | topologies inédites |
|---|---|---|---|
| `gnn_k0` (K = 0, pas de passage de messages) | 0,785 | 0,792 | 0,735 |
| `mlp_flat` (mêmes 8 variables du flux, sans graphe) | 0,407 | 0,403 ± 0,002 (3 graines) | 0,372 |
| **`gnn_gatv2`** | **0,128** | **0,128 [0,118 ; 0,138]** | 0,115 [0,105 ; 0,126] |

`gnn_k0` (0,792) n'est qu'à peine meilleur que la baseline naïve : avec K = 0, les nœuds lien et les attributs de
saut sont **déconnectés** de la lecture, donc ce modèle mesure l'apport du **graphe entier**, pas du passage de
messages seul. La mesure propre, à information strictement égale, est `mlp_flat` → `gnn_gatv2` : **× 3,1**.

**Limite assumée** : pas de recherche d'hyperparamètres (une exécution coûte 2,4 h de GPU ; la grille prévue en
aurait coûté 30 à 60) — la configuration est défendue par les courbes d'apprentissage, pas par un réglage.

Détail, équations de mise à jour, fonctions expliquées : `docs/explications/05_gnn.md`.

## 7. L6 — Comparaison, discussion, limites

### H1 — confirmée, contre le meilleur adversaire

`lgbm`, pas `mlp_flat`, est le modèle plat le plus fort du projet (0,352). Différence appariée par bootstrap sur
les scénarios (1 000 rééchantillonnages, volet C) : `gnn_gatv2 − lgbm` = **−0,2243 [−0,2484 ; −0,2023]**, le GNN
meilleur dans **100 %** des tirages. Le gain annoncé est **× 2,75**, une comparaison plus dure à réfuter que le
× 3,1 contre `mlp_flat` (qui isole le passage de messages à information égale ; `lgbm`, lui, reçoit déjà les
meilleurs agrégats manuels possibles et perd quand même).

### H2 — vraie seulement au palier sévère

Au seuil principal (τ = 1,05, 83 % des flux en événement), un modèle qui ignore la charge (naïf, `gnn_k0`) se
retrouve calibré sur la majorité événementielle et **s'améliore** sur elle (naïf : 1,322 → 0,719) — un artefact de
proportion, pas une propriété des modèles. Au palier sévère (τ = 2,0, 16 % des flux), H2 tient en entier :

| | reste | sévère | dégradation |
|---|---|---|---|
| naïf | 0,770 | 1,098 | +0,328 |
| `gnn_k0` | 0,737 | 1,078 | +0,340 |
| `lgbm` | 0,325 | 0,491 | +0,166 |
| **`gnn_gatv2`** | **0,124** | **0,145** | **+0,021** |

Le GNN se dégrade **8 fois moins** que `lgbm`.

### H3 — infirmée

L'écart brut « inédites − test » est négatif pour tous les modèles (apparence de meilleure généralisation sur
topologies inédites), y compris pour le GNN (−0,0125 [−0,0273 ; +0,0022]). Une fois la charge égalisée par classes
de ρ (volet C), il tombe à **−0,0039 [−0,0189 ; +0,0119]** — l'intervalle contient 0. `gnn_k0` montre le mécanisme :
significatif en brut (−0,0574 [−0,1141 ; −0,0044]), nul à charge égale (+0,0007). **Le GNN transfère sans perte**
vers des topologies inédites (comportement attendu d'un modèle à poids partagés), mais rien ne montre qu'il y
généralise *mieux* : l'écart apparent était un effet de charge, pas de structure.

### H4 — confirmée, sans ambiguïté

MAE_log test par type d'événement (`none` 34 632 flux, `congestion` 148 514, `both` = congestion+famine 18 807) :

| | none | congestion | both | lgbm / gnn |
|---|---|---|---|---|
| `gnn_gatv2` | 0,121 | 0,106 | **0,315** | — |
| `lgbm` | 0,229 | 0,295 | 1,026 | — |
| ratio `lgbm`/`gnn` | | | | 1,9× → 2,8× → **3,3×** |

L'avantage relatif du GNN **croît avec la sévérité** — exactement le régime où une décision d'admission de slice
coûte le plus cher à se tromper. En absolu, même le GNN se dégrade fortement sur `both` (0,106 → 0,315) : il n'est
pas parfait sur la famine, seulement nettement le moins mauvais.

### Analyse d'erreur (figure `L6_01_erreur_vs_rho.png`)

Sur tout le spectre de ρ, le GNN reste la courbe la plus basse à toute charge ; les deux modèles aveugles à la
charge (naïf, `gnn_k0`) sont les plus hauts et quasi plats, faute d'information pour réagir à ρ ; `lgbm` et
`mlp_flat` se dégradent progressivement après le coude de τ = 1,05.

### Limites

- **Données simulées**, réseau de transport sans RAN, scénarios statiques : un événement est un état, pas un
  instant.
- **M/M/1/K sous-performe M/M/1** sans explication établie (pistes non vérifiées : plafond de buffer, hypothèse de
  Poisson).
- **Pas de recherche d'hyperparamètres ni d'ablation par groupe sur le GNN** (coût GPU) ; compensée par l'ablation
  équivalente sur LightGBM, qui converge vers la même conclusion (charge et WFQ dominent).
- **Une seule graine de GNN** pour les comparaisons H1/H3 (`lgbm` et `mlp_flat` en ont 5 et 3) ; le bruit de graine
  mesuré sur `mlp_flat` est faible (0,5 % de la valeur) mais ce n'est pas une preuve directe pour le GNN.
- **Le bootstrap suppose l'indépendance entre scénarios** une fois tirés ; deux scénarios de la même topologie
  restent corrélés par construction.

## 8. Conclusion

*Dans quelles conditions une approche GNN apporte-t-elle une valeur ajoutée pour la gestion dynamique des slices
B5G ?*

**Toujours, par rapport au meilleur modèle plat testé** — × 2,75 contre LightGBM, intervalle de confiance qui
exclut 0. **Particulièrement dans le régime chargé et à risque** : l'avantage se creuse avec la sévérité, de × 1,9
sur les flux calmes à **× 3,3 sur les flux affamés** — le régime où une décision d'admission de slice coûte le plus
cher à se tromper. En revanche, le GNN ne généralise pas *mieux* qu'un modèle plat à des topologies inédites une
fois la charge égalisée : son avantage vient de ce qu'il modélise la **charge partagée entre flux**, pas d'une
capacité supérieure à transférer vers des structures nouvelles. Le gain n'est donc pas générique — il est
spécifiquement celui de voir, par construction, la compétition entre flux sur les liens et les files qu'ils
partagent.
