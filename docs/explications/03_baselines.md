> **Brouillon rédigé par Claude à partir du code et des résultats du volet C** (`src/baselines_ml.py`,
> `src/bootstrap.py`, `results/`), pour ne pas bloquer L6 pendant que le volet C finalise son propre travail.
> À relire, corriger et signer (JOURNAL) avant la soutenance — c'est la personne qui a écrit ce code qui doit
> pouvoir en répondre seule au jury.

# 03 — Baselines tabulaires et importance des variables (L3)

Les chiffres viennent de `results/metrics_all.csv`, `results/ridge_alpha_grid.csv`, `results/importance_shap_gain.csv`,
`results/bootstrap_ci.csv`/`bootstrap_diff.csv`. Figures : `L3_01_courbe_apprentissage.png`,
`L3_02_shap_*.png`, `L3_03_ablation.png`.

## Ce qu'on a fait

On a entraîné deux modèles « plats » (une ligne = un flux, un vecteur de variables résumées à la main) sur
`ALL_FEATURES` : une régression **Ridge** et un **LightGBM**, tous deux prédisant log10(délai). Pour LightGBM, on a
répété l'entraînement sur **5 graines**, retiré chaque groupe de variables un par un pour mesurer son importance
réelle (pas seulement son importance statistique), et calculé l'importance par **gain** (interne à l'arbre) et par
**SHAP** (contribution marginale, méthode agnostique au modèle) pour vérifier qu'elles s'accordent.

**Définitions utiles.**
- *Gain* : à chaque fois qu'une variable sert à couper un arbre, la réduction d'erreur qu'elle apporte ; sommée sur
  toute la forêt. C'est une mesure interne au modèle.
- *SHAP* : pour un flux donné, combien chaque variable a poussé la prédiction au-dessus ou en dessous de la
  moyenne, moyenné en valeur absolue sur un échantillon de flux. Indépendant du modèle, donc un bon recoupement.
- *Ablation par groupe* : réentraîner tout le modèle sans un groupe de variables (pas juste l'ignorer après coup),
  pour mesurer sa contribution réelle — une variable peut avoir un gain élevé parce qu'une autre, corrélée, est
  absente du groupe retiré ; seul le réentraînement capture ça.

## Pourquoi ces choix

| Choix | Alternative rejetée | Pourquoi (chiffre ou principe) |
|---|---|---|
| Règle log10 / log1p / brut **décidée sur train** par variable (`fit_rules`) | Standardiser toutes les variables pareil | Les variables de charge (débit, capacités, termes de file) s'étalent sur plusieurs ordres de grandeur ; les traiter en brut écraserait leur variance utile sous celle des grandes valeurs. |
| α de Ridge choisi sur une **grille complète**, publiée (`ridge_alpha_grid.csv`) | Une seule valeur par défaut | Les 15 valeurs testées (1e-3 à 1e4) sont toutes publiées, pas seulement la meilleure — exigé par §4.2 du plan. |
| LightGBM : arrêt précoce sur la **MAE_log de val** (patience 100, jusqu'à 3000 arbres) | Nombre d'arbres fixé à l'avance | Pas de nombre d'arbres choisi au hasard ; la courbe train/val (`L3_01`) montre où l'ajout d'arbres cesse d'aider. |
| **5 graines pour LightGBM**, 1 seule pour Ridge | 5 graines pour les deux | Ridge est déterministe à solveur fermé (pas de source d'aléa à moyenner) ; LightGBM a un sous-échantillonnage aléatoire des lignes et des colonnes à chaque arbre, donc varie d'une graine à l'autre. |
| Ablation = **réentraînement** sans le groupe | Mettre les colonnes à zéro après coup | Un modèle entraîné sans un groupe peut réallouer l'information à des variables corrélées restantes ; l'ablation après coup mesurerait autre chose (la sensibilité locale, pas la contribution réelle). |
| Importance **gain et SHAP**, comparées | Gain seul | Le gain peut sur-compter une variable répétée à plusieurs niveaux de l'arbre ; SHAP est agnostique au modèle. Les deux s'accordent sur le top 3 (figure `L3_02`), ce qui crédibilise le classement. |

## Ce que montrent les résultats

**MAE_log sur test** [IC 95 % bootstrap sur les scénarios, volet C] :

| Modèle | val | test | topologies inédites |
|---|---|---|---|
| Ridge | 0,499 | 0,510 [0,481 ; 0,538] | 0,453 [0,424 ; 0,481] |
| **LightGBM** | 0,348 ± 0,002 (5 graines) | **0,352 [0,326 ; 0,379]** | 0,321 [0,294 ; 0,350] |

LightGBM bat Ridge d'un facteur 1,45 : le délai dépend d'**interactions et de seuils** (l'effet de ρ sur le délai
change de régime près de la saturation, cf. L1 D1.9) qu'un modèle linéaire ne capture pas, et qu'un ensemble
d'arbres apprend nativement par des coupures successives.

**Observation à expliquer au jury : la grille d'α de Ridge est quasiment plate** (`ridge_alpha_grid.csv`, de 0,499267
à 0,499270 entre α = 1e-3 et α = 10 — une variation à la 5ᵉ décimale). Avec 934 542 lignes de train pour une
régression linéaire, le problème est déjà extrêmement bien déterminé : la régularisation ne sert à éviter le
surapprentissage que quand il y a trop peu de données par rapport au nombre de variables, ce qui n'est pas le cas
ici (27 variables). Ce n'est pas un bug ni un échec du réglage : à ce volume de données, Ridge **est** proche de sa
meilleure version possible, et c'est justement pour ça qu'il plafonne si loin derrière LightGBM — la limite n'est
pas la régularisation, elle est dans la forme linéaire du modèle.

**Importance des variables** (figure `L3_02_shap_vs_gain.png`) : `rho_link_max` domine (36 % du gain, 21 % du SHAP),
suivi des termes de file M/M/1/K (`mm1k_queue_max` 17 %, `mm1k_queue_sum` 12 %). Gain et SHAP s'accordent sur les
8 premières variables, qui sont **toutes des mesures de charge ou de file** (ρ, M/M/1, M/M/1/K, part WFQ minimale) —
aucune variable de trafic brut (débit, taille de paquet) n'apparaît dans le top 8.

**Ablation par groupe** (figure `L3_03_ablation.png`, Δ MAE_log = groupe retiré − modèle complet, seed 0 ; ligne
pointillée = écart-type entre les 5 graines de LightGBM, 0,0019, le plancher de bruit) :

| Groupe retiré | Δ MAE_log | Au-dessus du bruit ? |
|---|---|---|
| `utilization` (ρ, nb de flux concurrents) | **+0,0582** | oui, très largement |
| `wfq` (part WFQ, ρ de la file) | **+0,0342** | oui |
| `queueing` (sommes M/M/1, M/M/1/K) | +0,0093 | oui |
| `slice` (type, δ, prov_ratio) | +0,0064 | oui, de justesse |
| `capacity`, `path`, `scenario`, `traffic` | −0,0007 à −0,0051 | non (dans le bruit, parfois négatif) |

Les deux groupes qui comptent vraiment, `utilization` et `wfq`, sont exactement ceux que la charge réseau détermine
(ρ et la garantie de bande WFQ), pas le trafic lui-même. C'est cohérent avec L1 (D1.5 : le trafic réel est constant
par type de slice) et avec le volet A (D3.12 : le passage de messages du GNN sert précisément à calculer ces mêmes
grandeurs — ρ, M/M/1, WFQ — le long du graphe plutôt que de les recevoir pré-calculées).

**Lien avec le GNN.** `lgbm` (0,352) est le meilleur modèle plat du projet, devant `mlp_flat` (0,403, volet A) : un
ensemble d'arbres règle mieux ce problème tabulaire qu'un MLP. C'est donc `lgbm`, pas `mlp_flat`, qui est
l'adversaire à battre pour H1 — voir JOURNAL D3.12, où le GNN le bat d'un facteur 2,75 avec un intervalle de
confiance qui exclut 0 (différence appariée −0,2243 [−0,2484 ; −0,2023], GNN devant dans 100 % des rééchantillonnages).

## Questions probables du jury

**Pourquoi LightGBM bat-il Ridge d'un facteur 1,45 ?**
Le délai dépend d'effets de seuil (la queue explose près de ρ = 1, cf. L1) et d'interactions entre variables (le
nombre de flux concurrents compte différemment selon que le lien est déjà chargé ou non). Un modèle linéaire ne
peut représenter ni l'un ni l'autre sans qu'on les lui donne à la main ; un ensemble d'arbres les apprend par
coupures successives.

**Pourquoi l'ablation retire-t-elle tout un groupe plutôt qu'une variable à la fois ?**
Parce que les variables d'un groupe sont corrélées entre elles (ex. `rho_link_max`, `rho_link_mean`,
`rho_link_sum`) : retirer une seule variable laisserait les autres compenser et sous-estimerait l'importance du
groupe. Retirer le groupe entier force le modèle à se passer réellement de cette information.

**Pourquoi la régularisation de Ridge ne change presque rien ?**
Avec 934 542 lignes d'entraînement pour 27 variables, le problème est très bien déterminé : il n'y a pas de
surapprentissage à corriger dans la plage d'α testée. Ce n'est pas un défaut de réglage ; ça montre que la limite
de Ridge est sa forme linéaire, pas un manque de régularisation.

**Si LightGBM est si bon, pourquoi construire un GNN ?**
Parce que LightGBM reçoit des agrégats du chemin **calculés à la main** (ρ max, sommes M/M/1), alors que le GNN les
calcule lui-même en propageant l'information le long du graphe réel. Le GNN bat `lgbm` d'un facteur 2,75 avec un
intervalle de confiance qui exclut 0 (volet A, D3.12) : les agrégats à la main laissent encore de l'information sur
la table.

## Le code à connaître

### 1. `fit_rules` / `apply_rules` (`src/baselines_ml.py`) — dompter les queues lourdes

```python
elif s.min() > 0 and s.max() / s.min() > 100:
    rules[c] = "log10"                 # strictement positif, étalé sur >2 ordres de grandeur
elif s.min() >= 0 and s.max() > 100:
    rules[c] = "log1p"                 # compteurs non négatifs
else:
    rules[c] = "raw"                   # indicatrices, ratios déjà bornés
```
La règle est décidée sur les statistiques du **train seulement** (`X[is_tr]`), puis appliquée telle quelle à
val/test : aucune fuite, et `log10` est protégé par un `clip(lower=1e-12)` car val/test peuvent descendre sous le
minimum observé sur train.

### 2. `run_lgbm` (`src/baselines_ml.py`) — arrêt précoce et contrôle du surapprentissage

```python
booster = lgb.train({**LGBM_PARAMS, "seed": SEED + seed}, dtr, num_boost_round=LGBM_MAX_TREES,
                    valid_sets=[dtr, dva], valid_names=["train", "val"],
                    callbacks=[lgb.early_stopping(LGBM_PATIENCE, verbose=False),
                               lgb.record_evaluation(curve)])
```
`LGBM_MAX_TREES = 3000` est un plafond, pas une cible : l'entraînement s'arrête dès que la MAE_log de val ne
s'améliore plus pendant 100 arbres (`LGBM_PATIENCE`), et `curve` garde l'historique complet pour tracer
`L3_01_courbe_apprentissage.png` — la preuve que le nombre d'arbres retenu n'est pas arbitraire.

### 3. `_weights` (`src/bootstrap.py`) — le bootstrap apparié par scénario

```python
def _weights(S, B, seed):
    rng = np.random.default_rng(seed)
    W = rng.multinomial(S, np.full(S, 1.0 / S), size=B)   # B tirages de S scenarios avec remise
    return np.vstack([np.ones((1, S)), W])                 # ligne 0 = poids 1 : l'estimation ponctuelle
```
Chaque ligne de `W` est un rééchantillonnage des **scénarios** (jamais des flux directement, parce qu'ils sont
corrélés à l'intérieur d'un scénario — JOURNAL D2.1). La même matrice `W` sert à tous les modèles d'un groupe : les
différences entre modèles sont donc **appariées** (même tirage de scénarios des deux côtés), ce qui resserre
l'intervalle de confiance de la différence par rapport à deux intervalles calculés séparément.
