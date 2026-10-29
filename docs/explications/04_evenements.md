# 04 — Baselines analytiques et événements (volet B)

> Fichiers : `src/baselines_analytic.py`, `src/events.py`, `src/events_viz.py`, `data/processed/events.parquet`,
> `results/L4_tau_choice.csv`, `results/L4_coverage.csv`, `reports/figures/L4_01_scenarios.png`.
> Décisions : `JOURNAL.md`, entrées D3.1 à D3.3. Tous les chiffres ci-dessous viennent de ces fichiers.

## 1. Ce qu'on a fait (en bref)

1. On a mis en place quatre **baselines sans apprentissage** fondées sur la théorie des files d'attente (M/M/1, M/M/1/K vue lien, M/M/1/K vue WFQ, plus une version calibrée). Elles donnent la **référence physique** : que vaut un modèle qui ne connaît que les formules du réseau ?
2. On a défini deux **événements** à partir des seules entrées du réseau : la **congestion** (un lien du chemin est surchargé) et la **famine** (le flux n'a presque aucune part WFQ garantie sur un lien saturé).
3. On a choisi le seuil de congestion **τ = 1,05 sur le jeu d'entraînement uniquement**, en cherchant le coude de la courbe « délai médian en fonction de ρ ».
4. On a publié `events.parquet` (un indicateur par flux) pour que les volets A et C évaluent leurs modèles « sur les événements » avec la même définition.
5. On a vérifié visuellement la définition sur quatre petits scénarios (figure `L4_01_scenarios.png`).

## 2. Pourquoi ces choix (et ce qu'on a rejeté)

### Pourquoi des baselines analytiques ?
Le guide demande de comparer le GNN à une référence. Une référence qui apprend (Ridge, LightGBM) mesure « ce que les données permettent » ; une référence analytique mesure « ce que la théorie permet sans données ». Si un GNN ne bat pas une formule M/M/1, son intérêt est faible. C'est aussi l'argument télécom : un opérateur fait confiance à un modèle de files d'attente, le GNN doit faire mieux.

Rappel des formules, par saut, avec ρ = λ/μ et S le temps de service d'un paquet :

- **M/M/1** : T = S / (1 − ρ). Diverge quand ρ → 1, d'où l'écrêtage ρ ≤ 0,99 (`RHO_CLIP`).
- **M/M/1/K** (file à K places, ici K = taille de file + 1) : le tampon borne l'attente, l'excès de trafic devient de la **perte**, pas du délai. Valable pour tout ρ, y compris ρ > 1.
- Le délai du flux est la **somme des délais par saut** le long de son chemin.

Les trois estimations sont déjà dans `flow_features` (`mm1_sum`, `mm1k_link_sum`, `mm1k_queue_sum`) et ne sont calculées que depuis les entrées (débits, capacités, routage, poids WFQ).

### Pourquoi définir les événements par les ENTRÉES et pas par le délai ?
Si on disait « événement = délai > 1 s », on évaluerait les modèles sur les cas où ils ont le plus de mal par construction, et on ne pourrait pas utiliser la définition pour **prévoir** : l'orchestrateur ne connaît pas le délai avant de simuler. Avec ρ calculé depuis les entrées, la définition est utilisable en production et ne fuit aucune sortie de simulation. **Alternative rejetée** : seuil sur le délai ou la perte.

### Pourquoi « un état » et pas « un instant » ?
Le jeu de données n'a **pas d'axe temporel** : chaque scénario est un instantané stationnaire. Un événement ne peut donc pas être un changement dans le temps. C'est un **état du réseau** (surcharge, famine). Il faut l'écrire clairement dans le rapport, parce que le guide parle de « changement brusque ».

### Pourquoi τ = 1,05 ?
Sur **train**, on a regardé la médiane de log10(délai) par classe de ρ_max (`results/L4_tau_choice.csv`). Le plus grand saut entre deux classes voisines est **+0,742** (soit un facteur 5,5), entre (1,0 ; 1,05] et (1,05 ; 1,1]. Juste avant, entre (0,95 ; 1,0] et (1,0 ; 1,05], le saut n'est que de **+0,134**. Le coude est donc à 1,05, pas au seuil théorique 1,0. **Alternative rejetée** : τ = 1,0, parce que la théorie dit « saturation » ; les données disent que le délai ne saute qu'un peu plus tard.

### Pourquoi cette règle de famine ?
Journal D1.8 : une part WFQ quasi nulle (< 1e-3) n'a d'effet que si le lien est saturé, car WFQ redistribue la capacité inutilisée. D'où la règle « part < 1e-3 **et** ρ_lien ≥ τ ». Elle est volontairement simple et sert de repère.

## 3. Ce que montrent les résultats

### 3.1 Baselines analytiques (MAE sur log10(délai), jeu de test)

| Modèle | MAE_log test | val | test_unseen_topo |
|---|---|---|---|
| naive_median_slice (plancher) | 0,822 | 0,815 | 0,790 |
| analytic_mm1 | **0,649** | 0,658 | 0,646 |
| analytic_mm1k_link | 0,839 | 0,836 | 0,815 |
| analytic_mm1k_link_cal (a = 1,740 ; b = 1,318, ajustés sur train) | 0,672 | 0,682 | 0,667 |
| analytic_mm1k_wfq | 1,141 | 1,095 | 0,925 |

- **M/M/1 simple est la meilleure formule** et la seule nettement meilleure que le plancher naïf.
- **M/M/1/K ne bat pas le plancher** (0,839 contre 0,822) alors qu'il est théoriquement plus réaliste près de la saturation. Erreur signée sur val : il **sous-estime** fortement dès que ρ_max > 1,05 (−0,67 puis −0,80 en log10).
- **La vue WFQ est la pire en moyenne**, mais c'est la meilleure sur les cas sévères (voir 3.3).
- Ce n'est pas expliqué complètement. Pistes **non vérifiées** : le plafond de K = 33 paquets borne le délai simulé moins bien que prévu quand plusieurs sauts sont saturés, et l'hypothèse de Poisson sous-estime la rafale. À tester avant de l'affirmer à l'oral.

### 3.2 Les événements couvrent presque tous les flux

Source : `results/L4_coverage.csv`, règle τ = 1,05.

| Split | % de flux en événement | dont famine | % de scénarios avec au moins un événement |
|---|---|---|---|
| train | 82,2 | 10,7 | 98,2 |
| val | 84,6 | 9,7 | 98,8 |
| test | 82,9 | 9,3 | 98,1 |
| test_unseen_topo | 85,2 | 9,2 | 98,5 |

La cause est dans les données : **85,4 % des flux de train ont ρ_max ≥ 1** (médiane 1,245 ; 90ᵉ centile 2,21 ; 95ᵉ centile 2,48). Le générateur du jeu simule beaucoup de situations sur-provisionnées. Conséquence : « événement » n'est pas une situation rare ici, c'est le régime dominant.

La famine est toujours **incluse dans** la congestion : elle exige ρ_lien ≥ τ, donc ρ_max ≥ τ. Il n'existe pas de cas « famine seule » ; les types observés sont `none`, `congestion` et `both`.

### 3.3 Erreur selon l'événement (MAE_log, test)

| Modèle | tous | non-événement | événement | sévère (τ = 2,0) |
|---|---|---|---|---|
| naive_median_slice | 0,822 | 1,322 | 0,719 | 1,098 |
| analytic_mm1 | 0,649 | 0,363 | 0,708 | 0,983 |
| analytic_mm1k_link | 0,839 | 0,307 | 0,950 | 1,330 |
| analytic_mm1k_link_cal | 0,672 | 0,553 | 0,697 | 0,920 |
| analytic_mm1k_wfq | 1,141 | 2,448 | 0,871 | **0,612** |

- Comparer « événement » à « non-événement » met 83 % des flux face à 17 % de flux peu chargés : peu informatif.
- Le palier **sévère** (`is_event_tau_2_0`, 16,1 % des flux de test) est la comparaison utile. **Le classement s'y inverse** : la vue WFQ (0,612) bat M/M/1 (0,983) et le plancher naïf (1,098).
- Lecture : quand le trafic est fort, ce qui décide du délai est la **part garantie** de la file, pas la charge globale du lien. C'est précisément l'information qu'un modèle à graphe (qui voit la compétition dans chaque file) pourrait mieux exploiter. C'est une **hypothèse** à tester avec les volets A et C, pas un résultat.

### 3.4 Vérification visuelle

La figure `reports/figures/L4_01_scenarios.png` montre quatre scénarios de train :

- scénario 4618 : ρ max 0,45, aucun événement ;
- scénario 5792 : congestion simple, 7 flux sur 7 en événement, aucune famine ;
- scénario 290 : famine, un saut affamé sur un lien à ρ élevé (croix noire) ;
- scénario 2294 : **2 flux sur 15 seulement** en événement. L'événement est par flux, pas par scénario.

Les positions des nœuds viennent d'un placement automatique : les distances n'ont aucun sens.

## 4. Questions probables du jury

**Q1. Pourquoi τ = 1,05 et pas 1 ?**
Parce que sur train la médiane du log10 du délai n'augmente que de 0,134 en passant de la classe (0,95 ; 1,0] à (1,0 ; 1,05], puis de 0,742 entre (1,0 ; 1,05] et (1,05 ; 1,1]. Le coude est là. Le seuil 1,0 est dans les colonnes de sensibilité (`is_event_tau_1_0`) : on peut montrer que les conclusions ne changent pas, ou le dire si elles changent. On n'a pas utilisé val ni test pour choisir.

**Q2. Pourquoi ne pas définir l'événement par le délai observé ?**
Parce que le délai est la cible. Évaluer les modèles sur « les flux au délai extrême » sélectionne sur la sortie et rend le résultat circulaire. Et en production on ne connaît pas le délai avant la simulation, alors que ρ se calcule depuis la configuration.

**Q3. 83 % des flux sont des événements : ça reste un événement ?**
Non, pas au sens d'un cas rare, et on le dit. C'est une propriété du jeu (85,4 % des flux de train ont ρ_max ≥ 1). On propose donc deux lectures : le palier sévère τ = 2,0 (16,1 % des flux de test) et la comparaison « charge faible contre forte » (bonus B4). Le seuil n'est pas truqué pour obtenir 10 % : le coude donne 1,05, et on rapporte la couverture telle qu'elle est.

**Q4. Pourquoi M/M/1/K, plus réaliste, est-il moins bon que M/M/1 ?**
On ne l'a pas encore démontré. On observe qu'il sous-estime le délai quand ρ_max > 1,05 (−0,67 à −0,80 en log10, val). Les pistes sont le plafond K = 33 et l'hypothèse de Poisson, mais je ne les présente pas comme établies. Ce qui est établi : sur ce jeu, la formule la plus simple est la meilleure formule à tester contre le GNN.

**Q5. Le jeu n'a pas de temps, que veut dire « événement » ?**
Un état du réseau : un lien surchargé ou une file affamée dans une configuration donnée. C'est cohérent avec l'usage visé, car l'orchestrateur compare des configurations candidates, il n'observe pas une série temporelle.

**Q6. Pourquoi la famine est-elle toujours incluse dans la congestion ?**
Parce que la règle de famine exige un lien à ρ ≥ τ ; or ρ_max sur le chemin est au moins égal à ce ρ. Aucun flux n'est donc « famine sans congestion ». C'est une conséquence de la définition, pas un résultat sur le réseau.

**Q7. Y a-t-il une fuite dans `events.parquet` ?**
Non : seules `rho_link_max` (flux) et `wfq_share`, `rho_link` (sauts) interviennent, toutes calculées depuis les entrées. Le délai ne sert qu'à **choisir τ sur train** et à **vérifier** la règle, jamais à la calculer pour un flux de val ou de test.

## 5. Fonctions expliquées ligne par ligne

### 5.1 `mm1_sojourn` (dans `src/features.py`)

```python
def mm1_sojourn(rho, service_time, rho_clip=RHO_CLIP):
    rho = np.clip(rho, 0.0, rho_clip)
    return service_time / (1.0 - rho)
```

- `np.clip(rho, 0.0, rho_clip)` : on ramène ρ dans [0 ; 0,99]. Au-dessus de 1, la formule n'a plus de sens (file instable) et en 1 elle donne une division par zéro.
- `service_time / (1.0 - rho)` : c'est T = S / (1 − ρ). À ρ = 0,5 le délai vaut 2 S ; à 0,99 il vaut 100 S. La valeur 0,99 fixe donc le plafond du modèle.
- Conséquence à savoir défendre : tous les liens saturés reçoivent la **même** valeur écrêtée, donc M/M/1 ne distingue pas ρ = 1,2 de ρ = 3.

### 5.2 `starvation_by_tau` (dans `src/events.py`)

```python
def starvation_by_tau(hops, taus):
    low = hops[hops.wfq_share < STARVE_SHARE]
    out = {}
    for t in taus:
        hit = low[low.rho_link >= t].groupby(["scenario_id", "flow_id"]).size()
        out[t] = hit
    return pd.DataFrame(out).notna()
```

- `low = hops[hops.wfq_share < STARVE_SHARE]` : on garde les **sauts** dont la part WFQ garantie est inférieure à 1e-3. On le fait une fois, avant la boucle, pour ne pas filtrer 7,4 millions de lignes à chaque seuil.
- Pour chaque `t`, `low[low.rho_link >= t]` garde ceux dont le **lien** est saturé au sens du seuil ; `groupby(...).size()` compte, par flux, combien de tels sauts il traverse.
- `out[t] = hit` : une série indexée par (scénario, flux), seulement pour les flux qui ont au moins un tel saut.
- `pd.DataFrame(out).notna()` : le DataFrame réunit les index ; une case vide (NaN) signifie « ce flux n'a aucun saut affamé à ce seuil ». `notna()` convertit en booléen : `True` = au moins un saut affamé. Les flux qui n'apparaissent dans aucune série sont absents ; ils sont remplis avec `False` plus tard.

### 5.3 `build_events` (dans `src/events.py`)

```python
f = pd.read_parquet(".../flow_features.parquet", columns=["scenario_id", "flow_id", "rho_link_max"])
h = pd.read_parquet(".../hop_features.parquet", columns=["scenario_id", "flow_id", "wfq_share", "rho_link"])
st = starvation_by_tau(h, sorted(set(TAU_SENSITIVITY + [TAU])))
f = f.merge(st.reset_index(), on=["scenario_id", "flow_id"], how="left")
```

- On ne charge que les colonnes nécessaires (gain de mémoire) : `rho_link_max` côté flux, `wfq_share` et `rho_link` côté sauts. Aucune colonne de sortie de simulation n'est lue.
- La fusion `how="left"` conserve tous les flux ; ceux sans saut affamé reçoivent NaN, que la boucle suivante transforme en `False`.

```python
cong = f.rho_link_max >= TAU
starv = f[TAU]
f["event_type"] = np.select([cong & starv, cong, starv], ["both", "congestion", "starvation"], "none")
f["is_event"] = f.event_type != "none"
```

- `cong` : la congestion, au sens `ρ_max ≥ 1,05`.
- `starv` : la colonne booléenne de famine calculée pour τ = 1,05.
- `np.select` évalue les conditions **dans l'ordre** : d'abord « les deux », puis congestion seule, puis famine seule ; sinon `"none"`. L'ordre compte : mettre `cong` avant `cong & starv` ferait disparaître le type `both`.
- `is_event` est vrai dès que le type n'est pas `none`.

```python
for t in TAU_SENSITIVITY:
    f[tau_col(t)] = (f.rho_link_max >= t) | f[t]
```

- Pour chaque seuil de sensibilité, on **recalcule toute la règle** (congestion OU famine à ce seuil), pas seulement la congestion. Les colonnes `is_event_tau_0_9`, `_1_0`, `_1_05`, `_1_1`, `_2_0` permettent de refaire n'importe quelle analyse avec un autre τ sans relancer le pipeline.

## 6. Limites à écrire dans le rapport

- **Données simulées** : un simulateur de paquets, pas des mesures réelles.
- **Réseau de transport sans RAN** : pas de radio, pas de mobilité.
- **Instantanés statiques** : pas de dynamique temporelle ; un « événement » est un état.
- **Hypothèses M/M/1** : arrivées de Poisson et services exponentiels, alors que les flux du jeu sont rafaleux.
- **Seuil τ choisi sur un critère** (le coude de la médiane) : un autre critère donnerait un autre seuil. D'où les colonnes de sensibilité.
- **Événements majoritaires** : l'évaluation « événement contre non-événement » est peu contrastée ; le palier sévère et le découpage par charge (B4) sont plus parlants.
