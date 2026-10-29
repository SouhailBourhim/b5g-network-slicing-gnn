# Journal des décisions expérimentales

Format : date — décision — raison (appuyée sur une donnée, un graphe ou un principe réseau) — alternative rejetée.

---

## 2026-10-28 — Étape 0 : données

**D0.1 — Source des données : copie locale vérifiée, pas de re-téléchargement.**
- Raison : `slicing-simulations.zip`, `datanetAPI.py`, `features.pdf` et le dossier déjà extrait étaient dans `~/Downloads`. MD5 du zip = `2a4a9a47bdeaa0c02304c40115c56273`, identique à la valeur Zenodo (v2.2.1). Copiés dans `data/raw/` (MD5 revérifié après copie), originaux intacts.
- Alternative rejetée : re-télécharger depuis Zenodo (inutile, même fichier bit à bit).

**D0.2 — Écrire notre propre parseur au lieu d'utiliser `DatanetAPI.__iter__` tel quel.**
- Raison : chaque échantillon stocke des matrices N×N (N = 416 nœuds dans le scénario 0, soit 173 056 cellules) alors que seules ~390 cellules contiennent un flux. L'API officielle parse toutes les cellules en Python pur, ce qui est très lent sur ~7 900 scénarios. Notre parseur ne lit que les cellules des flux réels, déclarés dans `slices/*.json`. Il est **validé contre l'API officielle** sur des échantillons (test pytest).
- Raison secondaire (sécurité) : l'API décode les fichiers `slices_*.json` avec `jsonpickle.decode`, qui peut instancier des objets Python arbitraires. Ces fichiers sont du JSON standard, on les lit avec `json`.
- Alternative rejetée : utiliser l'API officielle partout (correct mais trop lent, et `jsonpickle` est inutilement risqué).

**D0.3 — Le débit offert d'un flux est calculé depuis les paramètres du générateur, pas depuis le champ `bandwidth` nominal.**
- Constat : pour eMBB, `traffic_string = 0,18000000,1500,10,2,6000,12000,18000` donne un débit nominal de 18 Mb/s, mais les paquets suivent une loi binomiale {12000, 18000} bits (moyenne 15000), donc le débit réel est 1500 × 15000 = 22,5 Mb/s. Le débit généré mesuré (`AvgBw` = 22 482 kb/s) le confirme. Même facteur ×1,25 pour mMTC (paquets 320/480 bits), URLLC cohérent (256/1600, moyenne 928).
- Décision : `rate_bps = AvgPktsLambda × (PktSize1 + PktSize2) / 2`. Cette valeur se calcule **sans simuler** et reste donc une entrée légitime. À vérifier sur tout le jeu.
- Alternative rejetée : utiliser `bandwidth`, qui sous-estime la charge de 25 % pour eMBB et mMTC et fausserait toute utilisation ρ.

**D0.4 — `linkUsage.txt` (utilisation, pertes, occupation mesurées par port et par file) = sortie de simulation, donc exclue des entrées.**
- Raison : ces mesures proviennent de la même simulation que le délai cible. Les utiliser comme entrées reviendrait à donner au modèle une partie de la réponse (fuite). De plus l'utilisation mesurée sature à 1 (on observe 1.000011) alors que la charge offerte peut dépasser 1. Elle sert uniquement à **valider** notre ρ calculé depuis les entrées.
- Alternative rejetée : utiliser l'utilisation mesurée comme feature (performances artificiellement bonnes, inutilisable en « what-if » car inconnue avant simulation).

## 2026-10-28 — Étape 1 : parsing et exploration (sous-ensemble de dev, 4 archives / 360 scénarios)

**D1.1 — Les chemins viennent de `routing_K.txt`, pas du champ `path` de `slices_K.json`.**
- Constat : seuls ~57 % des flux ont un chemin simulé identique au chemin du fichier slices (sous-ensemble de dev). Avec les chemins de `routing_K.txt`, ρ calculé depuis les entrées reproduit l'utilisation mesurée par le simulateur (corrélation 0,97–0,999 par scénario, MAE < 0,003) ; avec les chemins du fichier slices, la corrélation tombe à 0,66 dans certains scénarios. L'API officielle utilise aussi `routing_K.txt`.
- Alternative rejetée : utiliser le `path` du fichier slices (décrit par `features.pdf`), qui ne correspond pas au trafic réellement simulé.

**D1.2 — `delta` = facteur de provisionnement ; on dérive `prov_ratio` = réservation / débit réel.**
- Constat : sur chaque port, le poids WFQ de la file d'une slice = Σ réservations de ses flux / capacité × 100, et réservation = débit nominal × (0,85 + 0,3·δ) (δ=0,09 → 0,877 ; δ=0,481 → 0,994 ; δ=0,9 → 1,12). Chaque file dédiée contient exactement une slice.
- Conséquence : par rapport au débit **réel**, eMBB et mMTC sont toujours sous-provisionnés (prov_ratio ∈ [0,68 ; 0,92]), URLLC ∈ [0,85 ; 1,15].

**D1.3 — Part WFQ par saut = poids de la file du flux / Σ poids du port ; plancher à 1e-5.**
- Constat : quand le routage simulé s'écarte du chemin prévu, le flux traverse des ports où sa slice n'a pas de réservation : sa file a un poids 0,001 (part ≈ 1e-5). Ces flux sont « affamés » : délai > 1 s, pertes ≈ 99,9 % (les 49 flux > 1 s des 10 premiers scénarios ont tous une part min ≈ 1e-5 et aucun ne suit son chemin prévu). La file par défaut reçoit 100 − Σ réservations, parfois négatif (14 sauts / 354 k) → plancher 1e-5, valeur brute conservée.
- Alternative rejetée : ignorer WFQ et ne garder que ρ du lien (vue FIFO) : n'explique pas les délais de plusieurs secondes sur des liens à ρ ≈ 1.

**D1.4 — `delay = -1` = sentinelle « aucun paquet reçu » (100 % de pertes) : flux exclus de la régression du délai, gardés pour la perte.**

**D1.5 — Plusieurs « entrées » sont constantes et donc sans information :** loi d'inter-arrivée (Poisson), loi de taille (binomiale), file de 32 paquets, `maxAvgLambda` ; le débit réel est constant par type de slice (eMBB 22,5 Mb/s, URLLC 3,5 Mb/s, mMTC 250 kb/s). La variabilité du délai vient donc de la **topologie, du routage, de la compétition entre flux et de l'allocation WFQ**, pas du profil de trafic individuel. C'est un argument en faveur d'un modèle relationnel (GNN).

**D1.6 — Résolution des sorties : le simulateur écrit délai et gigue avec 6 décimales.** Les gigues < 1e-6 s² et les délais ~1e-5 s sont quantifiés (pic à 1e-6 dans l'histogramme de la gigue). La gigue reste une cible secondaire.

## 2026-10-28 — Étape 1 : confirmation sur le jeu complet (79 archives, parsing 9,8 min)

**D1.7 — Les constats du sous-ensemble de dev tiennent sur tout le jeu** (`results/eda_summary.json`) : 7 127 scénarios valides (773 écartés, statut ≠ OK), 162 topologies (41 scénarios/topologie en médiane), 1 576 669 flux, 7 446 192 sauts. Débit mesuré / débit calculé = 1,00005 (médiane). Chemins identiques au fichier slices pour 58 % des flux seulement ; corrélation min(ρ,1) vs utilisation mesurée = 0,984 avec `routing_K.txt`. Délai : 8,3 ordres de grandeur, asymétrie 38 brute / 0,99 en log10. 9 flux sans paquet reçu.

**D1.8 — Flux affamés = part WFQ quasi nulle *et* lien saturé, mais la règle n'est pas suffisante.**
- Constat : 45 744 des 46 194 flux > 1 s ont, sur au moins un saut, une part WFQ < 1e-3 sur un lien à ρ ≥ 1. Une part quasi nulle sur un lien non saturé est sans effet (15 flux > 1 s sur 104 392), car WFQ redistribue la capacité inutilisée. Mais 77 158 flux qui remplissent la condition restent sous 10 ms.
- Conséquence : une règle à seuil sur les entrées détecte presque tous les flux affamés mais avec beaucoup de faux positifs. Hypothèse à tester : un modèle qui voit la compétition réelle dans chaque file (GNN) les sépare mieux. Utile pour la définition des événements (L4).

**D1.9 — Coude du délai à ρ ≈ 1 (figure L1_05).** La médiane du délai est multipliée par ~10 entre ρ_max = 0,95 et 1,05 (eMBB, mMTC), puis plafonne : la file de 32 paquets borne l'attente, l'excès devient de la perte (M/M/1/K). Point de départ du choix de τ en L4.

## 2026-10-28 — Étape 2 (prérequis communs, avant le travail en parallèle)

**D2.1 — Unité de split = le scénario ; test « topologies inédites » découpé en premier.**
- 15 % des topologies tirées *dans chaque quartile de taille* du cœur (S ≤ 13, M ≤ 21, L ≤ 33, XL > 33 nœuds) → 24 topologies, 1 080 scénarios (`test_unseen_topo`). Puis les scénarios restants en 70/15/15 groupés par scénario, stratifiés par tercile de ρ max du scénario → 4 233 / 907 / 907. Seed 42, `results/splits.csv` versionné, tests `tests/test_splits.py`.
- Raison : les flux d'un scénario partagent des liens (fuite si on les sépare) ; découper les topologies en premier garantit qu'aucune n'apparaît des deux côtés. La stratification par taille met petits et grands réseaux dans le test inédit ; celle par charge garde la même proportion de scénarios congestionnés partout.
- Constat à garder pour H3 : les topologies retenues sont moins chargées (ρ max médian 1,35 contre 1,63 en train). L'écart d'erreur sur `test_unseen_topo` mélangera donc « topologie nouvelle » et « charge différente » ; à contrôler en L6 (comparer à charge égale).
- Alternative rejetée : split aléatoire par flux (fuite, preuve expérimentale à faire en L2 par l'équipe).

**D2.2 — Métriques communes : MAE / MSE / RMSE sur log10(délai) + MdAPE en secondes, toujours par type de slice** (`src/metrics.py`, tests à la main). Baseline naïve (médiane train par slice) : MAE_log = 0,82 sur test.

**D2.3 — Contrats de fichiers et données partagées.** Prédictions `results/preds/<modèle>__seed<k>.parquet`, événements `data/processed/events.parquet`, évaluation unique `src/evaluate.py` → `results/metrics_all.csv`. Tables traitées publiées dans la release GitHub `data-v1` avec SHA-256 (`src/data_sync.py`).
- Alternatives rejetées : Git LFS (quota gratuit 1 Go), dossier partagé (ni version ni vérification d'intégrité), chacun re-parse (10 min et risque de versions différentes).

**D2.4 — Entraînement sur GPU** (RTX 5070 Laptop, torch 2.14.1+cu130). L'index cu128 s'arrête à torch 2.11 ; cu130 est supporté par le pilote (CUDA 13.1).

## 2026-10-28 — Étape 2 : formulation et expérience de fuite (L2)

**D2.5 — La fuite est mesurée, pas seulement argumentée** (`src/leakage.py`, 1 500 scénarios hors topologies inédites, 20 % de scénarios « jamais vus » E, LightGBM aux paramètres fixes non réglés, 3 graines).
- Split aléatoire par flux (A) : erreur annoncée MAE_log = 0,127 ± 0,005, erreur réelle sur E = 0,399 ± 0,013, soit un facteur d'optimisme de **3,15**. 100 % des flux de test ont un frère dans le train.
- Split par scénario (B) : annoncée 0,358 ± 0,024, réelle 0,410 ± 0,017 (facteur 1,14, écart de l'ordre de 2 écarts-types entre graines : variance entre scénarios, pas de fuite ; 0 % de flux de test avec un frère).
- Conséquence : toute évaluation du projet se fait avec les splits par scénario. Alternative rejetée : split aléatoire par flux (rapporterait 0,13 là où l'erreur réelle est 0,40).

**D2.6 — Le dépôt est public (décision de Souhail) : audit avant exposition.** Aucun jeton, mot de passe ou clé dans les fichiers suivis (recherche par motifs). Un chemin local contenant le nom d'utilisateur Windows apparaissait dans la sortie d'un notebook : corrigé (affichage d'un chemin relatif). L'adresse e-mail de commit reste visible dans l'historique (existait déjà). `fetch` des données passe par HTTPS sans compte ; seul `publish` exige `gh`.

**D2.7 — Répartition de l'équipe en trois volets** (`docs/plan_de_travail_equipe.docx`) : A = GNN/GPU (Souhail), B = réseau/événements, C = données/ML classique. L2, que nous avions déjà commencé, est terminé ici plutôt que laissé à C ; C le reprend pour le défendre.

## 2026-10-28 — Volet A (Souhail) : graphes et GNN (L5)

**D3.1 — Graphe hétérogène flux/lien, un par scénario** (`src/graphs.py`). Nœuds flux (type de slice, δ, prov_ratio, débit, taille de paquet, longueur du chemin) ; nœuds lien pour chaque lien utilisé (capacité, ρ, nb de flux, nb de files, M/M/1/K du lien, type accès/cœur/sortie) ; une arête par saut flux→lien et sa réciproque, avec en attribut ce qui appartient à la **paire** : position relative sur le chemin, part WFQ (log), ρ de la file (log), file par défaut, nb de flux dans la file, M/M/1/K de la file. Standardisation ajustée sur **train** seulement (`graphs/norm.json`). 7 127 graphes, 1,58 M nœuds flux, 1,68 M nœuds lien, 539 Mo, construits en 10 s ; tests `tests/test_graphs.py` (comptes = tables, arêtes valides et réciproques, pas de NaN).
- Les 1,58 M liens d'accès (source → antenne, un seul flux chacun) sont gardés comme nœuds : ils font partie du chemin (ρ = 0,45 pour eMBB) ; la compétition se joue sur les ~100 k liens de cœur et de sortie.
- Alternative rejetée : mettre la part WFQ sur le nœud lien (elle dépend du flux) ou sur le nœud flux (elle dépend du lien).

**D3.2 — Incident corrigé : cache de 330 Go.** `torch.from_numpy(tableau[i:j])` crée une *vue* sur tout le tableau ; `torch.save` écrivait donc le tableau complet pour chaque graphe. Corrigé en copiant chaque tranche (`torch.tensor`), fichier supprimé, taille réelle 539 Mo. Leçon : mesurer temps et taille sur 100 scénarios avant tout calcul complet.

**D3.3 — Modèle** (`src/models/gnn.py`) : encodeurs MLP, K itérations alternant liens←flux puis flux←liens (résiduel + LayerNorm + dropout), lecture MLP sur les flux. Deux convolutions : GATv2Conv avec `edge_dim` (attention inspectable) et EdgeSAGE (GraphSAGE moyenne dont le message voit les attributs d'arête, car SAGEConv les ignore). `gnn_k0` = K=0 = MLP sur les seules features du flux. Entraînement `src/train_gnn.py` sur GPU : MSE sur log10(délai), AdamW, ReduceLROnPlateau, arrêt précoce sur la MAE_log de val, test prédit une seule fois avec le meilleur checkpoint. Tests rapides (50 scénarios, 2 époques) OK sur les trois variantes.

## 2026-10-29 — Volet B (réseau et événements)

**D3.1 — Baselines analytiques : M/M/1 simple bat M/M/1/K et WFQ ; calibration sans gain net (T.).**
- Constat (MAE_log test) : analytic_mm1 0,649 ; analytic_mm1k_link 0,839 ; analytic_mm1k_wfq 1,141 ; analytic_mm1k_link_cal 0,672 (a = 1,740, b = 1,318, ajustés sur train) ; naïve 0,822.
- Erreur signée (val) : M/M/1/K sous-estime fortement quand ρ_max > 1,05 (−0,67 à −0,80 en log10) ; M/M/1 surestime près de ρ ≈ 1 (+0,73) puis sous-estime (−0,14 à −0,31).
- Alternative rejetée : choisir la meilleure des trois sans le dire ; les quatre sont publiées.

**D3.2 — Seuil de congestion τ = 1,05, choisi sur train (T.).**
- Raison : le plus grand saut de la médiane de log10(délai) entre deux classes de ρ_max est +0,742 entre (1,0 ; 1,05] et (1,05 ; 1,1] (results/L4_tau_choice.csv). Le coude est à 1,05, pas à 1,0.
- Alternative rejetée : τ = 1,0 (seuil théorique) ; le saut y est de +0,134 seulement.
- Sensibilité : colonnes is_event_tau_{0_9, 1_0, 1_05, 1_1, 2_0}.

**D3.3 — Les « événements » sont la majorité des flux : 82,9 % sur test (T.).**
- Constat : 85,4 % des flux de train ont ρ_max ≥ 1 (médiane 1,245). Congestion à τ = 1,05 : 82,2 % des flux de train ; famine (part WFQ < 1e-3 sur un lien à ρ ≥ τ) : 10,7 % ; aucun cas « famine seule » car la famine implique ρ ≥ τ.
- Conséquence : « événement vs non-événement » compare presque tout le jeu à une petite minorité peu chargée. Le palier « sévère » is_event_tau_2_0 (16,1 % des flux de test) est la comparaison utile ; à discuter avec l'équipe avant L6.
- Erreur MAE_log test, non-événement / événement / sévère : naïve 1,322 / 0,719 / 1,098 ; analytic_mm1 0,363 / 0,708 / 0,983 ; analytic_mm1k_wfq 2,448 / 0,871 / 0,612.
