# Journal des décisions expérimentales

Format : date — décision — raison (appuyée sur une donnée, un graphe ou un principe réseau) — alternative rejetée.

Auteur : sauf mention contraire, les décisions ci-dessous sont de **Souhail Bourhim (SB)**. Chaque coéquipier
signe les siennes avec ses initiales (volet B, volet C).

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

*(Renumérotées D4.x : D3.1-D3.3 étaient déjà prises par le volet A dans `a4656b5`. Contenu inchangé.)*

**D4.1 — Baselines analytiques : M/M/1 simple bat M/M/1/K et WFQ ; calibration sans gain net (T.).**
- Constat (MAE_log test) : analytic_mm1 0,649 ; analytic_mm1k_link 0,839 ; analytic_mm1k_wfq 1,141 ; analytic_mm1k_link_cal 0,672 (a = 1,740, b = 1,318, ajustés sur train) ; naïve 0,822.
- Erreur signée (val) : M/M/1/K sous-estime fortement quand ρ_max > 1,05 (−0,67 à −0,80 en log10) ; M/M/1 surestime près de ρ ≈ 1 (+0,73) puis sous-estime (−0,14 à −0,31).
- Alternative rejetée : choisir la meilleure des trois sans le dire ; les quatre sont publiées.

**D4.2 — Seuil de congestion τ = 1,05, choisi sur train (T.).**
- Raison : le plus grand saut de la médiane de log10(délai) entre deux classes de ρ_max est +0,742 entre (1,0 ; 1,05] et (1,05 ; 1,1] (results/L4_tau_choice.csv). Le coude est à 1,05, pas à 1,0.
- Alternative rejetée : τ = 1,0 (seuil théorique) ; le saut y est de +0,134 seulement.
- Sensibilité : colonnes is_event_tau_{0_9, 1_0, 1_05, 1_1, 2_0}.

**D4.3 — Les « événements » sont la majorité des flux : 82,9 % sur test (T.).**
- Constat : 85,4 % des flux de train ont ρ_max ≥ 1 (médiane 1,245). Congestion à τ = 1,05 : 82,2 % des flux de train ; famine (part WFQ < 1e-3 sur un lien à ρ ≥ τ) : 10,7 % ; aucun cas « famine seule » car la famine implique ρ ≥ τ.
- Conséquence : « événement vs non-événement » compare presque tout le jeu à une petite minorité peu chargée. Le palier « sévère » is_event_tau_2_0 (16,1 % des flux de test) est la comparaison utile ; à discuter avec l'équipe avant L6.
- Erreur MAE_log test, non-événement / événement / sévère : naïve 1,322 / 0,719 / 1,098 ; analytic_mm1 0,363 / 0,708 / 0,983 ; analytic_mm1k_wfq 2,448 / 0,871 / 0,612.

**D4.4 — Découpage « décalage de charge » (B4) : entraîner sur le tercile bas de ρ_max-scénario, tester sur milieu et haut (T.).**
- Règle : coupures Q1 = 1,283 et Q2 = 1,983 = terciles de `scen_rho_max` sur les scénarios de **train d'origine** ; `test_unseen_topo` exclu. Groupes : shift_train (1 412 scénarios, 265 055 flux), shift_val (303), shift_test_id (302), shift_test_mid (2 014, 498 153 flux), shift_test_high (2 016, 465 591 flux). Fichiers : `results/shift_splits.csv`, `results/shift_summary.json`, code `src/shift.py`.
- Les scénarios mid/high pris dans train, val ou test d'origine sont des tests légitimes : un modèle « shift » ne s'entraîne jamais dessus. Colonne `topology_in_shift_train` pour isoler l'effet charge (95,4 % des mid et 85,8 % des high sont sur des topologies vues).
- Constat : le tercile « bas » n'est pas peu chargé. ρ_max va de 0,45 à 1,28 et 85,4 % de ses flux sont déjà en événement (τ = 1,05). Le décalage est donc « surcharge modérée → forte », pas « réseau libre → saturé ». À écrire ainsi dans le rapport.
- Baselines (MAE_log, entraînées sur shift_train) id / mid / high : naïve 0,567 / 0,844 / 1,087 ; analytic_mm1 0,603 / 0,588 / 0,745 ; analytic_mm1k_link_cal 0,622 / 0,652 / 0,793. Mêmes tendances en ne gardant que les topologies vues (naïve 1,061, mm1 0,756 sur high).
- La calibration change avec le régime : (a, b) = (0,520 ; 0,981) sur shift_train contre (1,740 ; 1,318) sur le train complet.
## 2026-10-29 — Volet A : premiers résultats et ablations (L5)

**D3.4 — Premier entraînement complet de `gnn_gatv2`** (graine 0, 868 481 paramètres, hidden 128, K = 3, 4 têtes, dropout 0,1, lr 1e-3, weight decay 1e-5, lots de 32 graphes). 76 époques en 141 min sur la RTX 5070, meilleure époque 61, arrêt précoce ensuite. **MAE_log : 0,128 val / 0,128 test / 0,115 topologies inédites** (`results/metrics_all.csv`). Surapprentissage contrôlé et mesuré (figure `L5_02_courbes_apprentissage.png`) : train et val descendent ensemble jusqu'à l'époque ~36, puis val plafonne à 0,075 de MSE pendant que train continue vers 0,05 — l'écart reste faible et l'arrêt précoce tranche à 61.

**D3.5 — Contrôle « mêmes entrées, sans graphe » : `mlp_flat`** (`src/train_mlp.py`, 4 225 paramètres sur les 27 `ALL_FEATURES`, scaler ajusté sur train, même boucle d'entraînement). 76 époques en 11 min **sur CPU** (le GPU était occupé par le GNN). MAE_log : 0,407 val / 0,401 test / 0,371 inédites. Le GNN fait donc **3,1× mieux à information comparable**, et 6,4× mieux que la baseline naïve (0,82 test).

**D3.6 — L'ablation K = 0 ne mesure pas le passage de messages seul : il faut la lire avec `mlp_flat`.** `gnn_k0` (K = 0, 52 097 paramètres, 25 s, meilleure époque 7) donne 0,785 val / 0,792 test / 0,735 inédites, à peine mieux que la baseline naïve (0,815 val). Explication : avec K = 0 il n'y a aucune convolution, donc les nœuds lien et les attributs de saut sont **déconnectés de la lecture** — le modèle ne voit que les 8 features du nœud flux (type de slice, δ, prov_ratio, débit, taille de paquet, longueur du chemin), c'est-à-dire **ni ρ, ni capacité, ni part WFQ, ni terme de file**. `gnn_k0` mesure donc l'apport du **graphe entier** (structure + information portée par les liens et les sauts), pas celui du passage de messages.
- Lecture correcte, en encadrant avec les deux ablations : 0,792 → 0,401 (`gnn_k0` → `mlp_flat`) = ce que récupèrent les **agrégats calculés à la main** le long du chemin (ρ max/moyen/somme, sommes M/M/1 et M/M/1/K, part WFQ minimale) ; 0,401 → 0,128 (`mlp_flat` → `gnn_gatv2`) = ce que récupère en plus le **passage de messages appris**, à information équivalente. C'est ce second chiffre qui répond à H1.
- Confirme aussi D1.5 : `gnn_k0` ≈ baseline naïve, donc les propriétés propres d'un flux ne prédisent presque pas son délai ; c'est la charge partagée qui décide.

**D3.7 — Tous les modèles font *mieux* sur `test_unseen_topo` que sur `test` : c'est une propriété du jeu, pas une meilleure généralisation.** naïf 0,823 → 0,790 ; `gnn_k0` 0,792 → 0,735 ; `mlp_flat` 0,401 → 0,371 ; `gnn_gatv2` 0,128 → 0,115. La baseline naïve prédit une constante par type de slice et ne peut donc pas « mieux généraliser » à des topologies inédites : l'écart vient de ce que les topologies retenues sont moins chargées (D2.1 : ρ max médian 1,35 contre 1,63). **H3 ne peut pas se lire sur cette colonne telle quelle** ; il faut la comparaison à charge égale prévue en L6.

**D3.8 — Budget GPU : pas de recherche d'hyperparamètres étendue, et on le dit.** Une exécution complète de `gnn_gatv2` coûte 2,4 h sur la machine disponible ; la grille de 12 à 24 configurations prévue en A5 coûterait 30 à 60 h de GPU. Décision : garder la configuration initiale (valeurs usuelles pour un GNN de cette taille), la défendre par les courbes d'apprentissage (convergence propre, arrêt précoce à 61, pas de divergence train/val), et dépenser le budget sur ce qui répond aux hypothèses — ablations et répétitions — plutôt que sur un réglage fin.
- Alternative rejetée : une grille réduite sur sous-échantillon, qui classerait les configurations sur un proxy (moins de scénarios, moins d'époques) sans garantie que le classement tienne à plein budget. Mieux vaut une configuration non réglée et annoncée qu'un réglage dont on ne peut pas défendre la validité.

**D3.9 — Répétitions : 3 graines sur `mlp_flat` (CPU), 1 seule sur les GNN (GPU), et on s'appuie sur le bootstrap pour l'incertitude.** `mlp_flat` sur 3 graines donne **0,4032 ± 0,0020** de MAE_log sur le test (graines 0/1/2 : 0,4011 / 0,4050 / 0,4035). L'écart-type entre graines vaut donc **0,5 % de la valeur**, alors que l'écart entre `mlp_flat` et `gnn_gatv2` est de 0,275 — plus de cent fois l'écart-type. Avec 934 542 flux d'entraînement et 201 953 flux de test, le bruit de graine est négligeable devant les effets qu'on annonce.
- Décision : ne pas dépenser 2,4 h de GPU par graine supplémentaire de GNN. L'incertitude du tableau final vient du **bootstrap sur les scénarios** (volet C, `src/bootstrap.py`), qui mesure en plus la variabilité due à l'échantillon de scénarios — celle qui compte vraiment ici, puisque les flux d'un même scénario sont corrélés (D2.1).
- Honnêteté : cette mesure est faite sur `mlp_flat`, pas sur le GNN. Elle montre que le bruit de graine est petit **sur ce jeu et pour un modèle entraîné dessus**, pas que celui du GNN est identique (868 481 paramètres contre 4 225, et une courbe de val plus bruitée). À dire tel quel si la question est posée.
- Alternative rejetée : annoncer une moyenne ± écart-type sur 5 graines de GNN (12 h de GPU) pour un intervalle que le bootstrap donne gratuitement et sur la bonne source de variabilité.

**D3.10 — H2 sur nos modèles : la bonne frontière est « voit la charge ou non », et la première moitié de H2 est fausse sur ce jeu.** Avec `events.parquet` du volet B (τ = 1,05), MAE_log test non-événement / événement : `gnn_gatv2` 0,121 / 0,129 (**+0,008**) ; `mlp_flat` 0,290 / 0,427 (+0,136) ; `analytic_mm1` 0,363 / 0,708 (+0,345) ; `gnn_k0` 1,244 / 0,699 (−0,545) ; naïve 1,322 / 0,719 (−0,603).
- **Volet comparatif de H2 confirmé, et c'est l'argument le plus fort du GNN** : il est quasi insensible à la frontière (+0,008) là où le modèle plat à information équivalente se dégrade 17 fois plus. Un orchestrateur s'intéresse précisément au régime chargé.
- **Volet « tous les modèles se dégradent » infirmé** : naïve et `gnn_k0` font *mieux* sur les événements. Explication : 82,9 % des flux du test sont en événement (volet B, D3.3), donc une constante apprise sur train est calée sur la majorité — c'est-à-dire sur les flux en événement — et devient fausse sur la minorité non chargée. L'hypothèse H2 supposait implicitement des événements rares ; à réécrire dans le rapport au lieu de la déclarer « vérifiée ».
- **Au palier sévère du volet B (`is_event_tau_2`, 16 % des flux), H2 tient en entier** : reste / sévère = naïve 0,770 / 1,098 (+0,328) ; `gnn_k0` 0,737 / 1,078 (+0,340) ; `analytic_mm1` 0,585 / 0,983 (+0,398) ; `mlp_flat` 0,373 / 0,561 (+0,188) ; `gnn_gatv2` 0,124 / 0,145 (**+0,021**). L'inversion à τ = 1,05 était donc bien un artefact de proportion. Le GNN se dégrade 9 fois moins que le modèle plat.
- Calculé dans `notebooks/L5_gnn.py` à partir de `evaluation_frame()` et `events.parquet`, **pas** dans `src/evaluate.py`, qui ne produit que `event` / `non_event` depuis `is_event`. L'y ajouter servirait aux trois volets mais touche un fichier commun (§4.1 du plan) : à décider à trois.
- Note pour le volet B : la colonne s'appelle `is_event_tau_2` et non `is_event_tau_2_0` comme l'indique D4.2 (`tau_col()` formate en `%g`, donc 2.0 → « 2 » et 1.0 → « 1 »). Documentation à corriger, code correct.

**D3.11 — Pas d'ablation par groupe de variables sur le GNN, et pas de GNN sur le découpage « décalage de charge » (B4).** Chacune demanderait un entraînement complet de 2,4 h (4 groupes de variables = ~10 h ; shift = 2,4 h de plus).
- Raison : même arbitrage que D3.8 — le budget GPU va aux expériences qui tranchent une hypothèse (K = 0, contrôle plat, événements), pas à celles qui la nuancent.
- Compensation : le volet C fait l'ablation par groupe sur LightGBM avec les mêmes `FEATURE_GROUPS`, ce qui répond à « quelles variables comptent » sans mobiliser le GPU. Le volet B a déjà mesuré les baselines sur le décalage de charge.

## 2026-10-29 — Volet A : apres integration des volets B et C

**D3.12 — H1 confirmée contre la *meilleure* baseline plate, avec intervalle de confiance.** Depuis l'intégration du volet C, le modèle plat le plus fort n'est plus `mlp_flat` (0,403) mais **`lgbm` (0,352 [0,327 ; 0,379])**. Sur test, MAE_log : `gnn_gatv2` **0,128 [0,118 ; 0,138]**. Différence appariée par bootstrap sur les scénarios (`results/bootstrap_diff.csv`, volet C) : **−0,2243 [−0,2484 ; −0,2023]**, le GNN devant dans **100 %** des 1 000 rééchantillonnages.
- Conséquence pour le rapport : annoncer **× 2,75 contre `lgbm`** et non × 3,1 contre `mlp_flat`. C'est plus conservateur et bien plus défendable — on compare au meilleur adversaire, pas au plus faible.
- `mlp_flat` reste le contrôle méthodologique utile (même boucle d'entraînement, mêmes features, sans graphe), mais `lgbm` le bat (+0,0512 [+0,0377 ; +0,0638]) : un GBM règle mieux un problème tabulaire qu'un MLP, ce qui est attendu et à dire.

**D3.13 — H3 (« le GNN généralise mieux aux topologies inédites ») n'est PAS confirmée.** Comparaison à charge égale du volet C (`results/bootstrap_load_matched.csv`, repondération par classes de ρ max du test, 5 classes, 158 à 309 scénarios inédits par classe) :
- `gnn_gatv2` : écart inédites − test **brut** −0,0125 [−0,0273 ; +0,0022] → **à charge égale** −0,0039 [−0,0189 ; +0,0119]. L'intervalle contient 0 : aucune preuve d'une meilleure généralisation.
- Le cas qui prouve le mécanisme : `gnn_k0` passe de −0,0574 [−0,1141 ; **−0,0044**] (significatif) à +0,0007 [−0,0592 ; +0,0545] (non significatif) ; idem `ridge`, −0,0570 [−0,0977 ; −0,0181] → −0,0185 [−0,0598 ; +0,0208]. L'« avantage » sur les topologies inédites était bien un effet de charge (D3.7), maintenant quantifié.
- À écrire tel quel : H3 est **infirmée faute de preuve**, pas confirmée. Le GNN ne se dégrade pas non plus — il transfère sans perte, ce qui est déjà le comportement attendu d'un modèle inductif (poids partagés, aucune dépendance au nombre de nœuds).

**D3.14 — Corrections d'intégration (fichiers du volet C).** `notebooks/L3_importance.py` écrivait ses figures dans un dossier `figures/` à la racine : remplacé par `FIG_DIR` (`reports/figures/`) comme le reste du projet, les 5 PNG déplacés. `src/final_table.py` portait une note codée en dur (« seul lgbm a 5 graines ») devenue fausse après l'arrivée de `mlp_flat` (3 graines) : la note est maintenant calculée depuis `metrics_all.csv`.
- Reste à faire par le volet C : aucune entrée de JOURNAL pour ses décisions (grille d'alpha du ridge, arrêt précoce LightGBM, B = 1 000 et bootstrap par scénario, 5 classes de charge) — §4.5 du plan demande que chacun signe les siennes.

## 2026-10-29 — Volet C (données et ML classique)

*(Entrées D5.1 à D5.4 rédigées par Claude à partir du code et des résultats, en l'absence d'entrées du volet C —
voir §4.5 du plan : chaque personne signe ses propres décisions. À relire, corriger et signer avant la soutenance.)*

**D5.1 — Règle de transformation (log10 / log1p / brut) décidée par variable, sur train seulement.**
- Règle : log10 si strictement positif et max/min > 100 (débits, capacités, termes de file) ; log1p si non négatif et max > 100 (compteurs) ; brut sinon (indicatrices, ratios bornés). Décidée sur `X[is_train]`, appliquée telle quelle à val/test (`fit_rules`/`apply_rules`, `src/baselines_ml.py`).
- Alternative rejetée : une seule transformation pour toutes les variables, qui écraserait la variance utile des variables de charge sous celle des grandes valeurs brutes.

**D5.2 — Grille complète d'α pour Ridge, publiée dans `results/ridge_alpha_grid.csv` ; la régularisation ne change presque rien.**
- Constat : MAE_log de val quasi constante de α = 1e-3 à 10 (0,499267 à 0,499270, variation en 5ᵉ décimale), puis légère dégradation au-delà (0,503 à α = 1e4). Avec 934 542 lignes de train pour 27 variables, le problème est déjà très bien déterminé : pas de surapprentissage à corriger dans la plage testée.
- Conséquence : la limite de Ridge (MAE_log test 0,510) est sa forme linéaire, pas un défaut de réglage — à dire explicitement, ce n'est pas un échec de la recherche d'hyperparamètre.

**D5.3 — LightGBM : arrêt précoce (patience 100) sur la MAE_log de val, jusqu'à 3000 arbres ; 5 graines (aléa du sous-échantillonnage des lignes/colonnes), 1 seule graine pour Ridge (déterministe).**
- Résultat : LightGBM 0,352 [0,326 ; 0,379] sur test (5 graines, écart-type 0,0019), Ridge 0,510 [0,481 ; 0,538] (1 graine) — facteur 1,45. Courbe train/val publiée (`L3_01_courbe_apprentissage.png`) comme preuve que le nombre d'arbres n'est pas choisi arbitrairement.
- Ablation par groupe (réentraînement sans le groupe, pas mise à zéro après coup) : `utilization` (+0,0582) et `wfq` (+0,0342) dominent largement le bruit entre graines (écart-type 0,0019) ; `capacity`, `path`, `scenario`, `traffic` sont dans le bruit. Cohérent avec D1.5 (le trafic réel est constant par slice) et avec D3.12 (le GNN calcule ces mêmes grandeurs de charge le long du graphe).

**D5.4 — Bootstrap sur les SCÉNARIOS (jamais les flux), B = 1000, poids multinomiaux partagés entre modèles d'un même groupe (différences appariées) ; comparaison « topologies inédites vs test » repondérée par 5 classes de ρ max du test pour isoler l'effet de charge.**
- Raison : les flux d'un même scénario partagent des liens et des files, donc sont corrélés (D2.1) — un bootstrap sur les flux sous-estimerait l'incertitude réelle. Le partage des poids entre modèles resserre l'intervalle de la différence par rapport à deux intervalles calculés séparément.
- Résultat clé (voir aussi D3.13) : l'écart brut « inédites moins test » est négatif pour tous les modèles (apparence de meilleure généralisation), mais devient non significatif à charge égale pour `gnn_gatv2` (−0,0039 [−0,0189 ; +0,0119]) et change même de signe pour `gnn_k0` et `ridge` — la preuve quantitative que l'écart observé était un effet de charge, pas de topologie.
- Alternative rejetée : bootstrap sur les flux (sous-estimerait l'incertitude) ; comparer les bruts sans repondération (confondrait topologie nouvelle et charge différente, cf. D2.1).

**D3.15 — H4 confirmée : les flux affamés (`event_type == both`) sont là où tous les modèles échouent le plus, et où l'avantage du GNN est le plus net.** MAE_log test par `event_type` (moyenne sur les graines, `events.parquet` du volet B) :

| event_type | n flux | gnn_gatv2 | lgbm | mlp_flat | naive |
|---|---|---|---|---|---|
| none | 34 632 | 0,121 | 0,229 | 0,290 | 1,322 |
| congestion | 148 514 | 0,106 | 0,295 | 0,337 | 0,576 |
| **both** (congestion + famine) | 18 807 | **0,315** | 1,026 | 1,136 | 1,850 |

`both` (9,3 % des flux de test) est la catégorie où le délai réel est le plus extrême (les flux affamés, journal D1.8) : tous les modèles s'y dégradent fortement, mais le GNN reste 3,3 fois meilleur que `lgbm` (contre 2,8 fois sur `congestion` et 1,9 fois sur `none`) — l'écart relatif se creuse avec la sévérité. En valeur absolue, même le GNN se dégrade nettement (0,106 → 0,315) : ce n'est pas un modèle parfait sur la famine, juste le moins mauvais.
- Pas de catégorie « famine seule » (D4.3 volet B : la règle de famine implique congestion), donc `both` est la meilleure approximation disponible de « flux affamé ».
- Confirme H4 (annexe du plan) sans ambiguïté, contrairement à H2 où la première moitié n'était vraie qu'au palier sévère (D3.10).

**D3.16 — Deuxième graine de `gnn_gatv2` : arrêtée à la main par manque de temps (epoch 80), checkpoint epoque 77 utilisé.** Val MAE_log 0,1314 (vs 0,1282 pour la graine 0) ; le script sauvegarde un checkpoint à chaque amélioration, donc l'arrêt manuel n'a rien perdu — seule la prédiction finale (val/test/test_unseen_topo) a dû être relancée depuis le checkpoint (`src.train_gnn.predict`, un script ponctuel, pas d'entraînement).
- MAE_log test sur les 2 graines : 0,1277 / 0,1422, **moyenne 0,135, écart-type 0,0102** (7,6 % de la valeur). C'est 15× plus que l'écart-type mesuré sur `mlp_flat` (0,5 %, D3.9) : le bruit de graine du GNN n'est **pas** négligeable comme celui du MLP plat — la mise en garde de D3.9 était justifiée.
- Conséquence sur les chiffres déjà publiés : H1 tient toujours (diff appariée `gnn_gatv2 − lgbm` = −0,217 [−0,241 ; −0,196], GNN devant dans 100 % des tirages ; ratio **× 2,61**, plus proche de la réalité que le × 2,75 à 1 graine). H3 reste infirmée à charge égale (−0,0068 [−0,0214 ; +0,0085], contient 0) ; l'écart **brut** devient lui significatif (−0,0151 [−0,0301 ; −0,0007]) — un exemple de plus, et plus net, de pourquoi la correction à charge égale du volet C est nécessaire : sans elle on aurait maintenant conclu, à tort, à une vraie différence.
- La diapositive de la soutenance garde les chiffres à 1 graine (× 2,75, 0,128) : décision délibérée de ne pas la refaire sous pression de temps, l'écart est mineur et ne change aucune conclusion.
