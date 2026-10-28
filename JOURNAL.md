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
