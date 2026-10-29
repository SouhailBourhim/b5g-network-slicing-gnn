# 00 — Aide-mémoire de soutenance

Document commun, **une section par volet**. Chacun rédige et défend la sienne.

> **Mise à jour D3.16** : une 2e graine de `gnn_gatv2` est arrivée après la rédaction de ce fichier (MAE_log test moyenne sur 2 graines : **0,135 ± 0,0102**, contre 0,128 à 1 graine). Les conclusions (H1, H3) ne changent pas, seules les décimales bougent légèrement. Chiffres à jour : `notebooks/L5_gnn.py`/`L6_comparaison.py` (recalculent depuis `results/`) et `JOURNAL.md` D3.16.
Chiffres : `results/metrics_all.csv`. Détail des choix : `01_exploration.md`, `02_problematique.md`,
`05_gnn.md`, et `JOURNAL.md` pour les décisions datées.

---

## Le projet en 10 points

*(section commune — les trois volets sont livrés ; le point 10 attend la synthèse L6)*

1. Le jeu de Farreras et al. (2024) contient 7 127 simulations paquet par paquet de réseaux de transport :
   topologie, capacités, routage, slices admises (eMBB / URLLC / mMTC), trafic → délai, gigue et pertes par flux.
2. Une simulation coûte **10,8 h en moyenne**. On apprend un modèle « what-if » qui répond en millisecondes, pour
   qu'un orchestrateur puisse tester une admission de slice avant de la décider.
3. Cible : **log10 du délai moyen par flux** (le délai couvre 8,3 ordres de grandeur).
4. Le découpage se fait **par scénario**, jamais par flux — prouvé par une expérience de fuite (facteur 3,15).
5. Un second test tient **24 topologies jamais vues** à l'écart.
6. *(volet B)* Définition des événements depuis les entrées : congestion (ρ ≥ 1,05) et famine (part WFQ quasi
   nulle sur lien saturé) ; 82,9 % des flux de test sont en événement — propriété du jeu, pas un seuil arbitraire.
7. *(volet C)* Baselines plates : Ridge (0,510) et LightGBM (**0,352**, meilleur modèle plat du projet), importance
   des variables confirmée par gain et SHAP, ablations par groupe.
8. *(volet A)* Chaque scénario devient un **graphe flux / lien** et un GNN à passage de messages prédit le délai.
9. Résultat principal : `gnn_gatv2` **0,128 [0,118 ; 0,138]** de MAE_log contre **0,352** pour `lgbm` (le meilleur
   modèle plat, pas seulement le MLP) et 0,822 pour la baseline naïve — différence significative (IC bootstrap
   exclut 0, GNN devant dans 100 % des rééchantillonnages).
10. *(commun L6, à écrire)* Réponse finale : dans quelles conditions le GNN apporte-t-il vraiment quelque chose ?
    Ce qu'on sait déjà : oui contre le meilleur modèle plat (H1, D3.12) ; pas de preuve de meilleure généralisation
    aux topologies inédites une fois la charge égalisée (H3, D3.13) ; très robuste aux événements sévères
    (dégradation +0,021 contre +0,166 pour `lgbm`, soit 8 fois moins — chiffre recalculé avec `lgbm` après
    intégration du volet C, à verser au JOURNAL en D3.15).

---

## Volet A — Graphe et GNN (Souhail)

### Script « tableau blanc » : dessiner le graphe et le GNN sans notes

**Étape 1 — le réseau (20 s).** Dessiner 4 nœuds reliés par 3 liens orientés. Annoter un lien : « capacité
75 Mb/s ». Dire : *« voilà un scénario : une topologie, des capacités, un routage. »*

**Étape 2 — les flux qui se partagent un lien (30 s).** Tracer deux chemins de couleurs différentes qui passent par
**le même lien central**. Dire : *« le flux bleu est lent non pas à cause de ce qu'il est, mais à cause du flux
rouge qui occupe le même lien. C'est ça, le problème : le délai d'un flux dépend des autres flux. »*

**Étape 3 — passer au graphe du modèle (40 s).** À droite, redessiner en deux colonnes : les **flux** à gauche
(ronds), les **liens** à droite (carrés). Tirer un trait de chaque flux vers chaque lien qu'il traverse. Dire :
*« une arête = un saut. Le graphe n'est plus le réseau, c'est "qui traverse quoi". »*

**Étape 4 — ce qu'on écrit où (30 s).**
- sur un **rond** (flux) : type de slice, débit, longueur du chemin ;
- sur un **carré** (lien) : capacité, ρ = charge / capacité, nombre de flux ;
- sur une **arête** : la **part WFQ**. Insister : *« elle n'est ni au flux ni au lien, elle est au couple :
  le même lien garantit 10 % à l'un et 1 % à l'autre. »*

**Étape 5 — le passage de messages (40 s).** Dessiner une flèche colonne gauche → colonne droite, puis une flèche
retour. Dire : *« d'abord chaque lien écoute les flux qu'il porte et se met à jour ; ensuite chaque flux écoute ses
liens, déjà à jour. Un aller-retour = une itération. On en fait 3. »* Puis, en montrant le dessin :
*« après 3 itérations, le flux bleu "sait" quelque chose des liens de ses concurrents : la congestion se propage
dans le graphe comme elle se propage dans le réseau. »*

**Étape 6 — la sortie (15 s).** Flèche depuis un rond vers « MLP → log10(délai) ». Dire : *« on lit la prédiction
sur les nœuds flux uniquement. »*

**Étape 7 — la phrase qui conclut (15 s).** *« Un modèle plat doit résumer tout ça à l'avance avec des agrégats —
soit calculés à la main (ρ max, somme M/M/1, comme pour LightGBM), soit appris sur les variables brutes du flux
seul (le MLP plat). Le GNN calcule ses propres agrégats le long de la vraie structure, et ça se voit : 0,128 contre
0,352 pour le meilleur des deux, LightGBM. »*

### Trois chiffres à connaître par cœur

| | |
|---|---|
| MAE_log du GNN sur test | **0,128 [0,118 ; 0,138]** — soit 12 % d'erreur médiane sur le délai en secondes |
| Le meilleur modèle plat, `lgbm` (volet C) | **0,352** → le GNN fait **× 2,75** mieux, IC bootstrap exclut 0 |
| L'ablation K = 0 | **0,792**, à peine mieux que la baseline (0,822) |

### Le choix méthodologique à défendre en 30 secondes : la conception du graphe

*« On a mis les flux et les liens comme deux types de nœuds, et la part WFQ sur l'arête. Pourquoi ? Parce que le
délai d'un flux est causé par la contention sur les liens qu'il partage, et que la part WFQ appartient au couple
(flux, lien) : sur un même lien, deux slices n'ont pas la même garantie. La mettre sur le nœud lien obligerait à en
prendre la moyenne, ce qui effacerait exactement ce qui distingue un flux affamé d'un flux servi. Et on ne l'a pas
seulement argumenté : le même réseau sans passage de messages tombe à 0,792. »*

### Les deux pièges à désamorcer avant qu'on les trouve

**1. « Votre ablation K = 0 prouve que le passage de messages vaut × 6. »** Non, et il faut le dire soi-même.
Avec K = 0 les nœuds lien sont **déconnectés** de la lecture : le modèle ne voit ni ρ, ni capacité, ni WFQ. Donc
`gnn_k0` mesure l'apport du **graphe entier**, information comprise. La bonne mesure du passage de messages seul,
à information strictement égale, c'est `mlp_flat` → `gnn_gatv2` (même features brutes du flux) : 0,403 → 0,128,
soit × 3,1. Le × 2,75 contre `lgbm` est une comparaison différente et complémentaire : `lgbm` reçoit des agrégats
du chemin calculés à la main (ρ max, sommes M/M/1), donc plus d'information que `mlp_flat` — et le GNN le bat quand
même, en calculant ses propres agrégats le long du graphe plutôt que de les recevoir tout faits.

**2. « Votre modèle généralise mieux aux topologies inédites (0,115 < 0,128). »** Non, et c'est **tranché**, pas
seulement suspecté. La baseline naïve, qui prédit une constante par slice, y gagne aussi (0,822 → 0,790) — or elle
ne peut pas « mieux généraliser ». Ces 24 topologies sont simplement **moins chargées** (ρ max médian 1,35 contre
1,63). Le volet C a quantifié l'effet par un bootstrap repondéré à charge égale (JOURNAL D3.13) : l'écart du GNN
tombe de −0,0125 [−0,0273 ; +0,0022] brut à **−0,0039 [−0,0189 ; +0,0119]** à charge égale — l'intervalle contient
0, aucune preuve de meilleure généralisation. H3 est infirmée.

### Limites du volet A

- **Pas de recherche d'hyperparamètres** : 2,4 h par exécution, la grille prévue en coûterait 30 à 60. La
  configuration est défendue par les courbes d'apprentissage (convergence propre, arrêt précoce à l'époque 61),
  pas par un réglage.
- **Une seule graine pour le GNN.** Mesuré sur `mlp_flat`, le bruit de graine vaut 0,5 % de la valeur (3 graines,
  écart-type 0,002) contre un écart de 0,275 entre modèles : l'aléa d'initialisation ne décide de rien ici. Mais
  c'est mesuré sur le MLP, pas sur le GNN.
- **Pas d'ablation par groupe de variables sur le GNN** (une exécution complète par groupe). Le volet C fait
  l'équivalent sur LightGBM.
- **Données simulées**, réseau de transport **sans RAN**, scénarios **statiques** (pas de dynamique temporelle),
  hypothèses M/M/1 et M/M/1/K pour les baselines analytiques.

---

## Volet B — Réseau, baselines analytiques et événements

> **Section rédigée par Claude à partir de `docs/explications/04_evenements.md`** (déjà écrit par le volet B),
> pour que le squelette commun avance. À ajuster et signer par la personne qui défend ce volet.

### Script « tableau blanc » : la courbe du coude et les deux événements

**Étape 1 — la formule M/M/1 (15 s).** Écrire `T = S / (1 − ρ)`. Dire : *« ρ = charge offerte / capacité. Quand ρ
s'approche de 1, le dénominateur s'approche de 0 : le délai explose. »*

**Étape 2 — tracer la courbe délai médian vs ρ max (30 s).** Axe horizontal ρ max de 0 à 2, axe vertical délai
médian en log. Dessiner une courbe plate puis qui monte brutalement vers ρ = 1. Dire : *« c'est exactement ce que
dit la formule : un coude, pas une pente douce. »* Marquer le coude à **ρ = 1,05**, pas 1,0 : *« la théorie dit 1,
les données disent 1,05 — c'est cet écart qu'on a mesuré plutôt que supposé. »*

**Étape 3 — M/M/1/K, le mur plutôt que l'infini (20 s).** Tracer une seconde courbe qui plafonne au lieu
d'exploser. Dire : *« un vrai réseau a un tampon fini. Passé ce mur, l'excès de trafic n'attend plus, il se perd.
C'est pour ça que M/M/1/K reste défini même quand ρ dépasse 1, contrairement à M/M/1. »*

**Étape 4 — les deux événements (30 s).** Sur un petit schéma de réseau, colorer un lien en rouge (ρ ≥ 1,05) :
*« ça, c'est une congestion. »* Puis montrer un flux dont la part de file WFQ est quasi nulle sur ce lien rouge :
*« et ça, une famine : le flux a une part garantie proche de zéro sur un lien déjà saturé. » * Insister : *« les
deux se calculent uniquement depuis les entrées — jamais depuis le délai — sinon la définition ne serait pas
utilisable pour une configuration qu'on n'a pas encore simulée. »*

**Étape 5 — le piège de la proportion (20 s).** *« Avec ce seuil, 83 % des flux du test sont "en événement". Ce
n'est pas un seuil truqué : c'est que ce jeu de données simule surtout des réseaux chargés. On a donc ajouté un
palier "sévère" à ρ ≥ 2 (16 % des flux), qui est la comparaison qui compte vraiment. »*

### Trois chiffres à connaître par cœur

| | |
|---|---|
| Seuil de congestion | **τ = 1,05** — coude mesuré sur train (+0,742 en log10 du délai médian entre (1,0;1,05] et (1,05;1,1], contre +0,134 juste avant) |
| Couverture | **82,9 % des flux de test en événement**, 98,1 % des scénarios — propriété du jeu, pas un seuil arbitraire |
| Meilleure baseline analytique | `analytic_mm1` **0,649** en MAE_log test, loin devant `analytic_mm1k_link` (0,839) malgré une hypothèse moins réaliste |

### Le choix méthodologique à défendre en 30 secondes : définir l'événement depuis les entrées, jamais depuis le délai

*« Si on définissait l'événement par "délai > 1 s", on sélectionnerait les cas où les modèles se trompent le plus
par construction — le résultat serait circulaire. Et un orchestrateur ne connaît pas le délai avant de simuler : il
doit pouvoir dire "cette configuration va créer un événement" à partir de ce qu'il configure, pas de ce qu'il
observerait après coup. ρ et la part WFQ se calculent tous les deux depuis les entrées — c'est la seule définition
utilisable en production. »*

### Le piège à désamorcer avant qu'on le trouve

**« M/M/1/K est théoriquement plus réaliste près de la saturation, pourquoi est-il moins bon que M/M/1 tout
court ? »** Dire honnêtement que ce n'est pas expliqué : `analytic_mm1k_link` (0,839) ne bat même pas la baseline
naïve (0,822), alors qu'il sous-estime fortement le délai dès que ρ_max > 1,05 (−0,67 à −0,80 en log10 sur val).
Pistes non vérifiées : le plafond de file (K = 33 paquets) et l'hypothèse de Poisson sous-estiment la rafale quand
plusieurs sauts sont saturés en même temps. Ce qui est établi : sur ce jeu, la formule la plus simple est la
référence la plus dure à battre.

### Limites du volet B

- **Le jeu n'a pas d'axe temporel** : un événement est un état d'une configuration statique, jamais un changement
  dans le temps. À dire explicitement, car le guide parle de « changement brusque ».
- **La famine est toujours incluse dans la congestion** (la règle exige ρ_lien ≥ τ, donc ρ_max ≥ τ) : il n'existe
  pas de cas « famine seule » dans cette définition, seulement `none`, `congestion`, `both`.
- **La régression de M/M/1/K n'est pas expliquée** (voir le piège ci-dessus) — une limite assumée, pas cachée.
- **Au seuil principal (τ = 1,05), « événement vs non-événement » oppose 83 % des flux à 17 % peu chargés** : peu
  informatif seul. Le palier sévère (τ = 2,0) est la comparaison qui tranche réellement (volet A, JOURNAL D3.10).

---

## Volet C — Données, formulation et ML classique

> **Section rédigée par Claude à partir de `docs/explications/02_problematique.md` et `03_baselines.md`**
> (ce dernier lui-même un brouillon, voir sa note en tête). À ajuster et signer par la personne qui défend ce volet.

### Script « tableau blanc » : la preuve de la fuite, puis Ridge contre LightGBM

**Étape 1 — la fuite, en une image (30 s).** Dessiner deux scénarios comme deux sacs de billes, chaque bille = un
flux. Dire : *« les flux d'un même scénario partagent des liens et des files — ce sont presque des jumeaux. »*
Barrer un découpage qui mélange les billes des deux sacs au hasard (split par flux) ; entourer un découpage qui
garde chaque sac entier d'un seul côté (split par scénario). Donner le chiffre : *« avec le mauvais découpage,
l'erreur annoncée est 0,13 ; l'erreur réelle sur des scénarios jamais vus est 0,40 — 3,15 fois pire que ce qu'on
croyait. »*

**Étape 2 — même donnée, deux modèles (30 s).** Écrire côte à côte : `Ridge : y = Σ wᵢ xᵢ` et
`LightGBM : des coupures en cascade (x₁ > seuil ? ...)`. Dire : *« Ridge ne peut tracer qu'une droite dans l'espace
des variables. Le délai, lui, a un coude — il explose près de ρ = 1, puis plafonne à cause du tampon fini. Un
arbre peut représenter ce coude directement avec une coupure ; une droite ne peut que faire la moyenne entre avant
et après. »*

**Étape 3 — pourquoi régler α ne change rien (20 s).** Montrer la courbe plate de `ridge_alpha_grid.csv`. Dire :
*« avec 935 000 lignes d'entraînement pour 27 variables, il n'y a pas assez peu de données pour que la
régularisation ait un rôle à jouer. Le problème n'est pas le réglage, c'est la forme du modèle. »*

**Étape 4 — quelles variables comptent (20 s).** Montrer la figure gain/SHAP : *« les deux méthodes, indépendantes,
s'accordent sur le top 3 : ρ max du chemin, et les deux termes de file M/M/1/K. Aucune variable de trafic brut n'y
apparaît — c'est la charge qui décide, pas le profil du flux. »*

### Trois chiffres à connaître par cœur

| | |
|---|---|
| La fuite mesurée | Split par flux : erreur annoncée **0,127**, réelle **0,399** — facteur **3,15** |
| Meilleur modèle plat | `lgbm` **0,352 [0,326 ; 0,379]** en MAE_log test — bat Ridge (0,510) d'un facteur 1,45 |
| Ablation la plus forte | retirer `utilization` coûte **+0,058** de MAE_log, loin au-dessus du bruit entre graines (0,0019) |

### Le choix méthodologique à défendre en 30 secondes : prouver la fuite plutôt que l'affirmer

*« On aurait pu se contenter de dire "on découpe par scénario pour éviter la fuite" — c'est ce que tout le monde
fait. Au lieu de ça on a mesuré l'ampleur de l'erreur qu'un découpage par flux aurait causée : 3,15 fois trop
optimiste. Ce n'est pas un principe qu'on cite, c'est un chiffre qu'on a produit avec la même méthode que celle
utilisée ensuite pour tout le reste du projet. »*

### Le piège à désamorcer avant qu'on le trouve

**« LightGBM a 5 graines, Ridge une seule — c'est un traitement inégal ? »** Non : Ridge est un solveur fermé,
déterministe — relancer la même régression dix fois donne exactement le même résultat, il n'y a rien à moyenner.
LightGBM sous-échantillonne des lignes et des colonnes à chaque arbre (`subsample`, `colsample_bytree`), donc deux
exécutions diffèrent réellement. Moyenner sur les graines mesure cette variance-là, pas du favoritisme.

### Limites du volet C

- **Les coefficients de Ridge ne sont pas sauvegardés** dans un fichier (seulement affichés en console au moment de
  l'entraînement) : à relancer si le jury demande le détail.
- **L'ablation est faite sur une seule graine** de LightGBM par groupe retiré ; la comparaison au bruit entre
  graines (0,0019) sert de garde-fou, mais un intervalle de confiance propre demanderait 5 graines par ablation.
- **Le bootstrap suppose l'indépendance entre scénarios** une fois le tirage fait ; deux scénarios de la même
  topologie restent corrélés par construction (mêmes capacités, même routage), ce que le rééchantillonnage ne
  capture pas.

---

## La liste de contrôle

> Le guide du professeur (section 13) n'est pas dans ce dépôt — je n'ai pas son texte exact pour cocher ses items
> un par un. Ce qui suit reprend la checklist du **plan d'équipe** (`docs/plan_de_travail_equipe.docx`, §10), que
> j'ai : à recopier contre la vraie liste du guide avant la soutenance si elle diffère.

| Vérification | A | B | C |
|---|---|---|---|
| Le code tourne de bout en bout sur une machine propre (données data-v1) | ☑ (`train_gnn.py`, `train_mlp.py` testés `--max-scenarios`) | ☐ non revérifié par Claude | ☐ non revérifié par Claude |
| `load_splits()` et `src/metrics.py` utilisés, rien de maison | ☑ | ☑ | ☑ |
| Rien n'est ajusté sur val/test qui aurait dû l'être sur train | ☑ (normalisation sur train, `graphs/norm.json`) | ☑ (τ choisi sur train) | ☑ (scaler, règles log, α choisis sur train) |
| Prédictions via `save_predictions()`, dans `metrics_all.csv` | ☑ | ☑ | ☑ |
| Décisions dans `JOURNAL.md` avec raison chiffrée | ☑ (D3.1-D3.15) | ☑ (D4.1-D4.4, signées T.) | ☑ (D5.1-D5.4, **rédigées par Claude — à signer**) |
| `docs/explications/` : ≥ 3 questions du jury, 2 fonctions expliquées | ☑ `05_gnn.md` | ☑ `04_evenements.md` | ☑ `03_baselines.md` **(brouillon Claude — à relire)** |
| Tous les chiffres des textes sortent d'un fichier de `results/` | ☑ | ☑ | ☑ |
| Pull request relue par quelqu'un d'autre | ☐ à faire | ☐ à faire | ☐ à faire |

**Ce qui reste, au-delà de cette liste par volet :**
- Volet A : relancer `src.evaluate` / `src.bootstrap` / `src.final_table` une fois `gnn_gatv2 --seed 1` terminé.
- Volet C : signer D5.1-D5.4 (ou les corriger), relire `03_baselines.md`.
- Commun : `reports/rapport_final.md` n'existe pas encore (en cours) ; relecture croisée des trois PR.
