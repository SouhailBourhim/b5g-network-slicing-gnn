# 00 — Aide-mémoire de soutenance

Document commun, **une section par volet**. Chacun rédige et défend la sienne.
Chiffres : `results/metrics_all.csv`. Détail des choix : `01_exploration.md`, `02_problematique.md`,
`05_gnn.md`, et `JOURNAL.md` pour les décisions datées.

---

## Le projet en 10 points

*(section commune — à finaliser en L6, une fois les volets B et C livrés)*

1. Le jeu de Farreras et al. (2024) contient 7 127 simulations paquet par paquet de réseaux de transport :
   topologie, capacités, routage, slices admises (eMBB / URLLC / mMTC), trafic → délai, gigue et pertes par flux.
2. Une simulation coûte **10,8 h en moyenne**. On apprend un modèle « what-if » qui répond en millisecondes, pour
   qu'un orchestrateur puisse tester une admission de slice avant de la décider.
3. Cible : **log10 du délai moyen par flux** (le délai couvre 8,3 ordres de grandeur).
4. Le découpage se fait **par scénario**, jamais par flux — prouvé par une expérience de fuite (facteur 3,15).
5. Un second test tient **24 topologies jamais vues** à l'écart.
6. *(volet B)* Définition des événements : congestion (ρ ≥ τ) et famine (part WFQ quasi nulle sur lien saturé).
7. *(volet C)* Baselines plates : Ridge, LightGBM, et importance des variables par SHAP.
8. *(volet A)* Chaque scénario devient un **graphe flux / lien** et un GNN à passage de messages prédit le délai.
9. Résultat principal : **0,128 de MAE_log** contre 0,403 pour le MLP plat et 0,822 pour la baseline naïve.
10. *(commun L6)* Réponse finale : dans quelles conditions le GNN apporte-t-il vraiment quelque chose ?

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

**Étape 7 — la phrase qui conclut (15 s).** *« Un modèle plat doit résumer tout ça à l'avance avec des agrégats
choisis à la main — ρ max du chemin, somme M/M/1. Le GNN calcule ses propres agrégats, le long de la vraie
structure. C'est ce qui fait passer l'erreur de 0,403 à 0,128. »*

### Trois chiffres à connaître par cœur

| | |
|---|---|
| MAE_log du GNN sur test | **0,128** — soit 12 % d'erreur médiane sur le délai en secondes |
| Le MLP plat, mêmes informations | **0,403** → le passage de messages vaut **× 3,1** |
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
c'est `mlp_flat` → `gnn_gatv2` : 0,403 → 0,128, soit × 3,1, à information équivalente.

**2. « Votre modèle généralise mieux aux topologies inédites (0,115 < 0,128). »** Non plus. La baseline naïve, qui
prédit une constante par slice, y gagne aussi (0,822 → 0,790) — or elle ne peut pas « mieux généraliser ». Ces 24
topologies sont simplement **moins chargées** (ρ max médian 1,35 contre 1,63). La comparaison honnête se fait à
charge égale, en L6.

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

*(à rédiger par le volet B : définition de τ et sa sensibilité, les baselines M/M/1 et M/M/1/K, la vérification
visuelle des scénarios, l'analyse d'erreur en fonction de ρ.)*

---

## Volet C — Données, formulation et ML classique

*(à rédiger par le volet C : l'expérience de fuite et le protocole de découpage, Ridge et LightGBM, SHAP et
ablations par groupe, les intervalles de confiance par bootstrap.)*

---

## La liste de contrôle du guide

À cocher en L6, chaque item renvoyant à l'endroit qui le justifie (section 13 du guide du professeur).
