# 05 — Représentation en graphe et GNN (L5)

Les chiffres viennent de `results/metrics_all.csv`, `results/train_logs/*.csv` et `results/hpsearch_gnn.csv`.
Figures : `L5_01_exemple_graphe.png`, `L5_02_courbes_apprentissage.png`.

## Ce qu'on a fait

On a représenté **chaque scénario par un graphe** plutôt que par un vecteur de nombres : un nœud par flux, un nœud
par lien utilisé, et une arête par *saut* (un flux qui traverse un lien). On a ensuite entraîné un réseau de neurones
qui fait circuler de l'information le long de ces arêtes — les liens écoutent les flux qu'ils transportent, puis les
flux écoutent les liens de leur chemin — et qui prédit à la fin le délai de chaque flux en log10. Pour savoir si ce
mécanisme apporte vraiment quelque chose, on l'a comparé à deux contrôles : le **même réseau privé de passage de
messages** (`gnn_k0`) et un **MLP plat** recevant les mêmes informations résumées à la main (`mlp_flat`).
Résultat : 0,128 de MAE_log sur le test, contre 0,352 pour le meilleur modèle plat (`lgbm`, volet C), 0,403 pour
le MLP plat et 0,822 pour la baseline naïve.

**Définitions utiles.**
- *Graphe hétérogène* : un graphe où tous les nœuds ne sont pas de même nature (ici flux et liens), chacun avec ses
  propres variables.
- *Passage de messages* : chaque nœud reçoit un résumé de l'état de ses voisins, le combine avec le sien, et met à
  jour son état. Répété K fois, cela propage l'information à distance K dans le graphe.
- *Attribut d'arête* : une variable qui n'appartient ni au nœud de départ ni au nœud d'arrivée, mais au **couple**
  (ici : la part WFQ du flux sur ce lien précis).
- *Modèle inductif* : un modèle qui s'applique à un graphe de taille et de forme jamais vues, parce qu'il apprend
  une règle locale et non une table par nœud.

## Pourquoi ces choix

| Choix | Alternative rejetée | Pourquoi (chiffre ou principe) |
|---|---|---|
| Un graphe **flux / lien** par scénario | Un vecteur plat par flux | Le délai d'un flux dépend de **tous les autres flux qui partagent ses liens**, en nombre variable. Un vecteur de taille fixe ne peut en donner que des résumés choisis à la main. Mesuré : 0,403 → 0,128 en passant du résumé au graphe. |
| La part WFQ sur l'**arête** | Sur le nœud lien, ou sur le nœud flux | Elle dépend du couple : le même lien garantit des parts différentes à deux flux, et le même flux a des parts différentes selon le lien. La mettre sur un nœud serait faux. |
| Arêtes dans les **deux sens** (flux→lien et lien→flux) | Un seul sens | Il faut d'abord que le lien sache qui le charge, ensuite que le flux sache ce qu'il subit. Les deux demi-pas forment une itération. |
| **K = 3** itérations | K = 1 ; K très grand | Avec K = 1 un flux ne voit que ses liens ; avec K = 2 il voit les flux concurrents ; avec K = 3 il voit les liens de ces concurrents, donc la congestion qui se propage. Au-delà, coût en mémoire et risque de lissage excessif pour un gain non mesuré (budget GPU, D3.8). |
| **GATv2Conv** (variante principale) | GCN | GATv2 accepte `edge_dim`, donc la part WFQ entre dans le calcul, et il **pondère** les flux d'un lien (un eMBB à 22,5 Mb/s ne pèse pas comme un mMTC à 250 kb/s). Ses poids d'attention sont inspectables. |
| **EdgeSAGE** (variante écrite) | `SAGEConv` standard | `SAGEConv` **ignore** les attributs d'arête ; on a donc réécrit le message en `ReLU(W [h_src ‖ e])` pour que la part WFQ du saut atteigne le lien. |
| Résidu + LayerNorm à chaque demi-pas | Empiler les convolutions directement | Stabilise l'entraînement quand K grandit et évite que l'état initial du nœud soit écrasé. |
| Cible MSE sur log10(délai), sélection sur **MAE_log de val** | Sélectionner sur la MSE | La MSE entraîne (elle pénalise les gros ratés, les flux affamés), la MAE sélectionne car elle se lit directement comme une erreur relative typique. |
| Normalisation ajustée sur **train** uniquement | Sur tout le jeu | Sinon la moyenne et l'écart-type du test fuient dans l'entraînement (`graphs/norm.json`). |

**Pourquoi pas GCN, en deux phrases.** GCN est fait pour un graphe **homogène** : un seul type de nœud, un seul type
d'arête, alors que nos nœuds flux et liens n'ont ni les mêmes variables ni le même rôle. Et sa normalisation
symétrique pondère chaque voisin par 1/√(d_i·d_j), c'est-à-dire **uniquement par le degré** : la part WFQ du saut,
qui détermine pourtant le débit garanti au flux sur ce lien, ne pourrait pas entrer dans le message.

## Les équations de mise à jour

État caché d'un flux `h_f`, d'un lien `h_l`, attributs du saut `e_fl`. Au départ, `h_f = MLP_flux(x_f)` et
`h_l = MLP_lien(x_l)`. Pour k = 1 … K :

1. **Les liens écoutent les flux qu'ils transportent**
   `h_l ← LayerNorm( h_l + Dropout(ReLU( CONV_{f→l}( {h_f, e_fl : f traverse l}, h_l ) )) )`
2. **Les flux écoutent les liens de leur chemin** (déjà mis à jour à l'étape 1)
   `h_f ← LayerNorm( h_f + Dropout(ReLU( CONV_{l→f}( {h_l, e_fl : l sur le chemin de f}, h_f ) )) )`

Puis la lecture : `délai̅_f = MLP_lecture(h_f)`, en log10 des secondes.

Les deux convolutions possibles :
- **EdgeSAGE** : `CONV(·) = W_root·h_dst + moyenne_j ReLU( W_msg·[h_j ‖ e_ij] )` — moyenne, donc insensible au
  nombre de voisins, et le message voit l'arête.
- **GATv2** : `CONV(·) = Σ_j α_ij · W·h_j`, où `α_ij` est appris à partir de `(h_dst, h_j, e_ij)` puis normalisé par
  softmax sur les voisins — le lien peut donc accorder plus de poids à certains flux.

Avec **K = 0**, les deux boucles ne s'exécutent jamais : il ne reste que `MLP_lecture(MLP_flux(x_f))`. C'est le
modèle `gnn_k0`, et c'est exactement pour cela qu'il faut l'interpréter avec prudence (plus bas).

## Ce que montrent les résultats

**Le graphe d'un scénario** (figure `L5_01_exemple_graphe.png`, scénario 5792 du train). 7 flux, 15 liens utilisés,
27 sauts, les 3 types de slice. Les 7 flux traversent **le même lien 4→7**, qui est saturé (ρ ≥ 1) : c'est
précisément ce couplage qu'un vecteur plat par flux ne peut représenter qu'approximativement.

**MAE sur log10(délai)**, `results/metrics_all.csv` (`mlp_flat` moyenné sur 3 graines, les autres sur 1) :

| Modèle | val | test | topologies inédites | MdAPE test | paramètres | graines |
|---|---|---|---|---|---|---|
| `naive_median_slice` | 0,815 | 0,822 | 0,790 | 91,0 % | — | 1 |
| `gnn_k0` (K = 0) | 0,785 | 0,792 | 0,735 | 87,0 % | 52 097 | 1 |
| `mlp_flat` | 0,407 | 0,403 ± 0,002 | 0,372 | 43,9 % | 4 225 | 3 |
| **`gnn_gatv2`** (K = 3) | **0,128** | **0,128** | **0,115** | **12,1 %** | 868 481 | 1 |

**Le bruit de graine est négligeable ici** (D3.9) : `mlp_flat` donne 0,4011 / 0,4050 / 0,4035 sur trois graines, soit
un écart-type de 0,002 — **0,5 % de la valeur**, contre un écart de 0,275 entre `mlp_flat` et `gnn_gatv2`. Avec
934 542 flux d'entraînement, l'aléa d'initialisation ne décide de rien. C'est pourquoi on n'a pas dépensé 2,4 h de
GPU par graine supplémentaire de GNN ; l'intervalle de confiance du tableau final viendra du **bootstrap sur les
scénarios** (L6), qui mesure la bonne source de variabilité. À dire honnêtement : cette mesure est faite sur
`mlp_flat`, elle ne prouve pas que le GNN (868 481 paramètres) soit aussi stable.

Lecture du MdAPE : le GNN se trompe de **12 % sur le délai en secondes** pour un flux médian, contre 44 % pour le
MLP plat et 91 % pour la baseline.

**Par type de slice** (test, MAE_log) :

| Modèle | eMBB (8 630 flux) | URLLC (42 169) | mMTC (151 154) |
|---|---|---|---|
| `naive_median_slice` | 0,703 | 0,765 | 0,845 |
| `mlp_flat` | 0,504 | 0,420 | 0,393 |
| `gnn_gatv2` | 0,176 | 0,140 | 0,122 |

eMBB reste le plus difficile pour tous les modèles : ce sont les flux les plus gros (22,5 Mb/s), donc ceux qui
saturent les liens et subissent les délais les plus variables — et ils ne représentent que 4,3 % des flux.

**Attention au piège de l'ablation K = 0.** `gnn_k0` fait 0,792, à peine mieux que la baseline naïve (0,822). Ce
n'est **pas** la mesure de l'apport du passage de messages : avec K = 0 il n'y a aucune convolution, donc les nœuds
lien et les attributs de saut sont déconnectés de la lecture, et le modèle ne voit que les 8 variables du nœud flux
— **ni ρ, ni capacité, ni part WFQ, ni terme de file**. Il faut lire les deux ablations ensemble :

- **0,792 → 0,403** (`gnn_k0` → `mlp_flat`) : ce que récupèrent les **agrégats calculés à la main** le long du
  chemin (ρ max / moyen / somme, sommes M/M/1 et M/M/1/K, part WFQ minimale).
- **0,403 → 0,128** (`mlp_flat` → `gnn_gatv2`) : ce que récupère en plus le **passage de messages appris**, à
  information équivalente. **C'est ce chiffre-là qui répond à H1** : × 3,1.

Au passage, `gnn_k0` ≈ baseline confirme un constat de L1 (D1.5) : les propriétés propres d'un flux (son type, son
débit, sa longueur de chemin) ne prédisent presque pas son délai ; c'est la charge partagée qui décide.

**Contrôle du surapprentissage** (figure `L5_02_courbes_apprentissage.png`). Train et val descendent ensemble
jusqu'à l'époque ~36 ; ensuite val plafonne vers 0,075 de MSE pendant que train continue vers 0,05. L'écart existe
mais reste faible, et l'arrêt précoce tranche à l'époque 61 (76 époques au total, 141 min sur RTX 5070). Dropout 0,1
et weight decay 1e-5 étaient actifs.

**Limite assumée : pas de recherche d'hyperparamètres** (D3.8). Une exécution complète coûte 2,4 h ; la grille
prévue coûterait 30 à 60 h de GPU. On garde la configuration initiale et on la défend par les courbes
d'apprentissage, plutôt que d'annoncer un réglage fait sur un sous-échantillon dont rien ne garantit qu'il tienne à
plein budget.

**H2 — comportement sur les événements** (avec `data/processed/events.parquet` du volet B, τ = 1,05). MAE_log sur
test :

| Modèle | non-événement (34 632 flux) | événement (167 321 flux) | dégradation |
|---|---|---|---|
| `naive_median_slice` | 1,322 | 0,719 | −0,603 |
| `gnn_k0` | 1,244 | 0,699 | −0,545 |
| `analytic_mm1` (volet B) | 0,363 | 0,708 | +0,345 |
| `mlp_flat` | 0,290 | 0,427 | +0,136 |
| **`gnn_gatv2`** | **0,121** | **0,129** | **+0,008** |

La frontière n'est pas « modèle plat contre graphe », c'est **« voit la charge ou non »** :
- Les modèles qui voient la charge se dégradent sur les événements, comme le prédit H2 — mais le GNN
  **quasiment pas** (+0,008), alors que le modèle plat à information équivalente se dégrade **17 fois plus**
  (+0,136). Le volet comparatif de H2 est donc confirmé, et c'est l'argument le plus fort du GNN : il reste bon
  là où le réseau va mal, ce qui est exactement le régime qui intéresse un orchestrateur.
- Les modèles qui ne voient pas la charge (naïve, `gnn_k0`) font *mieux* sur les événements. Ce n'est pas une
  qualité : comme **82,9 % des flux du test sont en événement**, leur constante apprise sur train est calée sur
  cette majorité, et devient franchement fausse sur la minorité non-événement (1,322 pour la naïve).

**Au palier τ = 1,05, la première moitié de H2 (« tous les modèles se dégradent ») est donc fausse** — mais c'est
un artefact de proportion, pas une propriété des modèles. Le volet B propose un palier **sévère**
(`is_event_tau_2`, 16 % des flux), et là H2 tient en entier (MAE_log test, reste / sévère) :

| Modèle | reste | sévère (τ = 2,0) | dégradation |
|---|---|---|---|
| `naive_median_slice` | 0,770 | 1,098 | +0,328 |
| `gnn_k0` | 0,737 | 1,078 | +0,340 |
| `analytic_mm1` (volet B) | 0,585 | 0,983 | +0,398 |
| `mlp_flat` | 0,373 | 0,561 | +0,188 |
| **`gnn_gatv2`** | **0,124** | **0,145** | **+0,021** |

Presque tous les modèles se dégradent, y compris ceux qui « s'amélioraient » à τ = 1,05, et le classement de la
robustesse se confirme en plus net : le GNN se dégrade **9 fois moins** que le modèle plat à information
équivalente. C'est l'argument opérationnel le plus fort du volet A — le GNN reste bon dans le régime chargé, celui
qui décide une admission de slice. (Seul `analytic_mm1k_wfq` s'inverse encore, −0,631 : il surestime tellement les
délais qu'il « gagne » là où ils sont réellement énormes.)

Ce palier est calculé dans `notebooks/L5_gnn.py` et non dans `src/evaluate.py`, qui ne produit que
`event` / `non_event` : l'ajouter au fichier commun profiterait aux trois volets, mais c'est une décision d'équipe
(§4.1 du plan).

**H1 et H3, tranchées avec les intervalles de confiance du volet C** (bootstrap sur les scénarios, B = 1 000,
`results/bootstrap_ci.csv` et `bootstrap_diff.csv`). Depuis l'arrivée du volet C, le meilleur modèle plat n'est plus
`mlp_flat` mais **`lgbm`** :

| Modèle | MAE_log test [IC 95 %] |
|---|---|
| **`gnn_gatv2`** | **0,128 [0,118 ; 0,138]** |
| `lgbm` (volet C) | 0,352 [0,327 ; 0,379] |
| `mlp_flat` | 0,403 [0,377 ; 0,430] |

- **H1 confirmée contre le meilleur adversaire** : différence appariée `gnn_gatv2 − lgbm` = **−0,2243
  [−0,2484 ; −0,2023]**, le GNN devant dans **100 %** des rééchantillonnages. On annonce donc **× 2,75 contre
  `lgbm`**, pas × 3,1 contre `mlp_flat` : c'est plus conservateur et on se compare au plus fort.
- **H3 n'est pas confirmée.** À charge égale (repondération par classes de ρ max du test), l'écart
  « topologies inédites − test » du GNN est de **−0,0039 [−0,0189 ; +0,0119]** : l'intervalle contient 0. Le GNN
  **transfère sans perte** à des topologies jamais vues — ce qui est déjà le comportement attendu d'un modèle
  inductif — mais rien ne permet de dire qu'il y généralise *mieux* que les autres.

**Limite assumée : le test « topologies inédites » n'est pas plus dur, il est plus facile** (D3.7). Tous les modèles
y font mieux, **y compris la baseline naïve** (0,822 → 0,790) qui prédit pourtant une constante par slice et ne peut
pas « mieux généraliser ». L'écart vient de la charge : ρ max médian 1,35 contre 1,63 (D2.1). H3 ne se lit donc pas
sur cette colonne telle quelle ; il faut la comparaison à charge égale prévue en L6.

## Questions probables du jury

**Qu'est-ce que le « passage de messages », concrètement ?**
Chaque lien calcule un résumé des flux qu'il transporte (leur état caché, pondéré par la part WFQ du saut), et met à
jour son propre état avec. Puis chaque flux fait de même avec les liens de son chemin. Répété 3 fois, un flux
« apprend » l'état des liens de ses concurrents, donc la congestion qui se propage. Sans cela, il ne connaît que
lui-même.

**En quoi votre GNN n'est-il pas un simple MLP ?**
On l'a mesuré, c'est exactement l'ablation `gnn_k0` : le même réseau avec K = 0 se réduit à un MLP sur les variables
du flux et tombe à 0,792 (contre 0,128). Et face à un MLP qui reçoit les mêmes informations résumées à la main
(`mlp_flat`, 0,403), le GNN fait encore 3,1 fois mieux. La différence n'est donc pas la profondeur du réseau mais le
fait de calculer les agrégats **le long de la structure réelle** au lieu de les figer à l'avance.

**Pourquoi K = 3 et pas 5 ou 10 ?**
K est la distance d'information : K = 1 = mes liens, K = 2 = mes concurrents, K = 3 = les liens de mes concurrents.
Au-delà, le coût mémoire croît et le risque de lissage excessif (tous les nœuds convergent vers le même état)
augmente. On n'a pas pu tester plusieurs K à budget GPU constant (2,4 h par exécution) et on le dit plutôt que de
prétendre l'avoir réglé.

**Pourquoi mettre la part WFQ sur l'arête et pas sur le lien ?**
Parce qu'elle appartient au couple (flux, lien) : sur un même lien, deux flux de slices différentes ont des parts
garanties différentes ; et un même flux change de part d'un lien à l'autre. La mettre sur le nœud lien obligerait à
en prendre une moyenne, ce qui effacerait justement l'information qui distingue un flux affamé d'un flux servi.

**Pourquoi pas un GCN, qui est le GNN le plus connu ?**
GCN suppose un graphe homogène et pondère les voisins par leur degré uniquement : il ne sait pas distinguer un nœud
flux d'un nœud lien, ni lire un attribut d'arête. La part WFQ, qui détermine le débit garanti, n'aurait aucun moyen
d'entrer dans le calcul.

**Pourquoi `gnn_k0` s'arrête-t-il à 22 époques alors que `gnn_gatv2` en fait 76 ?**
Parce qu'il n'a plus rien à apprendre : dès la première époque il est à 0,798, et son meilleur score est 0,785
(époque 7) — 1,6 % de progrès sur toute l'exécution, puis du bruit entre 0,787 et 0,807. Avec seulement les
variables du flux, dont le débit et la taille de paquet sont constants par type de slice (D1.5), il atteint tout de
suite son plafond d'information. `gnn_gatv2` continue à trouver de nouveaux minima jusqu'à l'époque 61 parce qu'il
lui reste du signal à extraire. Cette convergence immédiate est donc un **argument** pour H1, pas un défaut de
réglage.

**Votre GNN est-il utilisable sur un réseau plus grand que ceux vus à l'entraînement ?**
Oui par construction : les poids sont partagés entre tous les nœuds et toutes les arêtes, donc rien ne dépend du
nombre de nœuds (c'est le caractère *inductif*). On l'a vérifié sur 24 topologies jamais vues (1 080 scénarios), où
l'erreur reste de 0,115 — en gardant à l'esprit que ce jeu est aussi moins chargé.

## Le code à connaître

### 1. `FlowLinkGNN.forward` (`src/models/gnn.py`) — les deux demi-pas

```python
h_f = self.enc_flow(data["flow"].x)          # état initial des flux   (MLP sur leurs variables)
h_l = self.enc_link(data["link"].x)          # état initial des liens
ei_fl, ei_lf = data[F2L].edge_index, data[L2F].edge_index   # sauts, dans les deux sens
e = data[F2L].edge_attr                      # attributs du saut (part WFQ, rang, ρ de la file…)
for k in range(self.layers):                 # K itérations
    m_l = self._conv(self.f2l[k], h_f, h_l, ei_fl, e, ...)  # 1. le lien agrège SES flux
    h_l = self.norm_l[k](h_l + self.drop(torch.relu(m_l)))  #    résidu + LayerNorm
    m_f = self._conv(self.l2f[k], h_l, h_f, ei_lf, e, None) # 2. le flux agrège SES liens (déjà à jour)
    h_f = self.norm_f[k](h_f + self.drop(torch.relu(m_f)))
out = self.readout(h_f).squeeze(-1)          # un délai prédit (log10) par nœud flux
```
L'ordre compte : on met d'abord les liens à jour, pour que les flux lisent ensuite un état de lien qui tient déjà
compte de la charge. `h_l + …` est le résidu : le nœud garde son information propre même après K mises à jour. Avec
`self.layers = 0` la boucle est vide et il ne reste que `readout(enc_flow(x))` : c'est `gnn_k0`.

### 2. `EdgeSAGE.message` (`src/models/gnn.py`) — pourquoi on n'a pas pris `SAGEConv`

```python
def message(self, x_j, edge_attr):                              # x_j = état du nœud émetteur
    return torch.relu(self.msg(torch.cat([x_j, edge_attr], dim=-1)))
```
Le message concatène l'état du voisin **et** les attributs du saut avant la transformation linéaire. `SAGEConv` de
PyTorch Geometric ne prend pas d'`edge_attr` : avec lui, la part WFQ n'atteindrait jamais le lien. L'agrégation est
une **moyenne** (`aggr="mean"`), donc un lien qui porte 200 flux et un qui en porte 3 produisent des messages à la
même échelle.

### 3. `build_graphs` (`src/graphs.py`) — découper les tableaux en un graphe par scénario

```python
fa, fb = offsets(f.scenario_id.to_numpy(), scenario_ids)   # lignes [début, fin) de chaque scénario
for k, sid in enumerate(scenario_ids):
    g = HeteroData()
    g["flow"].x = torch.tensor(xf[fa[k]:fb[k]])            # COPIE la tranche (voir ci-dessous)
    g["link"].x = torch.tensor(xl[la[k]:lb[k]])
    ei = torch.tensor(np.stack([h_flow[ha[k]:hb[k]], h_link[ha[k]:hb[k]]]))
    g["flow", "traverses", "link"].edge_index = ei
    g["link", "carries", "flow"].edge_index = ei.flip(0)   # même saut, sens inverse
```
Les trois tables (flux, liens, sauts) sont triées par `scenario_id`, donc les lignes d'un scénario sont
**contiguës** : `np.searchsorted` donne directement ses bornes, et on découpe sans refaire de `groupby`.
`ei.flip(0)` crée l'arête réciproque en échangeant les deux lignes de `edge_index`.

**Le piège coûteux (D3.2).** La première version utilisait `torch.from_numpy(tableau[i:j])`, qui crée une **vue**
sur tout le tableau et non une copie : `torch.save` écrivait donc le tableau entier pour chacun des 7 127 graphes,
soit un cache de 330 Go. Avec `torch.tensor(...)`, qui copie la tranche, le même cache fait 539 Mo.
