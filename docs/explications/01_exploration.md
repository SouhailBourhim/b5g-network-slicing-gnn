# 01 — Exploration des données (L1)

Tous les chiffres viennent de `results/eda_summary.json` (produit par `notebooks/L1_exploration.ipynb`).

## Ce qu'on a fait

On a lu les 79 archives du jeu de données et transformé chaque simulation (« scénario ») en quatre tableaux :
les scénarios, les liens du réseau, les flux de trafic, et les « sauts » (quel flux traverse quel lien, dans quelle
file d'attente). On a vérifié la qualité des données (valeurs manquantes, impossibles, doublons), puis calculé,
**uniquement à partir des entrées**, la charge de chaque lien (ρ = trafic offert / capacité) et la ressource
allouée à chaque slice par l'ordonnanceur WFQ. Enfin on a regardé comment le délai dépend de ces grandeurs.

Résultat : 7 127 scénarios valides, 162 topologies, 1,58 million de flux, 7,4 millions de sauts.

**Définitions utiles.**
- *Flux* : le trafic d'un équipement (un capteur mMTC, un terminal eMBB…) vers sa destination. C'est ce qu'on prédit.
- *ρ (rho)* : charge offerte / capacité d'un lien. ρ = 0,5 → le lien est occupé à moitié ; ρ > 1 → on lui envoie plus
  qu'il ne peut transmettre, l'excédent est mis en file puis perdu quand la file (32 paquets) est pleine.
- *WFQ (Weighted Fair Queuing)* : sur chaque port de routeur, chaque slice a sa propre file avec un poids ; une file
  de poids w reçoit au moins w / Σw de la capacité quand toutes les files sont actives. Le poids = la bande passante
  réservée à la slice. C'est **l'allocation de ressources** du problème.

## Pourquoi ces choix

| Choix | Alternative rejetée | Pourquoi |
|---|---|---|
| Notre propre parseur, **validé contre l'API officielle** (`tests/test_parsing.py`, 5 tests OK) | Utiliser `DatanetAPI` tel quel | L'API lit des matrices 400×400 (≈170 000 cases) alors que ~200 cases contiennent un flux : trop lent sur 7 000 scénarios. Elle décode aussi les fichiers avec `jsonpickle`, qui peut exécuter du code ; on utilise `json`. |
| Débit réel = paquets/s × taille moyenne réelle | Champ `bandwidth` du fichier slices | `bandwidth` sous-estime eMBB et mMTC de 25 % (rapport médian 1,25) ; notre débit colle au débit mesuré (rapport médian 1,00005). |
| Chemins lus dans `routing_K.txt` | Champ `path` du fichier slices | Seuls 58 % des flux ont le même chemin dans les deux fichiers. Avec `routing_K.txt`, ρ calculé reproduit l'utilisation mesurée (corrélation 0,984) ; c'est donc le routage réellement simulé. |
| Utilisation **mesurée** (`linkUsage.txt`) exclue des entrées | La donner au modèle | Elle sort de la même simulation que le délai : ce serait donner une partie de la réponse (fuite), et elle est inconnue avant simulation, donc inutilisable pour un « what-if ». Elle sert seulement à valider ρ. |
| Prédire **log10(délai)** | Délai brut | Le délai couvre 8 ordres de grandeur ; asymétrie 38 en brut, 0,99 en log. En brut, une erreur sur un flux à 1 000 s écraserait toutes les autres. |

## Ce que montrent les résultats

- **Inventaire** (figure `L1_01_inventaire.png`) : cœurs de 2 à 190 nœuds, 41 scénarios par topologie en médiane,
  221 flux par scénario en moyenne. 75 % des flux sont mMTC, 21 % URLLC, 4 % eMBB → on rapportera toujours les
  erreurs **par type de slice**.
- **Qualité** : aucune valeur impossible (pas de perte > 1, pas de délai négatif) sauf 9 flux à `delay = -1`,
  code du simulateur pour « aucun paquet reçu » (100 % de pertes) : exclus de la régression du délai.
- **Entrées constantes** : loi de Poisson, tailles binomiales, file de 32 paquets, débit fixe par type de slice.
  La variabilité du délai vient donc du **réseau** (topologie, routage, compétition, WFQ), pas du trafic individuel.
- **Validation de ρ** (figure `L1_02_validation_rho.png`) : les points sont sur ou sous la diagonale.
  Normal : le trafic perdu en amont n'arrive jamais en aval, donc l'utilisation mesurée ≤ charge offerte.
- **Cibles** (figure `L1_03_distributions_cibles.png`) : délai médian 0,5 ms (URLLC), 1,5 ms (mMTC), 13,7 ms (eMBB),
  mais un 99ᵉ centile entre 12 s et 484 s. Pertes moyennes 23–30 % : le réseau est souvent surchargé.
- **Délai vs ρ** (figure `L1_05_delai_vs_rho.png`) : **coude net à ρ ≈ 1**. Au-delà, le délai plafonne au lieu
  d'exploser : la file de 32 paquets borne l'attente, l'excédent devient de la perte (comportement M/M/1/K).
- **Délai vs part WFQ** (figure `L1_06_delai_vs_wfq.png`) : moins la file d'un flux est garantie, plus son délai
  est grand, pour les trois slices.
- **Flux affamés** : 2,9 % des flux dépassent 1 s. 45 744 des 46 194 flux > 1 s ont une file de poids quasi nul
  **sur un lien saturé** (ρ ≥ 1). Mais 77 158 flux dans la même situation restent sous 10 ms. Aucune règle simple
  ne sépare les deux cas → il faut un modèle qui voit **avec quels flux chaque file est en compétition**.
- **Corrélations** (figure `L1_07_correlations_cible.png`) : les meilleures variables sont les estimations de files
  d'attente (M/M/1/K sommé sur le chemin : Spearman 0,57) et le ρ maximal (0,53).

## Questions probables du jury

**Pourquoi ne pas utiliser l'API fournie avec le jeu de données ?**
On l'utilise comme référence : nos tests comparent nos valeurs aux siennes (délai, gigue, pertes, chemins, utilisation)
et tout concorde. On ne s'en sert pas pour tout lire parce qu'elle parcourt ~170 000 cases par scénario pour ~200 flux.

**Comment savez-vous que vos chemins sont les bons ?**
On a calculé l'utilisation de chaque lien avec chaque jeu de chemins et comparé à l'utilisation mesurée par le simulateur.
Avec `routing_K.txt` : corrélation 0,984, erreur moyenne 0,003. Avec les chemins du fichier slices, la corrélation
chute (0,66 sur certains scénarios). Le simulateur a donc utilisé `routing_K.txt`.

**Pourquoi le log du délai ?**
Parce que la distribution s'étale sur 8 ordres de grandeur (asymétrie 38). En log elle devient presque symétrique (0,99).
Physiquement, se tromper de 1 ms sur un flux à 0,1 ms est grave, sur un flux à 10 s c'est négligeable : c'est
l'erreur **relative** qui compte, et l'erreur en log est une erreur relative.

**Pourquoi URLLC, la slice « faible latence », a-t-elle des flux à plusieurs secondes ?**
Parce que dans ce jeu, quand le routage simulé s'écarte du chemin où la slice a réservé des ressources, ses paquets
tombent dans une file WFQ de poids ~0. Si le lien est saturé, cette file n'est presque jamais servie : délai énorme,
pertes ~99 %. C'est un problème d'**allocation**, exactement ce qu'un modèle « what-if » doit détecter.

**Pourquoi exclure l'utilisation mesurée alors qu'elle est très corrélée au délai ?**
Elle est produite par la simulation qu'on cherche à remplacer. En exploitation, avant de simuler, on ne l'a pas.
Notre ρ calculé depuis les entrées est disponible avant simulation et colle à la mesure (corrélation 0,984).

## Le code à connaître

### 1. `routing_path` (`src/parsing.py`) — reconstruire le chemin réellement simulé

```python
def routing_path(R, port_to_next, src, dst):
    path, node = [src], src
    while R[node][dst] != -1:                      # -1 = on est arrivé à dst
        node = port_to_next[node][R[node][dst]]    # port de sortie -> nœud voisin
        path.append(node)
        if len(path) > len(R):                     # garde-fou contre une boucle
            raise ValueError(...)
    return path
```
`R[node][dst]` = numéro du port que `node` utilise pour joindre `dst` (routage par destination, comme une table IP).
`port_to_next[node][port]` = le voisin au bout de ce port (lu sur les arêtes du graphe). On saute de port en port.

### 2. `parse_traffic_cell` (`src/parsing.py`) — le vrai débit offert

```python
out["mean_pkt_size"] = (out["pkt_size_1"] + out["pkt_size_2"]) / 2   # loi binomiale p = 0,5
out["rate_bps"] = out["avg_pkts_lambda"] * out["mean_pkt_size"]       # paquets/s × bits/paquet
```
Les paquets eMBB font 12 000 ou 18 000 bits avec probabilité ½ : moyenne 15 000 bits. À 1 500 paquets/s, cela
donne 22,5 Mb/s, pas les 18 Mb/s annoncés. Vérifié : débit mesuré / débit calculé = 1,00005 en médiane.

### 3. `hop_table` (`src/features.py`) — charges par lien et par file

```python
h["load_link"]  = g_link["rate_bps"].transform("sum")       # tous les flux du lien
h["rho_link"]   = h["load_link"] / h["capacity_bps"]        # vue FIFO
h["load_queue"] = g_queue["rate_bps"].transform("sum")      # flux de la même file WFQ
h["rho_queue"]  = h["load_queue"] / (h["wfq_share"] * h["capacity_bps"])  # vue débit garanti
```
`groupby(...).transform("sum")` additionne les débits de tous les flux qui partagent le lien (ou la file) et recopie
le total sur chaque ligne. ρ_lien compare au lien entier ; ρ_file compare au débit **garanti** à la file
(part WFQ × capacité). Un ρ_file de 1 000 signifie « la file reçoit 1 000 fois plus que ce qu'on lui garantit ».
