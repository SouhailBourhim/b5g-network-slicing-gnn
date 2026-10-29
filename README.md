# AI-Based Dynamic Resource Allocation for B5G Network Slicing

Projet INPT (Smart-ICT). Modèle « what-if » de QoS par flux (délai) pour des scénarios de network slicing,
jeu de données Farreras et al., *Data in Brief* 55 (2024) 110738, Zenodo 10.5281/zenodo.10610616 (v2.2.1, CC BY 4.0).

Répartition du travail : `docs/plan_de_travail_equipe.docx`.

## Installation (Windows, PowerShell)

```powershell
git clone https://github.com/SouhailBourhim/b5g-network-slicing-gnn.git
cd b5g-network-slicing-gnn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# torch d'abord : GPU NVIDIA
pip install "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130
#            ou CPU seulement
# pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Données : tout le monde travaille sur les mêmes fichiers

Les tables traitées (~270 Mo) sont publiées dans la release GitHub `data-v1` (dépôt public) avec leurs sommes SHA-256. Les données d'origine sont sous licence CC BY 4.0 : citer Farreras et al., *Data in Brief* 55 (2024) 110738.

```powershell
python -m src.data_sync fetch        # télécharge (HTTPS, sans compte) + vérifie dans data/processed/
python -m src.data_sync verify       # revérifie à tout moment
python -m pytest -q tests/test_splits.py tests/test_metrics.py   # le test du parseur exige les données brutes
```

Les données brutes (zip Zenodo 271 Mo, MD5 `2a4a9a47bdeaa0c02304c40115c56273`) ne sont nécessaires que pour
re-parser : les extraire dans `data/raw/slicing-simulations/`, puis `python -m src.parsing --workers 8`.

**Ne jamais modifier `data/processed/` à la main.** Si une table change, on publie `data-v2` et on prévient l'équipe.

## Pipeline

| Étape | Commande | Durée |
|---|---|---|
| Parsing → parquet | `python -m src.parsing --workers 8` | ~10 min |
| L1 Exploration | `python -m src.nb notebooks/L1_exploration.py` | ~2 min |
| Splits (gelés) | `python -m src.splits` | < 1 min |
| Expérience de fuite (L2) | `python -m src.leakage` → `results/leakage_experiment.csv` | ~10 s |
| Baseline naïve | `python -m src.baselines_naive` | < 1 min |
| Graphes PyG (L5) | `python -m src.graphs` → `data/processed/graphs/` (539 Mo) | ~10 s |
| GNN (volet A, **GPU**) | `python -m src.train_gnn --model gnn_gatv2 --seed 0` | ~2,4 h |
| Ablation K = 0 | `python -m src.train_gnn --model gnn_k0 --seed 0` | < 1 min |
| MLP plat (contrôle) | `python -m src.train_mlp --seed 0` (`--device cpu` si le GPU est pris) | ~11 min |
| L5 GNN | `python -m src.nb notebooks/L5_gnn.py` | ~1 min |
| Tableau de résultats | `python -m src.evaluate` → `results/metrics_all.csv` | < 1 min |

## Contrats entre les parties (`src/contracts.py`)

- **Prédictions** : `results/preds/<modele>__seed<k>.parquet` (`scenario_id`, `flow_id`, `pred_log10_delay`),
  écrites avec `save_predictions()`, pour val, test et test_unseen_topo.
- **Événements** : `data/processed/events.parquet` (`scenario_id`, `flow_id`, `is_event`, `event_type`, …),
  écrits avec `save_events()`, définis depuis les **entrées** uniquement.
- **Splits** : `load_splits()` (`src/splits.py`). Ne jamais refaire son propre split.
- **Métriques** : `src/metrics.py` (MAE/MSE/RMSE sur log10(délai), MdAPE). Ne pas recoder.

## Organisation

```
data/raw/            fichiers Zenodo (non versionnés)
data/processed/      tables parquet (release data-vN) + graphes PyG
src/                 parsing, features, splits, contracts, metrics, evaluate, modèles
notebooks/           L1..L5 (source .py « percent » + .ipynb exécuté)
reports/figures/     figures du rapport
results/             métriques CSV/JSON, prédictions
docs/explications/   explications pas à pas (français)
JOURNAL.md           journal des décisions
```
