# AI-Based Dynamic Resource Allocation for B5G Network Slicing

Projet INPT (Smart-ICT). Modèle « what-if » de QoS par flux (délai) pour des scénarios de network slicing,
jeu de données Farreras et al., *Data in Brief* 55 (2024) 110738, Zenodo 10.5281/zenodo.10610616 (v2.2.1, CC BY 4.0).

## Installation (Windows, Git Bash ou PowerShell)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
```

## Pipeline

| Étape | Commande | Durée (CPU) |
|---|---|---|
| Données brutes | `data/raw/` : zip (MD5 `2a4a9a47bdeaa0c02304c40115c56273`), dossier extrait, `datanetAPI.py`, `features.pdf` | — |
| Parsing → parquet | `.venv/Scripts/python -m src.parsing --workers 12` | ~10–15 min |
| L1 Exploration | `.venv/Scripts/python -m src.nb notebooks/L1_exploration.py` | ~2 min |
| Tests | `.venv/Scripts/python -m pytest -q` | < 1 min |

## Organisation

```
data/raw/            fichiers Zenodo (intacts)
data/processed/      tables parquet (scenarios, links, flows, flow_links, *_features) + graphes PyG
src/                 parsing, features, eda, splits, modèles, métriques
notebooks/           L1..L5 (source .py « percent » + .ipynb exécuté)
reports/figures/     figures du rapport
reports/rapport_final.md
docs/explications/   explications pas à pas (français)
results/             métriques CSV/JSON
JOURNAL.md           journal des décisions
```
