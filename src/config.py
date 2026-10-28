"""Central paths and constants, so every script agrees on where things live."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "slicing-simulations" / "2.2.1-10kprocessed"
PROCESSED_DIR = ROOT / "data" / "processed"
PARTS_DIR = PROCESSED_DIR / "parts"          # one parquet per tarball and per table
GRAPHS_DIR = PROCESSED_DIR / "graphs"        # cached PyG HeteroData
FIG_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints"

SEED = 42
SLICE_TYPES = ["eMBB", "URLLC", "mMTC"]      # fixed order for one-hot encodings
