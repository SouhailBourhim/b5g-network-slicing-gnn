"""Shared figure style: one color per slice type, thin marks, recessive grid."""
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.config import FIG_DIR

# Categorical slots 1-3 of the validated reference palette (all-pairs CVD-safe
# for 3 series). Color follows the entity: a slice type keeps its color in
# every figure of the report.
SLICE_COLORS = {"eMBB": "#2a78d6", "URLLC": "#eb6834", "mMTC": "#1baf7a"}
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#52514e"
SURFACE = "#fcfcfb"


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#b5b4ae", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 10,
        "axes.labelcolor": "#2b2b2a", "xtick.color": NEUTRAL, "ytick.color": NEUTRAL,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "font.family": "DejaVu Sans",
    })


def savefig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path
