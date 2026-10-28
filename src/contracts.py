"""File formats shared by the three workstreams (do not change without telling the team).

PREDICTIONS  results/preds/<model>__seed<k>.parquet
    scenario_id:int, flow_id:int, pred_log10_delay:float
    one row per flow with a defined delay (has_delay), for ALL splits the model can score
    (at least val, test, test_unseen_topo). Written with save_predictions().

EVENTS       data/processed/events.parquet          (owner: workstream "réseau / événements")
    scenario_id:int, flow_id:int, is_event:bool, event_type:str ('none'|'congestion'|'starvation'|'both'),
    plus one bool column per sensitivity threshold, e.g. is_event_tau_0_9, is_event_tau_1_0, ...
    Defined from INPUTS only (never from delay/loss). Written with save_events().
"""
from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR

PREDS_DIR = RESULTS_DIR / "preds"
EVENTS_PATH = PROCESSED_DIR / "events.parquet"
EVENT_TYPES = {"none", "congestion", "starvation", "both"}


def save_predictions(df: pd.DataFrame, model: str, seed: int) -> str:
    """Validate and write one model's predictions."""
    need = {"scenario_id", "flow_id", "pred_log10_delay"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"predictions miss columns {missing}")
    # compact dtypes: prediction files are committed to git (~3 MB per model and seed)
    out = df[["scenario_id", "flow_id", "pred_log10_delay"]].astype(
        {"scenario_id": "int32", "flow_id": "int32", "pred_log10_delay": "float32"})
    if out.duplicated(["scenario_id", "flow_id"]).any():
        raise ValueError("duplicate (scenario_id, flow_id) in predictions")
    if not out.pred_log10_delay.notna().all():
        raise ValueError("NaN predictions")
    if "__" in model:
        raise ValueError("model name must not contain '__'")
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDS_DIR / f"{model}__seed{seed}.parquet"
    out.to_parquet(path, index=False)
    return str(path)


def load_predictions() -> pd.DataFrame:
    """All prediction files, with model and seed columns."""
    parts = []
    for p in sorted(PREDS_DIR.glob("*__seed*.parquet")):
        model, seed = p.stem.split("__seed")
        parts.append(pd.read_parquet(p).assign(model=model, seed=int(seed)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def save_events(df: pd.DataFrame) -> str:
    need = {"scenario_id", "flow_id", "is_event", "event_type"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"events miss columns {missing}")
    if not set(df.event_type.unique()) <= EVENT_TYPES:
        raise ValueError(f"event_type must be in {EVENT_TYPES}")
    if df.duplicated(["scenario_id", "flow_id"]).any():
        raise ValueError("duplicate (scenario_id, flow_id) in events")
    df.to_parquet(EVENTS_PATH, index=False)
    return str(EVENTS_PATH)


def load_events() -> pd.DataFrame | None:
    return pd.read_parquet(EVENTS_PATH) if EVENTS_PATH.exists() else None
