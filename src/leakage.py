"""Leakage experiment (L2): why the split unit must be the scenario, not the flow.

Same data, same model (LightGBM, fixed untuned parameters), same features. Only the way
the train/test split is drawn differs:

  A  random split of FLOWS            (flows of one scenario land on both sides)
  B  split of SCENARIOS, whole groups (no scenario on both sides)

For each scheme we measure
  estimated error : error on its own held-out part  (what a student would report)
  true error      : error on E = scenarios that NEITHER scheme has seen (what happens in use)
A scheme that leaks reports an estimate better than the true error ("optimistic gap").

Usage: python -m src.leakage              -> results/leakage_experiment.csv
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR
from src.features import ALL_FEATURES
from src.metrics import regression_metrics
from src.splits import load_splits

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.1, num_leaves=63, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=-1, verbose=-1)


def load_pool(n_scenarios: int, seed: int) -> pd.DataFrame:
    """Flows of a random sample of scenarios, excluding the unseen-topology test set."""
    splits = load_splits()
    pool = splits[splits.split != "test_unseen_topo"].scenario_id.to_numpy()
    chosen = np.random.default_rng(seed).choice(pool, size=min(n_scenarios, len(pool)), replace=False)
    f = pd.read_parquet(PROCESSED_DIR / "flow_features.parquet",
                        columns=["scenario_id", "flow_id", "slice_type", "log_delay", "has_delay"] + ALL_FEATURES)
    return f[f.scenario_id.isin(chosen) & f.has_delay].reset_index(drop=True)


def fit_eval(train: pd.DataFrame, tests: dict[str, pd.DataFrame], seed: int) -> dict[str, dict]:
    model = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
    model.fit(train[ALL_FEATURES], train.log_delay)
    out = {}
    for name, part in tests.items():
        out[name] = regression_metrics(part.log_delay, model.predict(part[ALL_FEATURES]))
    return out


def run(n_scenarios: int = 1500, seeds=(0, 1, 2), frac_new: float = 0.20) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        t0 = time.time()
        df = load_pool(n_scenarios, seed)
        rng = np.random.default_rng(1000 + seed)
        scen = rng.permutation(df.scenario_id.unique())
        n_new = int(frac_new * len(scen))
        E = df[df.scenario_id.isin(scen[:n_new])]                  # truly new scenarios
        rest = df[df.scenario_id.isin(scen[n_new:])]               # what A and B may use
        # A: random flow-level split 80/20
        is_test_a = rng.random(len(rest)) < 0.2
        A_train, A_test = rest[~is_test_a], rest[is_test_a]
        # B: scenario-level split 80/20 (same expected sizes)
        rest_scen = rng.permutation(rest.scenario_id.unique())
        test_scen = set(rest_scen[: int(0.2 * len(rest_scen))])
        in_b = rest.scenario_id.isin(test_scen)
        B_train, B_test = rest[~in_b], rest[in_b]
        # the leak, measured directly: share of test flows whose scenario also appears in train
        leak_a = A_test.scenario_id.isin(A_train.scenario_id).mean()
        leak_b = B_test.scenario_id.isin(B_train.scenario_id).mean()
        for scheme, tr, te, leak in (("A_flow_split", A_train, A_test, leak_a),
                                     ("B_scenario_split", B_train, B_test, leak_b)):
            res = fit_eval(tr, {"estimated": te, "true": E}, seed)
            for kind, m in res.items():
                rows.append({"scheme": scheme, "seed": seed, "kind": kind, "n_train_flows": len(tr),
                             "n_train_scenarios": tr.scenario_id.nunique(), "frac_test_flows_in_seen_scenario": float(leak),
                             **m})
        print(f"seed {seed}: pool {len(df):,} flows / {df.scenario_id.nunique()} scenarios, "
              f"E = {len(E):,} flows, done in {time.time() - t0:.0f}s")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "leakage_experiment.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenarios", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    res = run(a.n_scenarios, tuple(a.seeds))
    piv = res.pivot_table(index=["scheme", "kind"], values=["mae_log", "mse_log"], aggfunc=["mean", "std"]).round(4)
    print(piv.to_string())


if __name__ == "__main__":
    main()
