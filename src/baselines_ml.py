"""Tabular baselines (workstream C): Ridge and LightGBM on the flat per-flow features.

Target: y = log10(delay [s]).  Features: ALL_FEATURES (inputs only, see src/features.py).
Protocol (see JOURNAL D2.1-D2.3 and the team plan, section 4.2):
  * split = frozen load_splits(); one row per flow with a defined delay (has_delay)
  * everything that is learned (imputation values, log rules, scaler, ridge alpha,
    number of LightGBM trees) is fitted on TRAIN and chosen on VAL
  * the TEST split is never used to choose anything: this script prints train and val
    metrics only; the test numbers come from `python -m src.evaluate`, looked at once.

Usage (from the repository root):
  python -m src.baselines_ml --model ridge
  python -m src.baselines_ml --model lgbm --seed 0
  python -m src.baselines_ml --model lgbm --seed 0 --drop-group wfq     # ablation (C3)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import PROCESSED_DIR, RESULTS_DIR, SEED
from src.contracts import save_predictions
from src.features import ALL_FEATURES, FEATURE_GROUPS
from src.metrics import regression_metrics
from src.splits import fit_scaler_on_train, load_splits

PRED_SPLITS = ["val", "test", "test_unseen_topo"]      # what save_predictions must contain
RIDGE_ALPHAS = np.logspace(-3, 4, 15)

LGBM_PARAMS = dict(objective="regression",             # squared error, consistent with MSE_log
                   metric="l1",                         # early stopping on val MAE_log
                   learning_rate=0.05, num_leaves=63, min_child_samples=100,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   reg_lambda=1.0, n_jobs=-1, verbose=-1)
LGBM_MAX_TREES = 3000
LGBM_PATIENCE = 100


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(features: list[str]) -> pd.DataFrame:
    """One row per flow with a defined delay, with its split."""
    path = PROCESSED_DIR / "flow_features.parquet"
    available = set(pq.ParquetFile(path).schema.names)
    missing = [c for c in features if c not in available]
    if missing:
        raise KeyError(f"features absent de flow_features.parquet : {missing}")
    f = pd.read_parquet(path, columns=["scenario_id", "flow_id", "has_delay", "log_delay"] + features)
    f = f[f.has_delay].drop(columns="has_delay")
    f = f.merge(load_splits()[["scenario_id", "split"]], on="scenario_id", how="inner")
    assert f.log_delay.notna().all(), "NaN dans la cible alors que has_delay est vrai"
    assert not f.duplicated(["scenario_id", "flow_id"]).any(), "doublons (scenario_id, flow_id)"
    print("flux par split :", f.split.value_counts().to_dict())
    return f.reset_index(drop=True)


def report(df: pd.DataFrame, pred: np.ndarray, name: str) -> None:
    """Train and VAL metrics only (the test split is evaluated once, by src.evaluate)."""
    for s in ["train", "val"]:
        m = (df.split == s).to_numpy()
        r = regression_metrics(df.log_delay[m], pred[m])
        print(f"  {name:8s} {s:6s} MAE_log={r['mae_log']:.4f}  MSE_log={r['mse_log']:.4f}  n={r['n']}")


def write_predictions(df: pd.DataFrame, pred: np.ndarray, model: str, seed: int) -> None:
    out = df.assign(pred_log10_delay=pred)
    out = out[out.split.isin(PRED_SPLITS)]
    print("written:", save_predictions(out, model, seed=seed))


# --------------------------------------------------------------------------
# Ridge
# --------------------------------------------------------------------------
def fit_rules(X_train: pd.DataFrame) -> dict[str, str]:
    """How to tame heavy-tailed features, decided on TRAIN only.
    log10 : strictly positive and max/min > 100 (rates, capacities, queueing terms...)
    log1p : non-negative counts with max > 100
    raw   : everything else (binary indicators, delta, ratios of moderate range)."""
    rules = {}
    for c in X_train.columns:
        s = X_train[c]
        if s.nunique() <= 2:
            rules[c] = "raw"
        elif s.min() > 0 and s.max() / s.min() > 100:
            rules[c] = "log10"
        elif s.min() >= 0 and s.max() > 100:
            rules[c] = "log1p"
        else:
            rules[c] = "raw"
    return rules


def apply_rules(X: pd.DataFrame, rules: dict[str, str]) -> pd.DataFrame:
    out = X.copy()
    for c, r in rules.items():
        if r == "log10":
            out[c] = np.log10(out[c].clip(lower=1e-12))     # clip: val/test may go below the train min
        elif r == "log1p":
            out[c] = np.log1p(out[c].clip(lower=0))
    return out


def run_ridge(df: pd.DataFrame, feats: list[str]) -> None:
    from sklearn.linear_model import Ridge
    is_tr = (df.split == "train").to_numpy()
    is_va = (df.split == "val").to_numpy()
    y = df.log_delay.to_numpy()

    X = df[feats].replace([np.inf, -np.inf], np.nan)
    n_nan = int(X.isna().sum().sum())
    X = X.fillna(X[is_tr].median())                          # imputation value fitted on train
    rules = fit_rules(X[is_tr])
    print(f"NaN/inf imputés : {n_nan} | règles :", pd.Series(rules).value_counts().to_dict())
    X = apply_rules(X, rules)
    Z = fit_scaler_on_train(X, is_tr).transform(X)           # scaler fitted on train only

    grid = []
    for a in RIDGE_ALPHAS:
        p = Ridge(alpha=a).fit(Z[is_tr], y[is_tr]).predict(Z[is_va])
        grid.append({"alpha": a, "val_mae_log": float(np.mean(np.abs(p - y[is_va])))})
    grid = pd.DataFrame(grid)
    RESULTS_DIR.mkdir(exist_ok=True)
    grid.to_csv(RESULTS_DIR / "ridge_alpha_grid.csv", index=False)  # every config tried, not only the best
    best = float(grid.loc[grid.val_mae_log.idxmin(), "alpha"])
    print(f"alpha choisi sur val : {best:.4g}  (grille complète : results/ridge_alpha_grid.csv)")

    model = Ridge(alpha=best).fit(Z[is_tr], y[is_tr])
    pred = model.predict(Z)
    report(df, pred, "ridge")
    coefs = pd.Series(model.coef_, index=feats).sort_values(key=np.abs, ascending=False)
    print("coefficients standardisés les plus grands :\n", coefs.head(8).round(3).to_string())
    write_predictions(df, pred, "ridge", seed=0)             # deterministic: a single seed


# --------------------------------------------------------------------------
# LightGBM
# --------------------------------------------------------------------------
def run_lgbm(df: pd.DataFrame, feats: list[str], seed: int, model_name: str) -> None:
    import lightgbm as lgb
    is_tr = (df.split == "train").to_numpy()
    is_va = (df.split == "val").to_numpy()
    y = df.log_delay.to_numpy()
    X = df[feats].replace([np.inf, -np.inf], np.nan).astype(np.float32)   # LightGBM handles NaN natively

    dtr = lgb.Dataset(X[is_tr], y[is_tr])
    dva = lgb.Dataset(X[is_va], y[is_va], reference=dtr)
    curve = {}
    t0 = time.time()
    booster = lgb.train({**LGBM_PARAMS, "seed": SEED + seed}, dtr, num_boost_round=LGBM_MAX_TREES,
                        valid_sets=[dtr, dva], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(LGBM_PATIENCE, verbose=False),
                                   lgb.record_evaluation(curve)])
    print(f"arbre(s) retenus : {booster.best_iteration} (arrêt précoce sur val) | {time.time() - t0:.0f}s")

    RESULTS_DIR.mkdir(exist_ok=True)                         # learning curve = overfitting control
    pd.DataFrame({"iteration": np.arange(1, len(curve["val"]["l1"]) + 1),
                  "train_mae_log": curve["train"]["l1"],
                  "val_mae_log": curve["val"]["l1"]}).to_csv(
        RESULTS_DIR / f"{model_name}_curve_seed{seed}.csv", index=False)

    pred = booster.predict(X, num_iteration=booster.best_iteration)
    report(df, pred, model_name)
    imp = pd.Series(booster.feature_importance("gain"), index=feats)
    print("importance (gain, % du total) :\n", (100 * imp / imp.sum()).sort_values(ascending=False).head(8).round(1).to_string())
    write_predictions(df, pred, model_name, seed=seed)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["ridge", "lgbm"], required=True)
    ap.add_argument("--seed", type=int, default=0, help="repetition index k (lgbm only)")
    ap.add_argument("--drop-group", choices=list(FEATURE_GROUPS), default=None,
                    help="ablation: remove one feature group (lgbm only) -> model lgbm_no_<group>")
    args = ap.parse_args()

    feats = ALL_FEATURES
    name = args.model
    if args.drop_group:
        if args.model != "lgbm":
            ap.error("--drop-group est prévu pour lgbm uniquement")
        feats = [c for g, cols in FEATURE_GROUPS.items() if g != args.drop_group for c in cols]
        name = f"lgbm_no_{args.drop_group}"
    print(f"modèle {name} | {len(feats)} features | seed {args.seed}")

    df = load_data(feats)
    if args.model == "ridge":
        run_ridge(df, feats)
    else:
        run_lgbm(df, feats, args.seed, name)


if __name__ == "__main__":
    main()
