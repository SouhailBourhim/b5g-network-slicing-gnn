"""A4 control: flat MLP on ALL_FEATURES, same inputs as the flat baselines, no graph.

Isolates the contribution of message passing: same training loop as train_gnn.py,
but the model only sees each flow's own feature vector (no flow/link graph).

  python -m src.train_mlp --seed 0                 # full training
  python -m src.train_mlp --max-scenarios 50 --epochs 2   # smoke test
  python -m src.train_mlp --seed 0 --device cpu    # force CPU (e.g. GPU busy with the GNN run)

Loss: MSE on log10(delay). Model selection: lowest val MAE_log (early stopping).
Outputs mirror train_gnn.py: checkpoints/mlp_flat__seed<k>.pt, results/train_logs/,
logs/, results/preds/mlp_flat__seed<k>.parquet, results/hpsearch_mlp.csv.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import CKPT_DIR, LOGS_DIR, PROCESSED_DIR, RESULTS_DIR
from src.contracts import save_predictions
from src.features import ALL_FEATURES
from src.metrics import regression_metrics
from src.models.gnn import mlp
from src.splits import fit_scaler_on_train, load_splits
from src.train_gnn import set_seed


def load_flows() -> pd.DataFrame:
    f = pd.read_parquet(PROCESSED_DIR / "flow_features.parquet",
                        columns=["scenario_id", "flow_id", "has_delay", "log_delay"] + ALL_FEATURES)
    f = f[f.has_delay].merge(load_splits()[["scenario_id", "split"]], on="scenario_id")
    assert not f[ALL_FEATURES].isna().any().any(), "NaN features in flow_features.parquet"
    return f


@torch.no_grad()
def predict(model, X: torch.Tensor, device, chunk: int = 200_000) -> np.ndarray:
    model.eval()
    out = [model(X[i:i + chunk].to(device)).cpu().numpy() for i in range(0, len(X), chunk)]
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5, help="weight decay (AdamW)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15, help="early stopping on val MAE_log")
    ap.add_argument("--batch-size", type=int, default=8192, help="flows per batch")
    ap.add_argument("--max-scenarios", type=int, default=None, help="subsample train/val (smoke tests)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--no-preds", action="store_true", help="do not predict the test sets")
    a = ap.parse_args()

    set_seed(a.seed)
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    run = f"mlp_flat__seed{a.seed}"
    for d in (CKPT_DIR, LOGS_DIR, RESULTS_DIR / "train_logs"):
        d.mkdir(parents=True, exist_ok=True)
    log = open(LOGS_DIR / f"{run}.log", "w", encoding="utf-8", buffering=1)

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    f = load_flows()
    if a.max_scenarios:
        keep = {s: set(sorted(g.scenario_id.unique())[:a.max_scenarios if s == "train" else max(10, a.max_scenarios // 4)])
                for s, g in f.groupby("split")}
        f = f[[sid in keep.get(sp, ()) for sid, sp in zip(f.scenario_id, f.split)]]

    scaler = fit_scaler_on_train(f[ALL_FEATURES], (f.split == "train").to_numpy())
    Xall = torch.tensor(scaler.transform(f[ALL_FEATURES]), dtype=torch.float32)
    yall = torch.tensor(f.log_delay.to_numpy(), dtype=torch.float32)
    splits = {s: f.split.to_numpy() == s for s in ("train", "val", "test", "test_unseen_topo")}

    model = mlp(len(ALL_FEATURES), a.hidden, 1, a.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    say(f"run {run} | device {device} | "
        f"{splits['train'].sum():,} train / {splits['val'].sum():,} val flows | {n_params:,} parameters")
    say(json.dumps(vars(a)))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    train_ds = TensorDataset(Xall[splits["train"]], yall[splits["train"]])
    loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True)
    Xval, yval = Xall[splits["val"]], yall[splits["val"]].numpy()
    best, best_epoch, history = float("inf"), -1, []
    ckpt = CKPT_DIR / f"{run}.pt"

    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        train_mse = tot / n
        vm = regression_metrics(yval, predict(model, Xval, device).squeeze(-1))
        sched.step(vm["mae_log"])
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_mse_log": train_mse, "val_mae_log": vm["mae_log"],
                        "val_mse_log": vm["mse_log"], "lr": lr, "seconds": time.time() - t0})
        improved = vm["mae_log"] < best - 1e-4
        if improved:
            best, best_epoch = vm["mae_log"], epoch
            torch.save({"state_dict": model.state_dict(), "args": vars(a), "epoch": epoch, "val_mae_log": best,
                        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}, ckpt)
        say(f"epoch {epoch:3d} | train MSE {train_mse:.4f} | val MAE {vm['mae_log']:.4f} MSE {vm['mse_log']:.4f} "
            f"| lr {lr:.1e} | {time.time() - t0:.1f}s{'  *' if improved else ''}")
        pd.DataFrame(history).to_csv(RESULTS_DIR / "train_logs" / f"{run}.csv", index=False)
        if epoch - best_epoch >= a.patience:
            say(f"early stop: no val improvement for {a.patience} epochs")
            break

    model.load_state_dict(torch.load(ckpt, weights_only=False)["state_dict"])
    say(f"best epoch {best_epoch}, val MAE_log {best:.4f}")
    summary = {"run": run, "model": "mlp_flat", "seed": a.seed, "hidden": a.hidden, "dropout": a.dropout,
               "lr": a.lr, "wd": a.wd, "batch_size": a.batch_size, "max_scenarios": a.max_scenarios,
               "best_epoch": best_epoch, "val_mae_log": best, "n_params": n_params,
               "minutes": sum(h["seconds"] for h in history) / 60}
    if not a.no_preds:
        for s in ("val", "test", "test_unseen_topo"):
            f.loc[splits[s], "pred_log10_delay"] = predict(model, Xall[splits[s]], device).squeeze(-1)
        preds = f[splits["val"] | splits["test"] | splits["test_unseen_topo"]]
        say("predictions -> " + save_predictions(preds, "mlp_flat", a.seed))
    hp = RESULTS_DIR / "hpsearch_mlp.csv"
    pd.DataFrame([summary]).to_csv(hp, mode="a", header=not hp.exists(), index=False)
    say(json.dumps(summary))


if __name__ == "__main__":
    main()
