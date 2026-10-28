"""Train the flow-link GNN (volet A). Runs on the GPU when available.

  python -m src.train_gnn --model gnn_gatv2 --seed 0                 # full training
  python -m src.train_gnn --model gnn_sage --seed 0
  python -m src.train_gnn --model gnn_k0 --seed 0                    # K = 0 ablation
  python -m src.train_gnn --model gnn_gatv2 --max-scenarios 50 --epochs 2   # smoke test

Loss: MSE on log10(delay) over flows with a defined delay. Model selection: the epoch
with the lowest MAE_log on VAL (early stopping). TEST sets are only predicted at the end,
with the selected checkpoint, and written with save_predictions() for src/evaluate.py.

Outputs (run name = <model>__seed<k>, or <model>__<tag>__seed<k> with --tag)
  checkpoints/<run>.pt                 best weights + config
  results/train_logs/<run>.csv         one row per epoch (train loss, val MAE/MSE, lr, time)
  logs/<run>.log                       same, human readable
  results/preds/<model>__seed<k>.parquet   (unless --no-preds, e.g. during the HP search)
  results/hpsearch_gnn.csv             one summary row appended per run
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.config import CKPT_DIR, LOGS_DIR, RESULTS_DIR
from src.contracts import save_predictions
from src.graphs import load_graphs
from src.metrics import regression_metrics
from src.models.gnn import FlowLinkGNN

MODELS = {"gnn_gatv2": ("gatv2", None), "gnn_sage": ("sage", None), "gnn_k0": ("sage", 0)}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, graphs, device, batch_size: int) -> pd.DataFrame:
    """Predictions for every flow of the given graphs (one row per flow)."""
    model.eval()
    out = []
    for batch in DataLoader(graphs, batch_size=batch_size):
        batch = batch.to(device)
        pred = model(batch)
        sid = batch.scenario_id[batch["flow"].batch]          # scenario of each flow node
        out.append(pd.DataFrame({"scenario_id": sid.cpu().numpy(),
                                 "flow_id": batch["flow"].flow_id.cpu().numpy(),
                                 "pred_log10_delay": pred.float().cpu().numpy(),
                                 "y": batch["flow"].y.cpu().numpy(),
                                 "y_mask": batch["flow"].y_mask.cpu().numpy()}))
    return pd.concat(out, ignore_index=True)


def evaluate(model, graphs, device, batch_size) -> dict:
    p = predict(model, graphs, device, batch_size)
    p = p[p.y_mask]
    return regression_metrics(p.y, p.pred_log10_delay)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="gnn_gatv2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3, help="K message-passing iterations (ignored by gnn_k0)")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5, help="weight decay (AdamW)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15, help="early stopping on val MAE_log")
    ap.add_argument("--batch-size", type=int, default=32, help="graphs (scenarios) per batch")
    ap.add_argument("--max-scenarios", type=int, default=None, help="subsample train/val (smoke tests)")
    ap.add_argument("--tag", default="", help="extra name part, e.g. for the HP search")
    ap.add_argument("--no-preds", action="store_true", help="do not predict the test sets")
    a = ap.parse_args()

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv, forced_layers = MODELS[a.model]
    layers = a.layers if forced_layers is None else forced_layers
    run = f"{a.model}__{a.tag}__seed{a.seed}" if a.tag else f"{a.model}__seed{a.seed}"
    for d in (CKPT_DIR, LOGS_DIR, RESULTS_DIR / "train_logs"):
        d.mkdir(parents=True, exist_ok=True)
    log = open(LOGS_DIR / f"{run}.log", "w", encoding="utf-8", buffering=1)

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    train, val = load_graphs("train"), load_graphs("val")
    if a.max_scenarios:
        train, val = train[: a.max_scenarios], val[: max(10, a.max_scenarios // 4)]
    g0 = train[0]
    model = FlowLinkGNN(g0["flow"].x.size(1), g0["link"].x.size(1), g0["flow", "traverses", "link"].edge_attr.size(1),
                        conv=conv, hidden=a.hidden, layers=layers, heads=a.heads, dropout=a.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    say(f"run {run} | device {device}"
        f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''} | "
        f"{len(train)} train / {len(val)} val graphs | {n_params:,} parameters | K={layers}")
    say(json.dumps(vars(a)))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True)
    best, best_epoch, history = float("inf"), -1, []
    ckpt = CKPT_DIR / f"{run}.pt"

    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            mask = batch["flow"].y_mask
            pred = model(batch)
            loss = torch.mean((pred[mask] - batch["flow"].y[mask]) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * int(mask.sum())
            n += int(mask.sum())
        train_mse = tot / n
        vm = evaluate(model, val, device, a.batch_size * 2)
        sched.step(vm["mae_log"])
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_mse_log": train_mse, "val_mae_log": vm["mae_log"],
                        "val_mse_log": vm["mse_log"], "lr": lr, "seconds": time.time() - t0})
        improved = vm["mae_log"] < best - 1e-4
        if improved:
            best, best_epoch = vm["mae_log"], epoch
            torch.save({"state_dict": model.state_dict(), "args": vars(a), "layers": layers, "conv": conv,
                        "epoch": epoch, "val_mae_log": best}, ckpt)
        say(f"epoch {epoch:3d} | train MSE {train_mse:.4f} | val MAE {vm['mae_log']:.4f} MSE {vm['mse_log']:.4f} "
            f"| lr {lr:.1e} | {time.time() - t0:.1f}s{'  *' if improved else ''}")
        pd.DataFrame(history).to_csv(RESULTS_DIR / "train_logs" / f"{run}.csv", index=False)
        if epoch - best_epoch >= a.patience:
            say(f"early stop: no val improvement for {a.patience} epochs")
            break

    # ---- final: reload the best epoch, predict val + the two test sets once ----
    model.load_state_dict(torch.load(ckpt, weights_only=False)["state_dict"])
    say(f"best epoch {best_epoch}, val MAE_log {best:.4f}")
    summary = {"run": run, "model": a.model, "seed": a.seed, "tag": a.tag, "hidden": a.hidden, "layers": layers,
               "heads": a.heads, "dropout": a.dropout, "lr": a.lr, "wd": a.wd, "batch_size": a.batch_size,
               "max_scenarios": a.max_scenarios, "best_epoch": best_epoch, "val_mae_log": best,
               "n_params": n_params, "minutes": sum(h["seconds"] for h in history) / 60}
    if not a.no_preds:
        preds = [predict(model, val, device, a.batch_size * 2)]
        for split in ("test", "test_unseen_topo"):
            graphs = load_graphs(split)
            if a.max_scenarios:
                graphs = graphs[: max(10, a.max_scenarios // 4)]
            preds.append(predict(model, graphs, device, a.batch_size * 2))
        allp = pd.concat(preds, ignore_index=True)
        allp = allp[allp.y_mask]
        name = a.model if not a.tag else f"{a.model}_{a.tag}"
        say("predictions -> " + save_predictions(allp, name, a.seed))
    hp = RESULTS_DIR / "hpsearch_gnn.csv"
    pd.DataFrame([summary]).to_csv(hp, mode="a", header=not hp.exists(), index=False)
    say(json.dumps(summary))


if __name__ == "__main__":
    main()
