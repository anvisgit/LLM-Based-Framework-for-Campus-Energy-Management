"""Train and evaluate the mall power-flow GNN on generated graph scenarios."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from common import collate, pf_mask, Z_SCALE, Z_SHIFT
from model import PowerFlowGNN


def batches(indices, samples, batch_size, device, shuffle=False):
    ids = list(indices)
    if shuffle:
        random.shuffle(ids)
    for start in range(0, len(ids), batch_size):
        yield collate([samples[i] for i in ids[start:start + batch_size]]).to(device)


def loss_for(model, g):
    out = model(g)
    mask = pf_mask(g)
    # Predict voltage magnitude/angle unknowns only; known reference/PV values stay observed.
    target = (g.z - Z_SHIFT.to(g.z.device)) / Z_SCALE.to(g.z.device)
    vmva_mask = torch.zeros_like(mask)
    vmva_mask[:, 4:6] = mask[:, 4:6]
    if not vmva_mask.any():
        raise RuntimeError("This graph batch has no hidden Vm/Va targets")
    return ((out[vmva_mask] - target[vmva_mask]) ** 2).mean()


@torch.no_grad()
def evaluate(model, indices, samples, batch_size, device):
    model.eval()
    vm_abs, va_sq, count = 0.0, 0.0, 0
    for g in batches(indices, samples, batch_size, device):
        out = model(g)
        mask = pf_mask(g)
        vm_mask, va_mask = mask[:, 4], mask[:, 5]
        scale, shift = Z_SCALE.to(device), Z_SHIFT.to(device)
        pred_vm = out[:, 4] * scale[4] + shift[4]
        pred_va = out[:, 5] * scale[5] + shift[5]
        vm_abs += (pred_vm[vm_mask] - g.z[vm_mask, 4]).abs().sum().item()
        va_sq += ((pred_va[va_mask] - g.z[va_mask, 5]) ** 2).sum().item()
        count += int(vm_mask.sum().item())
    # The mall feeder uses PQ buses, so both masks cover the unknown bus voltages.
    vm_count = sum(int((samples[i]["bt"] == 0).sum()) for i in indices)
    va_count = vm_count + sum(int((samples[i]["bt"] == 1).sum()) for i in indices)
    if vm_count == 0 or va_count == 0:
        raise RuntimeError("Evaluation split has no unknown voltage targets")
    return {"vm_mae_pu": vm_abs / vm_count,
            "va_rmse_deg": float(np.sqrt(va_sq / va_count) * 180.0 / np.pi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mall11.pt")
    ap.add_argument("--out", default="models/mall11_gnn.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    # Small graph batches become much faster and more reproducible without
    # large BLAS/OpenMP thread pools on shared CPU environments.
    torch.set_num_threads(1)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device(a.device)
    samples = torch.load(a.data, map_location="cpu", weights_only=False)
    n = len(samples)
    if n < 10:
        raise ValueError(f"Need at least 10 scenarios for train/validation/test splits; got {n}")
    order = np.random.default_rng(a.seed).permutation(n).tolist()
    n_train, n_val = max(1, int(.8 * n)), max(1, int(.1 * n))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    train_ids = order[:n_train]
    val_ids = order[n_train:n_train + n_val]
    test_ids = order[n_train + n_val:]

    model = PowerFlowGNN(hidden=a.hidden, layers=a.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    best_val, best_epoch, stale = float("inf"), 0, 0
    out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, a.epochs + 1):
        model.train()
        train_loss, steps = 0.0, 0
        for g in batches(train_ids, samples, a.batch_size, device, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, g)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item(); steps += 1
        val = evaluate(model, val_ids, samples, a.batch_size, device)
        score = val["vm_mae_pu"] + val["va_rmse_deg"] / 180.0
        if score < best_val:
            best_val, best_epoch, stale = score, epoch, 0
            torch.save({"model_state": model.state_dict(), "hidden": a.hidden,
                        "layers": a.layers, "seed": a.seed,
                        "split": {"train": train_ids, "validation": val_ids,
                                  "test": test_ids}}, out_path)
        else:
            stale += 1
        print(json.dumps({"epoch": epoch, "train_loss_norm_mse": train_loss / max(steps, 1),
                          "validation": val, "best_epoch": best_epoch}))
        if stale >= a.patience:
            break

    checkpoint = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(model, test_ids, samples, a.batch_size, device)
    report = {"case": Path(a.data).stem, "scenarios": n, "seed": a.seed,
              "split_sizes": {"train": len(train_ids), "validation": len(val_ids),
                              "test": len(test_ids)}, "best_epoch": best_epoch,
              "test_metrics": test, "checkpoint": str(out_path)}
    metrics_path = out_path.with_name(out_path.stem + "_metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
