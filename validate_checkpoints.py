"""Recompute masked-state errors on the seeded held-out split for each case."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import GraphBatch, ac_physics, pf_mask, reconstruct
from model import PowerFlowGNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="mall11,case14,case30,case57,case118")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/validation_audit.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    report = {"seed": args.seed, "cases": {}}
    for case in args.cases.split(","):
        samples = torch.load(root / "data" / f"{case}.pt", map_location="cpu", weights_only=True)
        ckpt = torch.load(root / "models" / f"{case}_gnn.pt", map_location="cpu", weights_only=True)
        model = PowerFlowGNN(hidden=ckpt["hidden"], layers=ckpt["layers"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        n = len(samples)
        order = np.random.default_rng(args.seed).permutation(n).tolist()
        n_train, n_val = max(1, int(.8 * n)), max(1, int(.1 * n))
        if n_train + n_val >= n:
            n_train, n_val = n - 2, 1
        test_ids = order[n_train + n_val:]
        vm_abs = va_sq = max_residual = 0.0
        vm_count = va_count = 0
        per_sample = []
        with torch.inference_mode():
            for idx in test_ids:
                graph = GraphBatch(**{k: v for k, v in samples[idx].items() if torch.is_tensor(v)})
                mask = pf_mask(graph)
                state = reconstruct(model(graph), graph, mask)
                vm_mask, va_mask = mask[:, 4], mask[:, 5]
                vm_abs += (state[vm_mask, 4] - graph.z[vm_mask, 4]).abs().sum().item()
                va_sq += ((state[va_mask, 5] - graph.z[va_mask, 5]) ** 2).sum().item()
                vm_count += int(vm_mask.sum())
                va_count += int(va_mask.sum())
                dp, dq, _ = ac_physics(state[:, 4], state[:, 5], graph)
                rp, rq = dp - (state[:, 2] - state[:, 0]), dq - (state[:, 3] - state[:, 1])
                residual = torch.maximum(rp.abs(), rq.abs()).max().item()
                max_residual = max(max_residual, residual)
                per_sample.append({"scenario": idx, "max_abs_pq_residual_pu": residual})
        report["cases"][case] = {
            "heldout_scenarios": len(test_ids),
            "vm_mae_pu": vm_abs / max(vm_count, 1),
            "va_rmse_deg": float(np.sqrt(va_sq / max(va_count, 1)) * 180 / np.pi),
            "worst_predicted_state_pq_residual_pu": max_residual,
            "per_sample_residuals": per_sample,
            "note": "Seeded held-out split recomputed from the same data/split procedure; one seed is not a robustness study.",
        }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: {m: v for m, v in value.items() if m != "per_sample_residuals"}
                      for k, value in report["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
