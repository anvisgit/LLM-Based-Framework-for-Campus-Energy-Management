"""Masked graph pretraining, physics fine-tuning, and topology hold-out sweep."""
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
from common import (collate, random_mask, reconstruct, full_state_residual,
                    pf_mask, Z_SHIFT, Z_SCALE)
from model import PowerFlowGNN
from train_gnn import batches, loss_for, evaluate


def masked_loss(model, g, ratio, physics_weight=0.0, pgqg_weight=1.0):
    mask = random_mask(g, ratio)
    # Ensure each graph contributes a target even for low mask ratios.
    for j in range(g.num_graphs):
        ids = (g.batch == j).nonzero().flatten()
        if not mask[ids].any():
            mask[ids[0], 4] = True
        gen_ids = ids[g.bt[ids] != 0]
        if len(gen_ids):
            mask[gen_ids[0], 2:4] = True
    out = model(g, mask)
    target = (g.z - Z_SHIFT.to(g.z.device)) / Z_SCALE.to(g.z.device)
    recon = ((out[mask] - target[mask]) ** 2).mean()
    gen = ((g.bt != 0) & mask[:, 2:4].any(dim=1))
    pgqg = ((out[gen, 2:4] - target[gen, 2:4]) ** 2).mean() if gen.any() else recon * 0
    if physics_weight:
        zhat = reconstruct(out, g, mask)
        dp, dq = full_state_residual(zhat, g)
        phys = (dp.square().mean() + dq.square().mean())
    else:
        phys = recon * 0
    return recon + pgqg_weight * pgqg + physics_weight * phys, {
        "reconstruction": recon, "pgqg": pgqg, "physics": phys}


def save_state(path, model, hidden, layers, seed):
    torch.save({"model_state": model.state_dict(), "hidden": hidden,
                "layers": layers, "seed": seed}, path)


def pretrain(train_sets, device, a):
    model = PowerFlowGNN(hidden=a.hidden, layers=a.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    # Stage 1 explicitly learns generator Pg/Qg from low-ratio masks before
    # enabling the AC residual, which depends on those predicted injections.
    for stage, physics in ((1, 0.0), (2, a.physics_weight)):
        for epoch in range(a.pretrain_epochs):
            model.train(); total = 0.0; steps = 0
            for samples in train_sets:
                ids = np.random.permutation(len(samples))
                for start in range(0, len(ids), a.batch_size):
                    g = collate([samples[i] for i in ids[start:start+a.batch_size]]).to(device)
                    opt.zero_grad(set_to_none=True)
                    loss, terms = masked_loss(model, g, a.mask_ratio, physics, a.pgqg_weight)
                    if stage == 1:
                        loss = terms["pgqg"]
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    total += loss.item(); steps += 1
            print(json.dumps({"stage": stage, "epoch": epoch+1, "loss": total/max(steps,1)}))
    return model


def finetune_eval(init_state, target, holdout_name, sizes, device, a, output):
    n = len(target); rng = np.random.default_rng(a.seed)
    order = rng.permutation(n).tolist()
    n_test = max(1, int(.2*n)); n_val = max(1, int(.1*n))
    test_ids = order[:n_test]; val_ids = order[n_test:n_test+n_val]
    pool = order[n_test+n_val:]
    results = []
    for size in sizes:
        if size > len(pool):
            continue
        model = PowerFlowGNN(hidden=a.hidden,layers=a.layers).to(device)
        model.load_state_dict(init_state)
        train_ids = pool[:size]
        opt = torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
        # Small-data transfer: fixed short schedule avoids a large validation
        # split consuming the few available examples.
        for _ in range(a.finetune_epochs):
            model.train()
            for g in batches(train_ids,target,a.batch_size,device,shuffle=True):
                opt.zero_grad(set_to_none=True); loss=loss_for(model,g)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        metrics=evaluate(model,test_ids,target,a.batch_size,device)
        results.append({"holdout_grid":holdout_name,"finetune_samples":size,
                        "test_samples":len(test_ids),"metrics":metrics})
        print(json.dumps(results[-1]))
    output.write_text(json.dumps(results,indent=2),encoding="utf-8")
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir",default="data"); p.add_argument("--grids",default="case14,case30,case57,case118")
    p.add_argument("--holdout",default="case118"); p.add_argument("--out",default="results/transfer_case118.json")
    p.add_argument("--sizes",default="1,5,10,25,50,100")
    p.add_argument("--hidden",type=int,default=64); p.add_argument("--layers",type=int,default=4)
    p.add_argument("--batch_size",type=int,default=16); p.add_argument("--pretrain_epochs",type=int,default=20)
    p.add_argument("--finetune_epochs",type=int,default=10); p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--mask_ratio",type=float,default=.05); p.add_argument("--physics_weight",type=float,default=.05)
    p.add_argument("--pgqg_weight",type=float,default=2.0); p.add_argument("--seed",type=int,default=0)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    device=torch.device(a.device); grids=a.grids.split(",")
    if a.holdout not in grids: raise ValueError("holdout must appear in --grids")
    sets={name:torch.load(Path(a.data_dir)/f"{name}.pt",map_location="cpu",weights_only=False) for name in grids}
    model=pretrain([sets[n] for n in grids if n!=a.holdout],device,a)
    output=Path(a.out); output.parent.mkdir(parents=True,exist_ok=True)
    save_state(output.with_name(output.stem+"_pretrained.pt"),model,a.hidden,a.layers,a.seed)
    sizes=[int(v) for v in a.sizes.split(",")]
    finetune_eval(model.state_dict(),sets[a.holdout],a.holdout,sizes,device,a,output)

if __name__=="__main__":
    main()
