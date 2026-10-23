"""Reproduce the IEEE grid, supervised baseline, and transfer experiments."""
import argparse, json, os, subprocess, sys
from pathlib import Path

GRIDS = ["case14", "case30", "case57", "case118"]

def run(cmd, env):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, env=env)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--n",type=int,default=300); p.add_argument("--seed",type=int,default=0)
    p.add_argument("--data",default="data"); p.add_argument("--models",default="models")
    p.add_argument("--results",default="results"); p.add_argument("--holdout",default="case118")
    p.add_argument("--epochs",type=int,default=80); p.add_argument("--pretrain_epochs",type=int,default=4)
    p.add_argument("--skip_generate",action="store_true"); p.add_argument("--skip_baseline",action="store_true")
    p.add_argument("--skip_transfer",action="store_true"); p.add_argument("--device",default="cpu")
    a=p.parse_args(); root=Path(__file__).resolve().parent
    data=Path(a.data); models=Path(a.models); results=Path(a.results)
    data.mkdir(parents=True,exist_ok=True); models.mkdir(parents=True,exist_ok=True); results.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    vendor=str(root/"vendor")
    env["PYTHONPATH"]=vendor+os.pathsep+env.get("PYTHONPATH","")
    if not a.skip_generate:
        run([sys.executable,str(root/"gen_data.py"),"--grids",",".join(GRIDS),"--n",str(a.n),
             "--n1_frac","0.30","--stress_frac","0.25","--stress_kmin","1.3","--stress_kmax","1.6",
             "--seed",str(a.seed),"--out",str(data)],env)
    if not a.skip_baseline:
        for grid in GRIDS:
            run([sys.executable,str(root/"train_gnn.py"),"--data",str(data/f"{grid}.pt"),
                 "--out",str(models/f"{grid}_gnn.pt"),"--epochs",str(a.epochs),"--hidden","64",
                 "--layers","4","--batch_size","16","--seed",str(a.seed),"--device",a.device],env)
    if not a.skip_transfer:
        run([sys.executable,str(root/"pretrain_transfer.py"),"--data_dir",str(data),
             "--grids",",".join(GRIDS),"--holdout",a.holdout,"--out",str(results/f"transfer_{a.holdout}.json"),
             "--pretrain_epochs",str(a.pretrain_epochs),"--hidden","64","--layers","4",
             "--batch_size","16","--seed",str(a.seed),"--device",a.device],env)
    metrics={grid:json.loads((models/f"{grid}_gnn_metrics.json").read_text(encoding="utf-8")) for grid in GRIDS if (models/f"{grid}_gnn_metrics.json").exists()}
    summary={"seed":a.seed,"scenarios_per_grid":a.n,"grids":metrics,
             "data_stats":json.loads((data/"stats.json").read_text(encoding="utf-8")) if (data/"stats.json").exists() else {}}
    (results/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
