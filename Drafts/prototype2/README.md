# IEEE grid GNN experiment package

This package contains the generator, supervised GNN baseline, masked physics pretraining, topology hold-out fine-tuning, saved data/checkpoints, and result summaries.

## Results

See `results.md` for the per-grid metrics and limitations. `results/transfer_case118.png` plots voltage MAE and angle RMSE against the number of labeled fine-tuning samples. Raw metrics are in `results/summary.json` and `results/transfer_case118.json`.

## Reproduce

From this directory, install dependencies and run:

```powershell
python -m pip install -r requirements.txt
python run_experiments.py --n 300 --epochs 80 --pretrain_epochs 4 --holdout case118
python summarize_results.py --results results --models models --data data --holdout case118 --out results.md
```

The generation defaults sample normal loads in `[0.6, 1.4]`, stress loads in `[1.3, 1.6]`, and request 30% single-branch outages. Outages that island the graph are rejected. Accepted outage and stress fractions can be below their requested proportions because stressed or outaged power flows may fail to converge.

## Method notes

- The supervised baseline trains on power-flow unknowns, then reports voltage magnitude MAE and angle RMSE on 30 held-out scenarios per grid (seed 0).
- Masked pretraining uses a 5% feature mask by default. Stage one learns generator Pg/Qg targets; stage two adds the differentiable AC mismatch penalty.
- IEEE 118 is held out of pretraining. Its transfer curve fine-tunes the pretrained network on 1, 5, 10, 25, 50, or 100 labeled examples and scores a fixed set of 60 examples.
- IEEE 118's maximum state mismatch under the differentiable AC residual is 2.19e-4 p.u.; investigate this before relying on its physics-loss result.
- These are synthetic benchmark results, not validation on field networks.
