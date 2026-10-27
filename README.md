# LLM- and GNN-Based Framework for Campus Energy Management


## Run

From PowerShell in this folder, with Python 3.10 or 3.11 installed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
streamlit run app.py
```

The dashboard uses Ollama's `/api/chat` endpoint. Start Ollama and pull a model, for example `ollama pull qwen2.5:1.5b`. The sidebar lists models installed at the configured Ollama URL. GNN results work without Ollama; explanations need a reachable server. Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` to change the defaults. `requirements.txt` includes training libraries; `requirements-app.txt` is the smaller runtime set.

## Scenario input and demo search

Choose a grid and stored scenario, then adjust the load multiplier. Optional CSV overrides use `bus_id,pd_mw,qd_mvar` (bus IDs start at 1; omitted buses retain the selected scenario's scaled load). CSV-edited and scaled cases do not show solver-label errors because the stored solved labels no longer match the modified inputs.

The **Illustrative load feasibility search** sweeps 17 multipliers from 0.6× to 1.4× and returns the highest demand predicted to satisfy the demo voltage band and a configured residual threshold. This is a small GNN-based constraint search, not cost-aware scheduling, equipment dispatch, or a validated optimizer. Treat it as an experiment only.

## Recheck the held-out results

Run `python validate_checkpoints.py` to recalculate masked-voltage errors and maximum predicted-state P/Q residuals on each model's seeded test split. It writes `results/validation_audit.json`. This checks reproducibility for one split and seed; it does not establish robustness. For stronger evidence, repeat training and evaluation over multiple seeds and stress/topology holdouts before considering any real-world use.

## Container deployment and basic hardening

Build with `docker build -t campus-energy-copilot .`. On Windows, run locally with the Ollama host exposed to Docker:

```powershell
docker run --rm -p 127.0.0.1:8501:8501 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 campus-energy-copilot
```


## Results and limits

Test results use one seeded split of 300 synthetic scenarios per IEEE grid (240/30/30 train/validation/test):

| Grid | Voltage MAE (pu) | Angle RMSE |
|---|---:|---:|
| IEEE 14 | 0.00246 | 1.210° |
| IEEE 30 | 0.00224 | 0.705° |
| IEEE 57 | 0.00929 | 2.403° |
| IEEE 118 | 0.00218 | 1.501° |

IEEE 118 topology transfer at 100 fine-tuning labels: **0.01256 pu Vm MAE**, **4.875° Va RMSE**. Its supplied AC residual is **2.19e-4 pu**. Transfer errors remain high. These single-split synthetic results do not establish robustness or operational readiness.


## Layout

- `app.py` — Streamlit scenario selection, GNN inference, charts, Ollama REST client
- `model.py`, `common.py` — GNN, graph features, AC physics and residual functions
- `data/` — generated scenarios and generation statistics for all five demo cases
- `models/` — pretrained case checkpoints and their reported test metrics
- `results/` — topology-transfer outputs, metrics, plot, and experiment notes
- `gen_data.py`, `train_gnn.py`, `pretrain_transfer.py`, `run_experiments.py`, `summarize_results.py` — reproduction/training utilities
- `validate_checkpoints.py` — seeded held-out checkpoint audit
- `Dockerfile`, `requirements-app.txt`, `.gitignore` — container and repository hygiene

The `.pt` data/checkpoint files use PyTorch serialization. Only load these bundled files or files from a trusted source.
