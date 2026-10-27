# LLM- and GNN-Based Framework for Campus Energy Management

## Purpose

To build a system that can answer questions about campus energy usage and
help optimize it, by combining a physics-grounded graph neural network (GNN)
with a language-based interface — the GNN computes grid state, the LLM
explains it and takes natural-language queries, and neither is asked to do
the other's job.


##  PS Breakdown

The PS splits into three distinct pieces of work

- **The physics engine:** self-supervised masked
  pretraining on power-flow graphs, and cross-topology generalization
  testing, are established methods from the literature this project draws
  on directly (GraphMAE, the Joule GridFM roadmap, PowerGraph). Building
  and validating this at benchmark scale is necessary groundwork and the
  load-bearing piece of evidence for the whole PS, but on its own it is a
  reproduction, not a contribution.

- **The language interface :** conceptually simple — retrieve
  context, call the GNN for numbers, never compute physics in the LLM —
  but the part most likely to consume unplanned time in practice: tool-
  calling reliability, prompt iteration, and how cleanly the LLM reports
  numeric GNN output without distorting it.

- **The comparative evaluation :** this is the piece that
  actually answers the central question and is where any genuine
  contribution lives. It requires reproducing SafePowerGraph-LLM's setup
  closely enough for a fair comparison, which is effort on the same order
  as the physics engine itself, not a quick add-on at the end.



The line of thought, end to end: an LLM alone is not trustworthy for grid
physics → a GNN can be, by construction, but has only been validated on
benchmark-scale synthetic topologies → pair the two so each does what it's
suited for → test that pairing against the alternative (LLM-only) rather
than assume it wins → and only then push the same pipeline toward the
harder, currently-untested case of a real, small network, if the data to do
so becomes available.

## Plan

1. Solve power flow on public benchmark grids and generate a dataset of
   varied load conditions (solver + graph construction pipeline).
2. Train a supervised GNN baseline (predict masked node features, compare
   against the solver) as a correctness check before self-supervision.
3. Move to self-supervised masked pretraining with a physics-consistency
   loss term, following the GridFM-v0 architecture.
4. Hold out a topology, fine-tune on it, and measure transfer.
5. Build the RAG/tool-calling LLM interface on top of the trained GNN.
6. Reproduce an LLM-only baseline (SafePowerGraph-LLM-style) on the same
   benchmark and run the direct comparison in Objective 4.
7. If campus data becomes available, fine-tune and evaluate on it as the
   real-network test in Objective 5.

## References

- Hamann et al., "Foundation models for the electric power grid," *Joule* (2024)
- Huang et al., "Large Foundation Models for Power Systems," *IEEE PES General Meeting* (2024)
- Bernier et al., "SafePowerGraph-LLM: Novel Power Grid Graph Embedding and Optimization with Large Language Models" (2025)
- Karabulut et al., "Generalization of Graph Neural Network Models for Distribution Grid Fault Detection" (2025)
- Varbella et al., "PowerGraph: A Power Grid Benchmark Dataset for Graph Neural Networks" (2024)
- Piloto et al., "CANOS: A fast and scalable neural AC-OPF solver robust to N-1 perturbations" (2024)
- Hou et al., "GraphMAE: Self-Supervised Masked Graph Autoencoders" (2022)

## Run the app

From the repository root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
streamlit run app.py
```

The app loads bundled GNN checkpoints and synthetic scenarios. Ollama explanations are optional. Start Ollama and install a model with `ollama pull qwen2.5:1.5b`; the sidebar lists installed models. Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` to override the defaults.

## App results

Test results use one seeded split of 300 synthetic scenarios per IEEE grid (240/30/30 train/validation/test):

| Grid | Voltage MAE (pu) | Angle RMSE |
|---|---:|---:|
| IEEE 14 | 0.00246 | 1.210° |
| IEEE 30 | 0.00224 | 0.705° |
| IEEE 57 | 0.00929 | 2.403° |
| IEEE 118 | 0.00218 | 1.501° |

IEEE 118 topology transfer at 100 fine-tuning labels: **0.01256 pu Vm MAE**, **4.875° Va RMSE**. Its supplied AC residual is **2.19e-4 pu**. These synthetic single-split results do not establish robustness or operational readiness.
