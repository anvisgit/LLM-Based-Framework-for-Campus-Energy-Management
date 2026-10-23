# MallEnergyLLM Results and Practical Analysis

## Results

The following values are recorded in `data/stats.json` and `models/mall11_gnn_metrics.json`.

### Synthetic data generation

| Measure | Result |
|---|---:|
| Network case | `mall11` synthetic 11-bus feeder |
| Generated scenarios | 1,000 |
| Buses | 11 |
| Power-flow non-convergence rate | 0% |
| Accepted N-1 outage fraction | 0% |
| Scenarios with a voltage violation | 0% |
| Maximum P/Q mismatch in solved data | 7.79e-11 p.u. |
| Generation time | 1.94 minutes |

### GNN evaluation

| Measure | Result |
|---|---:|
| Train / validation / test scenarios | 800 / 100 / 100 |
| Random seed | 0 |
| Best epoch | 61 |
| Test voltage magnitude MAE | 0.000154 p.u. |
| Test voltage angle RMSE | 0.00523 degrees |

The voltage metrics are measured on unknown voltage magnitude and angle values in the held-out test split. Lower errors indicate closer predictions to the AC power-flow solutions used as targets.

## Practical analysis

The test errors are very small. On the sampled operating conditions for this fixed synthetic feeder, the GNN reproduces the solver's bus voltage magnitudes and angles closely. This is a promising proof of concept for learning power-flow solutions over a limited range of load scenarios.

The 0% non-convergence rate and near-zero maximum P/Q mismatch indicate that the generated solver cases converged and satisfy the data-generation residual check. The mismatch describes the reference solutions in the generated dataset; it does not verify that the GNN's predicted states satisfy power balance.

The reported performance applies only to a narrow test: 100 held-out scenarios drawn from the same sampling process and fixed 11-bus topology as the training data. The data statistics report no accepted line outages, and no generated scenario had a voltage violation. Therefore, these results do not demonstrate performance during contingencies, abnormal voltage conditions, unseen topologies, or real mall operations.

The evaluation uses one random seed and no baseline comparison. The test set is useful for an initial check, but broader claims would need repeated seeds, comparison against simple baselines and the conventional AC solver, and additional tests covering outage and out-of-range operating scenarios. A physics residual check on the GNN predictions would also be needed before considering them suitable for operational decisions.

## Interpretation

These results support the statement that the GNN performs very accurately on held-out samples from this synthetic feeder's sampled operating range. They do not establish a field-ready model, outage robustness, voltage-violation detection capability, or an end-to-end LLM assistant. The feeder and loads are synthetic and are not calibrated to a real property.
