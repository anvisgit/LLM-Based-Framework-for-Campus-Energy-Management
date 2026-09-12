# implemented the first building block of the GridFM approach 
# solved AC power flow on the IEEE 14-bus system using pandapower
# extracted the four node variables (P, Q, V, θ) per bus plus the line/edge connectivity
# identify, collect, and generate training data

import pandapower as pp
import pandapower.networks as pn
import pandas as pd
net = pn.case14()
pp.runpp(net, algorithm="nr")
bus_results = net.res_bus.copy()
bus_results.index.name = "bus_id"
bus_results = bus_results.rename(columns={
    "p_mw": "P (MW)",
    "q_mvar": "Q (MVAr)",
    "vm_pu": "V (p.u.)",
    "va_degree": "theta (deg)",
})

print("=== IEEE 14-bus: solved power flow (per bus) ===")
print(bus_results.round(4).to_string())

print("\n=== Line loading (top 5 most loaded lines) ===")
line_results = net.res_line[["p_from_mw", "q_from_mvar", "loading_percent"]]
print(line_results.sort_values("loading_percent", ascending=False).head(5).round(3))

print(f"\nConverged: {net['converged']}")
print(f"Number of buses: {len(net.bus)}")
print(f"Number of lines/transformers: {len(net.line) + len(net.trafo)}")
bus_results.to_csv("/mnt/user-data/outputs/ieee14_bus_powerflow.csv")
edges = net.line[["from_bus", "to_bus", "r_ohm_per_km", "x_ohm_per_km", "length_km"]].copy()
edges.to_csv("/mnt/user-data/outputs/ieee14_edges.csv", index=False)

print("\nSaved: ieee14_bus_powerflow.csv (node features), ieee14_edges.csv (graph edges)")
