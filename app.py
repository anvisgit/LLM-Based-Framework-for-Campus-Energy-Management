from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import plotly.express as px
import requests
import streamlit as st
import torch

from common import GraphBatch, ac_physics, pf_mask, reconstruct
from model import PowerFlowGNN

ROOT = Path(__file__).resolve().parent
CASES = {"Mall demo feeder (11 bus)": "mall11", "IEEE 14 bus": "case14",
         "IEEE 30 bus": "case30", "IEEE 57 bus": "case57", "IEEE 118 bus": "case118"}
BASE_MVA = 100.0


@st.cache_resource
def load_case(case: str):
    samples = torch.load(ROOT / "data" / f"{case}.pt", map_location="cpu", weights_only=True)
    checkpoint = torch.load(ROOT / "models" / f"{case}_gnn.pt", map_location="cpu", weights_only=True)
    model = PowerFlowGNN(hidden=checkpoint["hidden"], layers=checkpoint["layers"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return samples, model


def infer(case: str, scenario_id: int, load_scale: float = 1.0, demand_override: dict | None = None):
    samples, model = load_case(case)
    sample = samples[scenario_id]
    graph = GraphBatch(**{k: v.clone() for k, v in sample.items() if torch.is_tensor(v)})
    graph.z[:, :2] *= load_scale
    if demand_override is not None:
        for bus_id, values in demand_override.items():
            idx = bus_id - 1
            if idx < 0 or idx >= graph.z.shape[0]:
                raise ValueError(f"bus_id {bus_id} is outside this grid's 1..{graph.z.shape[0]} range")
            graph.z[idx, 0] = float(values[0]) / BASE_MVA
            graph.z[idx, 1] = float(values[1]) / BASE_MVA
    mask = pf_mask(graph)
    with torch.inference_mode():
        predicted = reconstruct(model(graph), graph, mask)
    p_bus, q_bus, p_from = ac_physics(predicted[:, 4], predicted[:, 5], graph)
    f, t = graph.ei
    r, x, b, tap, shift = graph.br.unbind(-1)
    ys = 1 / torch.complex(r, x)
    bc = torch.complex(torch.zeros_like(b), b / 2)
    tapc = torch.polar(tap, shift)
    voltage = torch.polar(predicted[:, 4], predicted[:, 5])
    sf = voltage[f] * torch.conj((ys + bc) / tap.square() * voltage[f] - ys / torch.conj(tapc) * voltage[t])
    st_ = voltage[t] * torch.conj(-ys / tapc * voltage[f] + (ys + bc) * voltage[t])
    losses_kw = (sf.real + st_.real) * BASE_MVA * 1000
    dp, dq = p_bus - (predicted[:, 2] - predicted[:, 0]), q_bus - (predicted[:, 3] - predicted[:, 1])
    comparable = load_scale == 1.0 and demand_override is None
    voltage_mae = (predicted[mask[:, 4], 4] - graph.z[mask[:, 4], 4]).abs().mean().item() if comparable else None
    angle_rmse = torch.sqrt(((predicted[mask[:, 5], 5] - graph.z[mask[:, 5], 5]) ** 2).mean()).item() * 180 / np.pi if comparable else None
    n = graph.z.shape[0]
    buses = [{"bus": f"Bus {i + 1}", "load_mw": float(graph.z[i, 0] * BASE_MVA),
              "voltage_pu": float(predicted[i, 4]), "reference_voltage_pu": float(graph.z[i, 4]),
              "angle_deg": float(predicted[i, 5] * 180 / np.pi)} for i in range(n)]
    branches = [{"from": f"Bus {int(f[i]) + 1}", "to": f"Bus {int(t[i]) + 1}",
                 "active_flow_mw": float(p_from[i] * BASE_MVA), "estimated_loss_kw": float(losses_kw[i])}
                for i in range(len(f))]
    return {"case": case, "scenario": scenario_id, "buses": buses, "branches": branches,
            "load_scale": load_scale,
            "total_load_mw": float(graph.z[:, 0].sum() * BASE_MVA),
            "voltage_min_pu": min(x["voltage_pu"] for x in buses),
            "voltage_max_pu": max(x["voltage_pu"] for x in buses),
            "estimated_loss_kw": float(losses_kw.sum()),
            "max_abs_pq_mismatch_pu": float(torch.maximum(dp.abs(), dq.abs()).max()),
            "scenario_vm_mae_pu": voltage_mae, "scenario_va_rmse_deg": angle_rmse,
            "has_solver_comparison": comparable}


@st.cache_data(show_spinner="Searching load scenarios with the GNN…")
def optimize_load(case: str, scenario_id: int):
    trials = [infer(case, scenario_id, float(s)) for s in np.linspace(0.6, 1.4, 17)]
    feasible = [x for x in trials if x["voltage_min_pu"] >= 0.95 and
                x["voltage_max_pu"] <= 1.05 and x["max_abs_pq_mismatch_pu"] <= 0.02]
    if not feasible:
        return None, trials
    return max(feasible, key=lambda x: x["total_load_mw"]), trials


def explain(question: str, snapshot: dict, model_name: str, base_url: str) -> str:
    system = "Answer in 1-2 sentences using only the labeled metrics. Do not calculate, change units, or invent limits. P/Q mismatch is not voltage error. Synthetic data; no operational advice."
    fields = {"Demand (MW)": "total_load_mw", "Voltage minimum (pu)": "voltage_min_pu",
              "Voltage maximum (pu)": "voltage_max_pu", "Estimated losses (kW)": "estimated_loss_kw",
              "Maximum P/Q mismatch (pu)": "max_abs_pq_mismatch_pu",
              "Sample voltage MAE (pu)": "scenario_vm_mae_pu",
              "Sample angle RMSE (degrees)": "scenario_va_rmse_deg"}
    summary = "\n".join(f"{label}: {snapshot[key]:.4g}" for label, key in fields.items()
                        if snapshot.get(key) is not None)
    r = requests.post(f"{base_url.rstrip('/')}/api/chat", timeout=90,
                      json={"model": model_name, "stream": False, "messages": [
                          {"role": "system", "content": system},
                          {"role": "user", "content": f"Question: {question}\nResults:\n{summary}"}]})
    r.raise_for_status()
    return r.json()["message"]["content"]


@st.cache_data(ttl=30)
def ollama_models(base_url: str) -> list[str]:
    r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
    r.raise_for_status()
    return [model["name"] for model in r.json().get("models", []) if model.get("name")]


st.set_page_config(page_title="version-3", layout="wide")
st.title("version-3")
with st.sidebar:
    st.header("Grid")
    case = CASES[st.selectbox("Grid", list(CASES))]
    count = 1000 if case == "mall11" else 300
    scenario_id = st.number_input("Scenario", min_value=0, max_value=count - 1, value=0, step=1)
    load_scale = st.slider("Load scale", 0.6, 1.4, 1.0, 0.05,
                           help="Scales the selected synthetic scenario.")
    st.caption("Synthetic scenarios.")
    st.divider()
    st.header("Ollama")
    ollama_url = st.text_input("Ollama URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    try:
        models = ollama_models(ollama_url)
    except requests.RequestException:
        models = []
        st.caption("Ollama unavailable.")
    if models:
        preferred = os.getenv("OLLAMA_MODEL", "")
        ollama_model = st.selectbox("Model", models, index=models.index(preferred) if preferred in models else 0)
    else:
        ollama_model = st.text_input("Model", os.getenv("OLLAMA_MODEL", "llama3.1:8b"))

st.subheader("Inputs")
uploaded = st.file_uploader("Demand overrides (CSV)", type=["csv"],
                            help="Columns: bus_id,pd_mw,qd_mvar. Bus IDs start at 1.")
demand_override = None
if uploaded is not None:
    try:
        import pandas as pd
        frame = pd.read_csv(uploaded)
        required = {"bus_id", "pd_mw", "qd_mvar"}
        if not required.issubset(frame.columns):
            raise ValueError("CSV needs columns bus_id,pd_mw,qd_mvar")
        if frame[list(required)].isna().any().any() or frame["bus_id"].duplicated().any():
            raise ValueError("CSV values must be complete and bus_id values unique")
        if (frame[["pd_mw", "qd_mvar"]] < 0).any().any():
            raise ValueError("Loads must be non-negative MW/Mvar values")
        demand_override = {int(row.bus_id): (float(row.pd_mw), float(row.qd_mvar)) for row in frame.itertuples()}
    except Exception as exc:
        st.error(f"Invalid demand CSV: {exc}")
        st.stop()

try:
    result = infer(case, int(scenario_id), float(load_scale), demand_override)
except Exception as exc:
    st.error(f"Could not load or run the selected GNN: {exc}")
    st.stop()

st.warning("Synthetic demo only; not live campus data or a control system.")
k = st.columns(4)
k[0].metric("Demand", f"{result['total_load_mw']:.2f} MW")
k[1].metric("Voltage", f"{result['voltage_min_pu']:.3f}–{result['voltage_max_pu']:.3f} pu")
k[2].metric("Losses", f"{result['estimated_loss_kw']:.1f} kW")
k[3].metric("Max P/Q mismatch", f"{result['max_abs_pq_mismatch_pu']:.2e} pu")

left, right = st.columns([1, 1])
with left:
    st.subheader("Voltage: GNN vs. solver")
    voltage_series = ["voltage_pu", "reference_voltage_pu"] if result["has_solver_comparison"] else ["voltage_pu"]
    fig = px.line(result["buses"], x="bus", y=voltage_series, markers=True,
                  labels={"value": "Voltage (pu)", "variable": "Series", "bus": "Bus"},
                  color_discrete_map={"voltage_pu": "#087e78", "reference_voltage_pu": "#ee8b32"})
    fig.update_layout(height=370, margin=dict(l=10, r=10, t=15, b=10))
    st.plotly_chart(fig, width="stretch")
    if result["has_solver_comparison"]:
        st.caption(f"Sample error: Vm MAE {result['scenario_vm_mae_pu']:.5f} pu; Va RMSE {result['scenario_va_rmse_deg']:.3f}°. Not a validation metric.")
    else:
        st.caption("Modified loads: solver comparison unavailable.")
with right:
    st.subheader("Ask Ollama")
    with st.form("ollama_form"):
        question = st.text_area("Question", placeholder="Explain these results.", height=100)
        submitted = st.form_submit_button("Explain", type="primary")
    if submitted and question.strip():
        with st.spinner(f"Asking {ollama_model}…"):
            try:
                st.session_state["answer"] = explain(question, result, ollama_model, ollama_url)
                st.session_state.pop("llm_error", None)
            except requests.RequestException as exc:
                st.session_state["llm_error"] = f"Ollama request failed: {exc}"
            except (KeyError, ValueError) as exc:
                st.session_state["llm_error"] = f"Unexpected Ollama response: {exc}"
    if st.session_state.get("answer"):
        st.markdown(st.session_state["answer"])
    if st.session_state.get("llm_error"):
        st.warning(st.session_state["llm_error"])
    with st.expander("Branches"):
        st.dataframe(result["branches"], width="stretch", hide_index=True)

with st.expander("Buses"):
    st.dataframe(result["buses"], width="stretch", hide_index=True)

st.subheader("Load feasibility")
st.caption("Tests 17 load scales (0.6x–1.4x); voltage 0.95–1.05 pu, mismatch ≤0.02 pu. Demo only; not dispatch.")
if st.button("Search feasible load"):
    try:
        best, trials = optimize_load(case, int(scenario_id))
        st.session_state["optimizer_result"] = (best, trials)
    except Exception as exc:
        st.error(f"Search failed: {exc}")
if st.session_state.get("optimizer_result"):
    best, trials = st.session_state["optimizer_result"]
    if best is None:
        st.info("No candidate met constraints; see table.")
    else:
        st.metric("Highest feasible demand", f"{best['total_load_mw']:.2f} MW", f"{best['load_scale']:.2f}x")
    st.dataframe([{k: x[k] for k in ["load_scale", "total_load_mw", "voltage_min_pu", "voltage_max_pu", "max_abs_pq_mismatch_pu", "estimated_loss_kw"]} for x in trials], width="stretch", hide_index=True)

st.subheader("Topology transfer")
st.markdown("IEEE 118 transfer improves with fine-tuning but remains weak at 100 labels.")
transfer_plot = ROOT / "results" / "transfer_case118.png"
if transfer_plot.exists():
    st.image(str(transfer_plot), caption="IEEE 118 transfer results")
