"""Generate synthetic mall or IEEE AC power-flow scenarios as graph data.

    python gen_data.py --grids mall11 --n 10000 --n1_frac 0.3
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top
from common import ac_physics, full_state_residual, GraphBatch, collate, BASE_MVA

def mall_feeder():
    """Synthetic 11-bus mall feeder; zones are planning proxies, not site data."""
    net = pp.create_empty_network(sn_mva=100.0)
    bus = [pp.create_bus(net, vn_kv=11.0, name=name) for name in
           ["Utility", "HVAC", "Lighting", "FoodCourt", "RetailEast", "RetailWest",
            "Cinema", "Parking", "ColdStorage", "Escalators", "Offices"]]
    pp.create_ext_grid(net, bus[0], vm_pu=1.0)
    # Synthetic radial distribution impedances, in ohms at 11 kV.
    for i in range(1, len(bus)):
        parent = 0 if i <= 3 else (1 + (i - 1) // 3)
        parent = min(parent, i - 1)
        pp.create_line_from_parameters(net, bus[parent], bus[i], length_km=0.25,
            r_ohm_per_km=0.12, x_ohm_per_km=0.08, c_nf_per_km=0.0,
            max_i_ka=0.8, name=f"Mall feeder {parent}-{i}")
    # Baseline loads (MW/Mvar); scenario generation scales these values.
    loads = [(0.30, .10), (.45, .14), (.65, .22), (.55, .18), (.50, .16),
             (.80, .32), (.25, .08), (.35, .12), (.18, .06), (.30, .10)]
    for i, (p, q) in enumerate(loads, start=1):
        pp.create_load(net, bus[i], p_mw=p, q_mvar=q, name=net.bus.name.iloc[i])
    return net


GRIDS = {"mall11": mall_feeder, "case14": pn.case14, "case30": pn.case30,
         "case57": pn.case57, "case118": pn.case118}
# MATPOWER column indices of pandapower's internal ppc
B_TYPE, B_PD, B_QD, B_GS, B_BS, B_VM, B_VA, B_KV = 1, 2, 3, 4, 5, 7, 8, 9
G_BUS, G_PG, G_QG, G_ST = 0, 1, 2, 7
BR_F, BR_T, BR_R, BR_X, BR_B, BR_TAP, BR_SH, BR_ST = 0, 1, 2, 3, 4, 8, 9, 10


def extract(net):
    """Read the solved state out of pandapower's internal ppc -> dict of tensors."""
    ppc = net._ppc
    base = float(ppc["baseMVA"])
    bus, gen, br = ppc["bus"], ppc["gen"], ppc["branch"]
    n = bus.shape[0]
    assert np.all(bus[:, 0] == np.arange(n)), "unexpected bus numbering"
    if np.any(bus[:, B_TYPE] == 4):
        return None                                        # isolated bus
    bt = (bus[:, B_TYPE] - 1).astype(np.int64)             # 1,2,3 -> PQ,PV,REF = 0,1,2
    pg, qg = np.zeros(n), np.zeros(n)
    on = gen[:, G_ST] > 0
    np.add.at(pg, gen[on, G_BUS].astype(int), gen[on, G_PG] / base)
    np.add.at(qg, gen[on, G_BUS].astype(int), gen[on, G_QG] / base)
    va = np.deg2rad(bus[:, B_VA]); va = va - va[bt == 2][0]
    z = np.stack([bus[:, B_PD] / base, bus[:, B_QD] / base, pg, qg, bus[:, B_VM], va], 1)
    brs = br[br[:, BR_ST] > 0]
    tap = np.where(brs[:, BR_TAP] == 0, 1.0, brs[:, BR_TAP])
    brp = np.stack([brs[:, BR_R], brs[:, BR_X], brs[:, BR_B], tap, np.deg2rad(brs[:, BR_SH])], 1)
    T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt)
    return dict(z=T(z, torch.float64), bt=T(bt, torch.long), gs=T(bus[:, B_GS] / base, torch.float64),
                bs=T(bus[:, B_BS] / base, torch.float64), kv=T(bus[:, B_KV], torch.float64),
                ei=T(brs[:, [BR_F, BR_T]].T, torch.long), br=T(brp, torch.float64))


def to_f32(s):
    return {k: (v.float() if v.is_floating_point() else v) for k, v in s.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="mall11")
    ap.add_argument("--n", type=int, default=10000, help="converged samples per grid")
    ap.add_argument("--n1_frac", type=float, default=0.3, help="fraction with one branch outage")
    ap.add_argument("--kmin", type=float, default=0.6)
    ap.add_argument("--kmax", type=float, default=1.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    stats = {}
    for name in a.grids.split(","):
        rng = np.random.default_rng(a.seed)
        net = GRIDS[name]()
        b_lp, b_lq = net.load.p_mw.values.copy(), net.load.q_mvar.values.copy()
        b_gp, b_gv = net.gen.p_mw.values.copy(), net.gen.vm_pu.values.copy()
        b_ev = net.ext_grid.vm_pu.values.copy()
        samples, tries, nonconv, n1_n, viol = [], 0, 0, 0, 0
        worst = 0.0; t0 = time.time()
        while len(samples) < a.n:
            tries += 1
            k = rng.uniform(a.kmin, a.kmax)
            net.load.p_mw = b_lp * k * np.clip(rng.normal(1, 0.1, len(b_lp)), 0.5, 1.5)
            net.load.q_mvar = b_lq * k * np.clip(rng.normal(1, 0.1, len(b_lq)), 0.5, 1.5)
            net.gen.p_mw = b_gp * k                       # slack picks up the rest
            net.gen.vm_pu = np.clip(b_gv + rng.uniform(-0.02, 0.02, len(b_gv)), 0.95, 1.05)
            net.ext_grid.vm_pu = np.clip(b_ev + rng.uniform(-0.02, 0.02, len(b_ev)), 0.95, 1.05)
            net.line.in_service = True
            if len(net.trafo): net.trafo.in_service = True
            outage = -1
            if rng.random() < a.n1_frac:
                nl, nt = len(net.line), len(net.trafo)
                j = int(rng.integers(nl + nt))
                if j < nl: net.line.at[net.line.index[j], "in_service"] = False
                else: net.trafo.at[net.trafo.index[j - nl], "in_service"] = False
                outage = j
                if len(top.unsupplied_buses(net)) > 0:    # would island the network
                    continue
            try:
                pp.runpp(net, numba=False)
            except Exception:
                nonconv += 1; continue
            s = extract(net)
            if s is None: continue
            g = collate([s])                               # float64 physics check
            dP, dQ = full_state_residual(g.z, g)
            worst = max(worst, float(torch.maximum(dP.abs(), dQ.abs()).max()))
            s = to_f32(s); s["outage"] = outage
            vm = s["z"][:, 4]
            viol += int(((vm < 0.95) | (vm > 1.05)).any()); n1_n += outage >= 0
            samples.append(s)
        torch.save(samples, out / f"{name}.pt")
        stats[name] = dict(n=len(samples), buses=int(samples[0]["z"].shape[0]),
                           nonconv_rate=nonconv / max(tries, 1), n1_frac=n1_n / len(samples),
                           frac_with_voltage_violation=viol / len(samples),
                           max_mismatch_pu=worst, minutes=(time.time() - t0) / 60)
        print(name, stats[name])
        assert worst < 1e-5, f"{name}: stored solutions violate power balance ({worst}) -> bug"
    json.dump(stats, open(out / "stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
