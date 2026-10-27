import math
import torch

BASE_MVA = 100.0
BT_PQ, BT_PV, BT_REF = 0, 1, 2
PD, QD, PG, QG, VM, VA = range(6)           # columns of the node state vector z
Z_SHIFT = torch.tensor([0., 0., 0., 0., 1.0, 0.0])
Z_SCALE = torch.tensor([1., 1., 1., 1., 0.05, 0.3])   # normalisation (p.u. / rad)


class GraphBatch:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to(self, dev):
        return GraphBatch(**{k: (v.to(dev) if torch.is_tensor(v) else v)
                             for k, v in self.__dict__.items()})


def collate(samples):
    keys = ["z", "bt", "gs", "bs", "kv", "br"]
    out = {k: torch.cat([s[k] for s in samples]) for k in keys}
    ei, batch, off = [], [], 0
    for g, s in enumerate(samples):
        n = s["z"].shape[0]
        ei.append(s["ei"] + off)
        batch.append(torch.full((n,), g, dtype=torch.long))
        off += n
    out["ei"] = torch.cat(ei, 1)
    out["batch"] = torch.cat(batch)
    out["num_graphs"] = len(samples)
    return GraphBatch(**out)


# --------------------------------------------------------------------------
# Differentiable AC power-flow physics (MATPOWER branch model)
# --------------------------------------------------------------------------
def ac_physics(vm, va, g):
    """Returns bus P/Q injections computed from (vm, va) and the from-side
    active branch flow. All in p.u. on a 100 MVA base. Differentiable."""
    f, t = g.ei
    dt = vm.dtype
    r, x, b, tap, sh = [c.to(dt) for c in g.br.unbind(-1)]
    V = torch.polar(vm, va)
    ys = 1.0 / torch.complex(r, x)
    bc = torch.complex(torch.zeros_like(b), b / 2)
    a = torch.polar(tap, sh)
    Yff = (ys + bc) / (tap * tap)
    Yft = -ys / torch.conj(a)
    Ytf = -ys / a
    Ytt = ys + bc
    Vf, Vt = V[f], V[t]
    Sf = Vf * torch.conj(Yff * Vf + Yft * Vt)
    St = Vt * torch.conj(Ytf * Vf + Ytt * Vt)
    n = vm.shape[0]
    P = vm.new_zeros(n).index_add(0, f, Sf.real).index_add(0, t, St.real)
    Q = vm.new_zeros(n).index_add(0, f, Sf.imag).index_add(0, t, St.imag)
    P = P + vm ** 2 * g.gs.to(dt)             # bus shunt  S = |V|^2 conj(Ysh)
    Q = Q - vm ** 2 * g.bs.to(dt)
    return P, Q, Sf.real


def full_state_residual(z, g):
    """P/Q mismatch at every bus for a full state vector z (physical units)."""
    P, Q, _ = ac_physics(z[:, VM], z[:, VA], g)
    return P - (z[:, PG] - z[:, PD]), Q - (z[:, QG] - z[:, QD])


# --------------------------------------------------------------------------
# Masks.  mask[i, c] = True  ->  feature c at bus i is hidden from the model
# --------------------------------------------------------------------------
def pf_mask(g):
    """The unknowns of a power-flow problem: Vm,Va @PQ ; Qg,Va @PV ; Pg,Qg @REF."""
    m = torch.zeros(g.z.shape, dtype=torch.bool, device=g.z.device)
    pq, pv, ref = g.bt == BT_PQ, g.bt == BT_PV, g.bt == BT_REF
    m[pq, VM] = True; m[pq, VA] = True
    m[pv, QG] = True; m[pv, VA] = True
    m[ref, PG] = True; m[ref, QG] = True
    return m


def random_mask(g, p):
    valid = torch.ones(g.z.shape, dtype=torch.bool, device=g.z.device)
    nogen = g.bt == BT_PQ
    valid[nogen, PG] = False; valid[nogen, QG] = False   # structurally zero
    valid[g.bt == BT_REF, VA] = False                    # reference angle = 0
    return (torch.rand(g.z.shape, device=g.z.device) < p) & valid


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def _slog(v):
    return torch.sign(v) * torch.log1p(v.abs())


def edge_features(g):
    r, x, b, tap, sh = g.br.unbind(-1)
    den = r * r + x * x + 1e-9
    gs_, bs_ = _slog(r / den), _slog(-x / den)
    fwd = torch.stack([gs_, bs_, b, tap - 1, sh, torch.ones_like(r)], -1)
    rev = torch.stack([gs_, bs_, b, 1 / tap - 1, -sh, torch.zeros_like(r)], -1)
    return torch.cat([fwd, rev])


def node_features(g, mask):
    sh, sc = Z_SHIFT.to(g.z.device), Z_SCALE.to(g.z.device)
    zn = (g.z - sh) / sc
    zin = torch.where(mask, torch.zeros_like(zn), zn)
    oh = torch.nn.functional.one_hot(g.bt, 3).float()
    return torch.cat([oh, zin, mask.float(), g.gs[:, None], g.bs[:, None],
                      (g.kv / 345.0)[:, None]], -1)


def reconstruct(out, g, mask):
    """Normalised network output -> physical z; observed entries are kept as-is."""
    sh, sc = Z_SHIFT.to(out.device), Z_SCALE.to(out.device)
    return torch.where(mask, out * sc + sh, g.z)
