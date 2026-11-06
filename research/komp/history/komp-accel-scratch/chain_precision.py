"""CPU FP32 emulation: per-chunk inverse chain (rank-2 Woodbury + Newton-Schulz) vs per-token LDL solve."""
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, sys.argv[1])
from komp_slot.reference import prepare_geometry

tau, ridge = .5, 1e-4

def kern(x, y):
    return torch.exp((x @ y.transpose(-1, -2) - 1) / tau)

def dicts(q, k, slots, init):
    """Per token: post-write dictionary, removed atom, slot (b=0,h=0 only)."""
    keys = F.normalize(init[0, 0], dim=-1, eps=1e-6).clone()
    kn = F.normalize(k[0, :, 0], dim=-1, eps=1e-6)
    out = []
    for t in range(k.shape[1]):
        j = int(slots[0, t, 0]); old = keys[j].clone()
        keys = keys.clone(); keys[j] = kn[t]
        out.append((keys.clone(), old, j))
    return out

def chain(seq, L, ns_steps, dtype=torch.float32):
    m = seq[0][0].shape[0]; eye = torch.eye(m, dtype=dtype)
    ps = []
    for t, (keys, old, j) in enumerate(seq):
        keys = keys.to(dtype); old = old.to(dtype)
        G = kern(keys, keys) + ridge * eye
        if t % L == 0:
            P = torch.linalg.inv(G)  # stands in for in-kernel Gauss-Jordan
        else:
            Gold = Gprev
            d = G[:, j] - Gold[:, j]
            u = d.clone(); u[j] = d[j] / 2
            e = eye[:, j]
            U = torch.stack([u, e], 1); V = torch.stack([e, u], 1)
            PU = P @ U
            cap = torch.eye(2, dtype=dtype) + V.T @ PU
            P = P - PU @ torch.linalg.inv(cap) @ (V.T @ P)
            for _ in range(ns_steps):
                P = P + P @ (eye - G @ P)
            P = (P + P.T) / 2
        Gprev = G
        ps.append(P @ kern(keys, old[None])[:, 0])
    return torch.stack(ps)

def run(case, t=64, m=32, d=64, L=16):
    torch.manual_seed(7)
    q = torch.randn(1, t, 1, d, dtype=torch.float64); k = torch.randn_like(q)
    init = torch.randn(1, 1, m, d, dtype=torch.float64)
    if case == 'near_duplicate':
        c = torch.randn(1, 1, 1, d, dtype=torch.float64)
        init = c + .02 * init; k = c.transpose(1, 2) + .02 * k
    slots = torch.randint(m, (1, t, 1)) if case != 'cyclic' else (torch.arange(t) % m)[None, :, None]
    _, p64, _, _ = prepare_geometry(q, k, slots, init, tau, ridge)
    ref = p64[0, :, 0]
    _, p32, _, _ = prepare_geometry(q.float(), k.float(), slots, init.float(), tau, ridge)
    rel = lambda x: float((x.double() - ref).square().mean().sqrt() / ref.square().mean().sqrt())
    seq = dicts(q, k, slots, init)
    row = {'torch_solve_fp32': rel(p32[0, :, 0])}
    for ns in (0, 1, 2):
        row[f'chain_L{L}_ns{ns}'] = rel(chain(seq, L, ns))
    print(case, m, {a: '%.2e' % b for a, b in row.items()})

for case in ('random', 'cyclic', 'near_duplicate'):
    for m in (16, 32, 64):
        run(case, m=m)
