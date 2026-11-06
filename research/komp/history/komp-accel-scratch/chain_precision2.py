"""FP32 emulation of geometry solve variants vs FP64: per-token LDL (Triton baseline algorithm),
Woodbury inverse chain, and square-root (W G W^T = I) chain. Reports p and z=G^-1 dp RMS rel error."""
import sys, math
import torch
import torch.nn.functional as F

tau, ridge = .5, 1e-4

def kern(x, y):
    return torch.exp((x @ y.transpose(-1, -2) - 1) / tau)

def ldl_solve(G, b):
    # Same pivot order/no pivoting as geometry._factor_solve, elementwise in G's dtype.
    a = G.clone(); m = a.shape[0]; L = torch.zeros_like(a); d = torch.zeros(m, dtype=a.dtype); b = b.clone()
    for j in range(m):
        col = a[:, j].clone(); d[j] = col[j]
        lj = torch.where(torch.arange(m) > j, col / col[j], torch.zeros_like(col))
        L[:, j] = lj
        mask = (torch.arange(m)[:, None] > j) & (torch.arange(m)[None, :] > j)
        a = a - torch.where(mask, col[:, None] * lj[None, :], torch.zeros_like(a))
        b = b - lj * b[j]
    x = b / d
    for j in reversed(range(m)):
        x[j] = x[j] - (L[j + 1:, j] * x[j + 1:]).sum()
    return x

def gauss_jordan_inv(G):
    a = G.clone(); m = a.shape[0]; inv = torch.eye(m, dtype=G.dtype)
    for j in range(m):
        piv = a[j, j]
        rowa = a[j] / piv; rowi = inv[j] / piv
        f = a[:, j].clone(); f[j] = 0
        a = a - f[:, None] * rowa[None]; inv = inv - f[:, None] * rowi[None]
        a[j] = rowa; inv[j] = rowi
    return inv

def chol_inv_sqrt(G):
    # W = L^{-1} from Cholesky (start of chunk), W G W^T = I.
    L = torch.linalg.cholesky(G)
    return torch.linalg.solve_triangular(L, torch.eye(G.shape[0], dtype=G.dtype), upper=False)

def sym_update(u, j, m, dtype):
    e = torch.zeros(m, dtype=dtype); e[j] = 1
    return u, e

def woodbury_step(P, G, Gold, j):
    m = G.shape[0]; d = G[:, j] - Gold[:, j]; u = d.clone(); u[j] = d[j] / 2
    e = torch.zeros(m, dtype=G.dtype); e[j] = 1
    Pu = P @ u; Pe = P[:, j]
    c = torch.tensor([[1 + Pu[j], Pe[j]], [u @ Pu, 1 + u @ Pe]], dtype=G.dtype)
    left = torch.stack([Pu, Pe], 1); right = torch.stack([Pe, Pu], 1)
    return P - left @ torch.linalg.inv(c) @ right.T

def sqrt_step(W, G, Gold, j):
    # W' = T W with T (I + a b^T + b a^T) T^T = I, a=W u, b=W e_j.
    m = G.shape[0]; d = G[:, j] - Gold[:, j]; u = d.clone(); u[j] = d[j] / 2
    a = W @ u; b = W[:, j]
    # Orthonormal basis of span{a,b}: Q (m x 2) via Gram-Schmidt; small matrix S = Q^T (I + ab^T + ba^T) Q.
    q1 = b / b.norm(); a2 = a - (q1 @ a) * q1; n2 = a2.norm()
    if n2 <= 1e-30 * a.norm().clamp_min(1e-30):
        Q = q1[:, None]
    else:
        Q = torch.stack([q1, a2 / n2], 1)
    A = Q.T @ a; B = Q.T @ b
    S = torch.eye(Q.shape[1], dtype=G.dtype) + A[:, None] * B[None] + B[:, None] * A[None]
    ev, vec = torch.linalg.eigh(S)
    Tsmall = vec @ torch.diag(ev.rsqrt() - 1) @ vec.T   # S^{-1/2} - I in the basis
    return W + Q @ (Tsmall @ (Q.T @ W))

def trajectory(k, slots, init):
    keys = F.normalize(init, dim=-1, eps=1e-6).clone(); kn = F.normalize(k, dim=-1, eps=1e-6)
    out = []
    for t in range(k.shape[0]):
        j = int(slots[t]); old = keys[j].clone(); keys = keys.clone(); keys[j] = kn[t]
        out.append((keys, old, j))
    return out

def experiment(case, t, m, d, Ls, seed=7):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(t, d, generator=g, dtype=torch.float64); init = torch.randn(m, d, generator=g, dtype=torch.float64)
    if case == 'near_duplicate':
        c = torch.randn(1, d, generator=g, dtype=torch.float64); init = c + .02 * init; k = c + .02 * k
    slots = torch.randint(m, (t,), generator=g) if case != 'cyclic' else torch.arange(t) % m
    dps = torch.randn(t, m, generator=g, dtype=torch.float64)
    traj = trajectory(k, slots, init)
    ref_p, ref_z = [], []
    for keys, old, j in traj:
        G = kern(keys, keys) + ridge * torch.eye(m, dtype=torch.float64)
        ref_p.append(torch.linalg.solve(G, kern(keys, old[None])[:, 0])); ref_z.append(torch.linalg.solve(G, dps[len(ref_z)]))
    ref_p, ref_z = torch.stack(ref_p), torch.stack(ref_z)
    rel = lambda x, r: float((x.double() - r).square().mean().sqrt() / r.square().mean().sqrt())
    f32 = [(kk.float(), o.float(), j) for kk, o, j in traj]
    eye = torch.eye(m)
    res = {}
    ps, zs = [], []
    for i, (keys, old, j) in enumerate(f32):
        G = kern(keys, keys) + ridge * eye
        ps.append(ldl_solve(G, kern(keys, old[None])[:, 0])); zs.append(ldl_solve(G, dps[i].float()))
    res['ldl'] = (rel(torch.stack(ps), ref_p), rel(torch.stack(zs), ref_z))
    for L in Ls:
        for name in ('woodbury', 'sqrt'):
            ps, zs = [], []
            for i, (keys, old, j) in enumerate(f32):
                G = kern(keys, keys) + ridge * eye
                if i % L == 0:
                    X = gauss_jordan_inv(G) if name == 'woodbury' else chol_inv_sqrt(G)
                else:
                    X = (woodbury_step if name == 'woodbury' else sqrt_step)(X, G, Gold, j)
                Gold = G
                P = X if name == 'woodbury' else X.T @ X
                b = kern(keys, old[None])[:, 0]
                if name == 'woodbury':
                    p = P @ b; z = P @ dps[i].float()
                else:
                    p = X.T @ (X @ b); z = X.T @ (X @ dps[i].float())
                ps.append(p); zs.append(z)
            res[f'{name}_L{L}'] = (rel(torch.stack(ps), ref_p), rel(torch.stack(zs), ref_z))
    print(f'{case:15s} t={t} m={m} d={d}', '  '.join(f'{a}: p {b[0]:.2e} z {b[1]:.2e}' for a, b in res.items()), flush=True)

for case in ('random', 'near_duplicate'):
    for (t, m, d) in ((9, 16, 32), (64, 16, 64), (64, 32, 64), (128, 64, 64)):
        for seed in (1, 2):
            experiment(case, t, m, d, (8, 16, 32), seed)
