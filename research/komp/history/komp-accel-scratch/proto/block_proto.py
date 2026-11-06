"""Float64 prototype of Block-KOMP: token reference, chunk-map form, and hand-derived map backward."""
import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
TAU, RIDGE = .5, 1e-4

def kern(x, y):
    return torch.exp((x @ y.transpose(-1, -2) - 1) / TAU)

def token_reference(q, k, v, g, beta, K0, S0, C):
    """Single head. q,k:[T,D] v:[T,V] g,beta:[T] K0:[M,D] S0:[M,V]."""
    qn, kn, keys = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6), F.normalize(K0, dim=-1, eps=1e-6)
    T, M = q.shape[0], K0.shape[0]
    coef, outs = S0, []
    for c0 in range(0, T, C):
        n = min(C, T - c0)
        U, S = keys, coef
        for s in range(n):
            t = c0 + s
            S = torch.exp(g[t]) * S
            err = v[t] - (S * kern(U, kn[t:t+1])).sum(0)
            U = torch.cat((U, kn[t:t+1])); S = torch.cat((S, (beta[t] * err)[None]))
            outs.append((S * kern(U, qn[t:t+1])).sum(0))
        keep, Sk, E, SE = U[n:n+M], S[n:n+M], U[:n], S[:n]
        P = torch.linalg.solve(kern(keep, keep) + RIDGE * torch.eye(M), kern(keep, E))
        keys, coef = keep, Sk + P @ SE
    return torch.stack(outs), keys, coef

def chunk_maps(Kold, Kn, Q, Knew, g, beta, n, C):
    """Per-chunk maps; Kold:[M,D] old dict, Kn:[C,D] new keys (rows>=n zero), Q:[C,D], Knew:[M,D] kept atoms."""
    M = Kold.shape[0]
    t = torch.arange(C); valid = (t < n).double()
    g = g * valid; beta = beta * valid
    Gc = torch.cumsum(g, 0); gam = torch.exp(Gc); GC = Gc[n-1]; gC = torch.exp(GC)
    D = torch.exp(Gc[:, None] - Gc[None, :]) * (t[:, None] > t[None, :])
    Dp = torch.exp(Gc[:, None] - Gc[None, :]) * (t[:, None] >= t[None, :])
    dend = torch.exp(GC - Gc) * valid
    Hk = kern(Kn, Kold) * valid[:, None]; Hn = kern(Kn, Kn) * valid[:, None] * valid[None, :]
    Rk = kern(Q, Kold) * valid[:, None]; Rn = kern(Q, Kn) * valid[:, None] * valid[None, :]
    L = beta[:, None] * (D * Hn)
    T = torch.linalg.inv(torch.eye(C) + L)
    WS = -T @ ((beta * gam)[:, None] * Hk)
    WV = T * beta[None, :]
    Rd = Dp * Rn
    OS = gam[:, None] * Rk + Rd @ WS
    OV = Rd @ WV
    a = torch.arange(M); c = torch.arange(C)
    mo = (a < n).double(); mn = ((M + c) < n).double()
    Kc = kern(Knew, Kold) * mo[None, :]; Nc = kern(Knew, Kn) * mn[None, :]
    Pinv = torch.linalg.inv(kern(Knew, Knew) + RIDGE * torch.eye(M))
    i = torch.arange(M)
    Jo = (a[None, :] == n + i[:, None]).double() + Pinv @ Kc
    Jn = ((M + c)[None, :] == n + i[:, None]).double() + Pinv @ Nc
    FS = gC * Jo + Jn @ (dend[:, None] * WS)
    FV = Jn @ (dend[:, None] * WV)
    return OS, OV, FS, FV

def chunk_forward(q, k, v, g, beta, K0, S0, C):
    qn, kn, K0n = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6), F.normalize(K0, dim=-1, eps=1e-6)
    T, M = q.shape[0], K0.shape[0]
    pool = torch.cat((K0n, kn))            # sequence position p -> pool row p+M
    S, outs = S0, []
    for c0 in range(0, T, C):
        n = min(C, T - c0)
        pad = lambda x: torch.cat((x, torch.zeros(C - x.shape[0], *x.shape[1:])))
        Kold = pool[c0:c0+M]; Kn = pad(pool[c0+M:c0+M+n]); Q = pad(qn[c0:c0+n]); Knew = pool[c0+n:c0+n+M]
        OS, OV, FS, FV = chunk_maps(Kold, Kn, Q, Knew, pad(g[c0:c0+n]), pad(beta[c0:c0+n]), n, C)
        V = pad(v[c0:c0+n])
        outs.append((OS @ S + OV @ V)[:n])
        S = FS @ S + FV @ V
    return torch.cat(outs), pool[T:T+M], S

def check(T=23, M=6, C=8, D=5, V=4, seed=0, zero=False):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen)
    q, k, v = r(T, D), r(T, D), r(T, V)
    g = -torch.rand(T, generator=gen) * .3
    beta = torch.rand(T, generator=gen)
    if zero:
        g[3] = -40.; beta[5] = 0.
    K0, S0 = r(M, D), r(M, V) * .3
    o1, k1, s1 = token_reference(q, k, v, g, beta, K0, S0, C)
    o2, k2, s2 = chunk_forward(q, k, v, g, beta, K0, S0, C)
    leaves = [x.clone().requires_grad_() for x in (q, k, v, g, beta, K0, S0)]
    leaves2 = [x.clone().requires_grad_() for x in (q, k, v, g, beta, K0, S0)]
    a = token_reference(*leaves, C); b = chunk_forward(*leaves2, C)
    w = [r(*x.shape) for x in a]
    ga = torch.autograd.grad(a, leaves, w); gb = torch.autograd.grad(b, leaves2, w)
    err = max(float((x - y).abs().max() / (x.abs().max() + 1e-30)) for x, y in zip(ga, gb))
    print(f'T={T} M={M} C={C} zero={zero} out {float((o1-o2).abs().max()):.1e} keys {float((k1-k2).abs().max()):.1e} '
          f'state {float((s1-s2).abs().max()):.1e} grads(rel) {err:.1e}')

for args in ((23, 6, 8), (40, 4, 8), (17, 8, 4), (64, 8, 16)):
    check(*args); check(*args, zero=True)

def chunk_maps_backward(Kold, Kn, Q, Knew, g, beta, n, C, dOS, dOV, dFS, dFV):
    """Hand-derived gradients of chunk_maps; returns dKold, dKn, dQ, dKnew, dg, dbeta (normalized-key space)."""
    M = Kold.shape[0]
    t = torch.arange(C); valid = (t < n).double()
    g = g * valid; beta = beta * valid
    Gc = torch.cumsum(g, 0); gam = torch.exp(Gc); GC = Gc[n-1]; gC = torch.exp(GC)
    lt = (t[:, None] > t[None, :]).double(); le = (t[:, None] >= t[None, :]).double()
    E = torch.exp(Gc[:, None] - Gc[None, :])
    D, Dp = E * lt, E * le
    dend = torch.exp(GC - Gc) * valid
    Hk = kern(Kn, Kold) * valid[:, None]; Hn = kern(Kn, Kn) * valid[:, None] * valid[None, :]
    Rk = kern(Q, Kold) * valid[:, None]; Rn = kern(Q, Kn) * valid[:, None] * valid[None, :]
    T = torch.linalg.inv(torch.eye(C) + beta[:, None] * (D * Hn))
    bg = beta * gam
    WS = -T @ (bg[:, None] * Hk); WV = T * beta[None, :]
    Rd = Dp * Rn
    a = torch.arange(M); c = torch.arange(C); i = torch.arange(M)
    mo = (a < n).double(); mn = ((M + c) < n).double()
    Kc = kern(Knew, Kold) * mo[None, :]; Nc = kern(Knew, Kn) * mn[None, :]
    Gk = kern(Knew, Knew)
    Pinv = torch.linalg.inv(Gk + RIDGE * torch.eye(M))
    PEo, PEn = Pinv @ Kc, Pinv @ Nc
    Jo = (a[None, :] == n + i[:, None]).double() + PEo
    Jn = ((M + c)[None, :] == n + i[:, None]).double() + PEn
    WSd, WVd = dend[:, None] * WS, dend[:, None] * WV
    # FS = gC Jo + Jn WSd ; FV = Jn WVd
    dgC = (dFS * Jo).sum()
    dJo = gC * dFS
    dJn = dFS @ WSd.T + dFV @ WVd.T
    dWSd = Jn.T @ dFS; dWVd = Jn.T @ dFV
    ddend = (dWSd * WS).sum(1) + (dWVd * WV).sum(1)
    dWS = dend[:, None] * dWSd; dWV = dend[:, None] * dWVd
    Zo, Zn = Pinv @ (dJo * mo[None, :]), Pinv @ (dJn * mn[None, :])
    dKc, dNc = Zo * mo[None, :], Zn * mn[None, :]
    dGk = -(Zo @ PEo.T + Zn @ PEn.T)
    # OS = gam Rk + Rd WS ; OV = Rd WV
    dRk = gam[:, None] * dOS
    dgam = (dOS * Rk).sum(1)
    dRd = dOS @ WS.T + dOV @ WV.T
    dWS = dWS + Rd.T @ dOS; dWV = dWV + Rd.T @ dOV
    dRn = dRd * Dp; dE = dRd * Rn * le
    # WS = -T diag(bg) Hk ; WV = T diag(beta)
    TtdWS = T.T @ dWS
    dHk = -bg[:, None] * TtdWS
    dbg = -(TtdWS * Hk).sum(1)
    dbeta = dbg * gam + (dWV * T).sum(0)
    dgam = dgam + dbg * beta
    dT = -dWS @ (bg[:, None] * Hk).T + dWV * beta[None, :]
    dL = -(T.T @ dT @ T.T) * lt
    dbeta = dbeta + (dL * D * Hn).sum(1)
    dHn = beta[:, None] * dL * D
    dE = dE + beta[:, None] * dL * Hn * lt
    # decay: gam_t = e^{Gc_t}; E[t,s] = e^{Gc_t-Gc_s}; dend_s = e^{GC-Gc_s}; gC = e^{GC}
    dEE = dE * E
    dGc = dgam * gam + dEE.sum(1) - dEE.sum(0) - ddend * dend
    dGC = dgC * gC + (ddend * dend).sum()
    dGc = dGc + dGC * (t == n - 1).double()
    dg = torch.flip(torch.cumsum(torch.flip(dGc, [0]), 0), [0]) * valid
    dbeta = dbeta * valid
    # kernels: kappa = exp((cos-1)/tau) -> dcos = dkappa * kappa / tau
    w = lambda dk, kv: dk * kv / TAU
    cHk = w(dHk * valid[:, None], Hk); cHn = w(dHn * valid[:, None] * valid[None, :], Hn)
    cRk = w(dRk * valid[:, None], Rk); cRn = w(dRn * valid[:, None] * valid[None, :], Rn)
    cKc = w(dKc, Kc); cNc = w(dNc, Nc); cG = w(dGk, Gk)
    dKn = cHk @ Kold + (cHn + cHn.T) @ Kn + cRn.T @ Q + cNc.T @ Knew
    dKold = cHk.T @ Kn + cRk.T @ Q + cKc.T @ Knew
    dQ = cRk @ Kold + cRn @ Kn
    dKnew = cKc @ Kold + cNc @ Kn + (cG + cG.T) @ Knew
    return dKold, dKn, dQ, dKnew, dg, dbeta

def check_maps(M=6, C=8, D=5, n=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen)
    unit = lambda x: F.normalize(x, dim=-1)
    Kold, Q, Knew = unit(r(M, D)), unit(r(C, D)), unit(r(M, D))
    Kn = unit(r(C, D)) * (torch.arange(C) < n).double()[:, None]
    g = -torch.rand(C, generator=gen) * .3; beta = torch.rand(C, generator=gen)
    leaves = [x.clone().requires_grad_() for x in (Kold, Kn, Q, Knew, g, beta)]
    maps = chunk_maps(*leaves[:4], leaves[4], leaves[5], n, C)
    dm = [r(*x.shape) for x in maps]
    ref = torch.autograd.grad(maps, leaves, dm)
    mine = chunk_maps_backward(Kold, Kn, Q, Knew, g, beta, n, C, *dm)
    names = ('Kold', 'Kn', 'Q', 'Knew', 'g', 'beta')
    errs = {nm: float((a - b).abs().max() / (a.abs().max() + 1e-30)) for nm, a, b in zip(names, ref, mine)}
    print(f'maps M={M} C={C} n={n}', {k: f'{v:.1e}' for k, v in errs.items()})

for M, C, n in ((6, 8, 8), (6, 8, 5), (4, 8, 8), (4, 8, 6), (8, 4, 4), (8, 4, 2), (16, 16, 16)):
    check_maps(M, C, 7, n)
