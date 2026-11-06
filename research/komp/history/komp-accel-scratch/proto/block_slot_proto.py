"""Float64: Block-KOMP with per-slot decay. Token reference vs chunk-map form vs hand backward of maps."""
import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
TAU, RIDGE = .5, 1e-4

def kern(x, y):
    return torch.exp((x @ y.transpose(-1, -2) - 1) / TAU)

def token_reference(q, k, v, g, beta, K0, S0, C):
    """g:[T,M] log decay per ring slot; atom at sequence position p lives in slot p mod M."""
    qn, kn, keys = F.normalize(q, dim=-1), F.normalize(k, dim=-1), F.normalize(K0, dim=-1)
    T, M = q.shape[0], K0.shape[0]
    slots = torch.arange(M)                     # initial atoms: positions -M..-1
    coef, outs = S0, []
    for c0 in range(0, T, C):
        n = min(C, T - c0)
        U, S, sl = keys, coef, slots
        for s in range(n):
            t = c0 + s
            S = torch.exp(g[t, sl])[:, None] * S
            err = v[t] - (S * kern(U, kn[t:t+1])).sum(0)
            U = torch.cat((U, kn[t:t+1])); S = torch.cat((S, (beta[t] * err)[None])); sl = torch.cat((sl, torch.tensor([t % M])))
            outs.append((S * kern(U, qn[t:t+1])).sum(0))
        keep, E = U[n:n+M], U[:n]
        P = torch.linalg.solve(kern(keep, keep) + RIDGE * torch.eye(M), kern(keep, E))
        keys, coef, slots = keep, S[n:n+M] + P @ S[:n], sl[n:n+M]
    return torch.stack(outs), keys, coef

def decays(g, n, C, M, o):
    """g:[C,M] (rows >= n zero). Returns Gam[C,M] by old atom, D, Dp [C,C], dend[C], gend[M]."""
    t = torch.arange(C); valid = (t < n).double()
    Gc = torch.cumsum(g * valid[:, None], 0)
    r_old = (o + torch.arange(M)) % M; r_new = (o + t) % M
    Gam = torch.exp(Gc[:, r_old])
    GcS = Gc[:, r_new]; dS = GcS.diagonal()
    le = (t[:, None] >= t[None, :]); lt = (t[:, None] > t[None, :])
    Ex = torch.exp(torch.where(le, GcS - dS[None, :], torch.zeros(())))
    dend = torch.exp(GcS[n-1] - dS) * valid
    gend = Gam[n-1]
    return Gam, Ex * lt, Ex * le, dend, gend

def chunk_maps(Kold, Kn, Q, Knew, g, beta, n, C, o):
    M = Kold.shape[0]
    t = torch.arange(C); valid = (t < n).double(); beta = beta * valid
    Gam, D, Dp, dend, gend = decays(g, n, C, M, o)
    Hk = kern(Kn, Kold) * valid[:, None]; Hn = kern(Kn, Kn) * valid[:, None] * valid[None, :]
    Rk = kern(Q, Kold) * valid[:, None]; Rn = kern(Q, Kn) * valid[:, None] * valid[None, :]
    T = torch.linalg.inv(torch.eye(C) + beta[:, None] * (D * Hn))
    WS = -T @ (beta[:, None] * Gam * Hk)
    WV = T * beta[None, :]
    Rd = Dp * Rn
    OS = Gam * Rk + Rd @ WS
    OV = Rd @ WV
    a = torch.arange(M); c = torch.arange(C); i = torch.arange(M)
    mo = (a < n).double(); mn = ((M + c) < n).double()
    Pinv = torch.linalg.inv(kern(Knew, Knew) + RIDGE * torch.eye(M))
    Jo = (a[None, :] == n + i[:, None]).double() + Pinv @ (kern(Knew, Kold) * mo[None, :])
    Jn = ((M + c)[None, :] == n + i[:, None]).double() + Pinv @ (kern(Knew, Kn) * mn[None, :])
    FS = Jo * gend[None, :] + Jn @ (dend[:, None] * WS)
    FV = Jn @ (dend[:, None] * WV)
    return OS, OV, FS, FV

def chunk_forward(q, k, v, g, beta, K0, S0, C):
    qn, kn, K0n = F.normalize(q, dim=-1), F.normalize(k, dim=-1), F.normalize(K0, dim=-1)
    T, M = q.shape[0], K0.shape[0]
    pool = torch.cat((K0n, kn)); S, outs = S0, []
    for c0 in range(0, T, C):
        n = min(C, T - c0)
        pad = lambda x: torch.cat((x, torch.zeros(C - x.shape[0], *x.shape[1:])))
        maps = chunk_maps(pool[c0:c0+M], pad(pool[c0+M:c0+M+n]), pad(qn[c0:c0+n]), pool[c0+n:c0+n+M],
                          pad(g[c0:c0+n]), pad(beta[c0:c0+n]), n, C, c0 % M)
        V = pad(v[c0:c0+n]); outs.append((maps[0] @ S + maps[1] @ V)[:n]); S = maps[2] @ S + maps[3] @ V
    return torch.cat(outs), pool[T:T+M], S

def map_backward_decay(g, beta, n, C, M, o, dGam, dD, dDp, ddend, dgend):
    """Gradient wrt g[C,M] from gradients wrt the decay quantities (scatter by ring slot)."""
    t = torch.arange(C); valid = (t < n).double()
    Gam, D, Dp, dend, gend = decays(g, n, C, M, o)
    r_old = (o + torch.arange(M)) % M; r_new = (o + t) % M
    u_of_r = (torch.arange(M) - o) % M
    Rnew = F.one_hot(r_new, M).double()
    dGc = (dGam * Gam)[:, u_of_r]
    Y = dD * D + dDp * Dp
    dGc = dGc + Y @ Rnew - Y.sum(0)[:, None] * Rnew
    Z = ddend * dend
    dGc = dGc - Z[:, None] * Rnew
    rowC = Z @ Rnew + (dgend * gend)[u_of_r]
    dGc = dGc + (t == n - 1).double()[:, None] * rowC[None, :]
    dGc = dGc * valid[:, None]
    return torch.flip(torch.cumsum(torch.flip(dGc, [0]), 0), [0])

def check(T=23, M=6, C=8, D=5, V=4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen)
    q, k, v = r(T, D), r(T, D), r(T, V)
    g = -torch.rand(T, M, generator=gen) * .4; beta = torch.rand(T, generator=gen)
    K0, S0 = r(M, D), r(M, V) * .3
    leaves = [x.clone().requires_grad_() for x in (q, k, v, g, beta, K0, S0)]
    leaves2 = [x.clone().requires_grad_() for x in (q, k, v, g, beta, K0, S0)]
    a = token_reference(*leaves, C); b = chunk_forward(*leaves2, C)
    w = [r(*x.shape) for x in a]
    ga = torch.autograd.grad(a, leaves, w); gb = torch.autograd.grad(b, leaves2, w)
    fe = max(float((x - y).abs().max()) for x, y in zip(a, b))
    ge = max(float((x - y).abs().max() / (x.abs().max() + 1e-30)) for x, y in zip(ga, gb))
    # decay-scatter backward vs autograd through decays()
    n = min(C, T); o = 3 % M
    gl = (-torch.rand(C, M, generator=gen) * .4).requires_grad_()
    outs = decays(gl, n, C, M, o)
    cot = [r(*x.shape) for x in outs]
    ref = torch.autograd.grad(outs, gl, cot)[0]
    mine = map_backward_decay(gl.detach(), None, n, C, M, o, *cot)
    de = float((ref - mine).abs().max() / ref.abs().max())
    print(f'T={T} M={M} C={C} fwd {fe:.1e} grads(rel) {ge:.1e} decay-scatter(rel) {de:.1e}')

for args in ((23, 6, 8), (40, 4, 8), (17, 8, 4), (64, 8, 16), (30, 5, 16)):
    check(*args)
