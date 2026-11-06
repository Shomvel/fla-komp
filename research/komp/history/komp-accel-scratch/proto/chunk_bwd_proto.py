"""Float64 prototype: matrix-form chunk forward/backward for the scalar-decay rank-2 slot recurrence.
Checked against autograd through the token loop. Tokens are 1-indexed in the math; arrays use index s-1."""
import torch
torch.set_default_dtype(torch.float64)

def token_loop(S0, h, p, r, v, a, b, slots):
    C, M = h.shape
    S = S0
    outs = []
    for s in range(C):
        e = torch.nn.functional.one_hot(slots[s], M).double()
        old = S[slots[s]]
        pre = a[s] * S
        oldp = a[s] * old
        err = v[s] - (h[s][:, None] * pre).sum(0)
        S = pre + (p[s] - e)[:, None] * oldp[None] + b[s] * e[:, None] * err[None]
        outs.append(S.T @ r[s])
    return torch.stack(outs), S

def decay(a):
    # d[x, y] = prod_{i=y+1..x} a_i for x >= y (x,y in 0..C), else 0. a is 1-indexed via a[i-1].
    C = a.shape[0]
    d = torch.zeros(C+1, C+1)
    for x in range(C+1):
        for y in range(x+1):
            d[x, y] = torch.prod(a[y:x]) if x > y else 1.
    return d

def chunk(S0, h, p, r, v, a, b, slots, dO=None, dSC=None):
    C, M = h.shape
    d = decay(a)
    E = torch.nn.functional.one_hot(slots, M).double()           # C x M
    # pseudo rows n = 2(s-1) + k-1 ; x_{s1}=p-e, x_{s2}=-b e ; y_{s1}=e, y_{s2}=h
    X = torch.stack([p - E, -b[:, None] * E], 1).reshape(2*C, M).T  # M x 2C
    Y = torch.stack([E, h], 1).reshape(2*C, M).T                    # M x 2C
    W = (b[:, None] * E).T                                          # M x C
    R = r.T                                                         # M x C
    tok = torch.arange(1, C+1).repeat_interleave(2)                 # token (1-indexed) of pseudo row
    s_ = torch.arange(1, C+1)
    def D(x, y):  # elementwise decay for index tensors (broadcast), 0 if x<y
        return d[x.clamp(0, C), y.clamp(0, C)] * (x >= y)
    # forward system
    Lm = D(tok[:, None]-1, tok[None, :]-1) * (tok[:, None] > tok[None, :]) * (Y.T @ X)
    T = torch.linalg.inv(torch.eye(2*C) - Lm)
    Bs = D(tok-1, torch.zeros_like(tok))[:, None] * Y.T              # 2C x M
    Bv = D(tok[:, None]-1, s_[None, :]) * (tok[:, None] > s_[None, :]) * (Y.T @ W)
    U = T @ (Bs @ S0 + Bv @ v)
    Os0 = D(s_, torch.zeros_like(s_))[:, None] * R.T
    Ox = D(s_[:, None], tok[None, :]-1) * (s_[:, None] >= tok[None, :]) * (R.T @ X)
    Ow = D(s_[:, None], s_[None, :]) * (s_[:, None] >= s_[None, :]) * (R.T @ W)
    O = Os0 @ S0 + Ox @ U + Ow @ v
    Xf = X * D(torch.full_like(tok, C), tok-1)[None, :]
    Wf = W * D(torch.full_like(s_, C), s_)[None, :]
    SC = d[C, 0] * S0 + Xf @ U + Wf @ v
    if dO is None:
        return O, SC
    # ---------------- backward ----------------
    Z = T.T @ (Xf.T @ dSC + Ox.T @ dO)                              # Zhat, 2C x V
    dS0 = d[C, 0] * dSC + Os0.T @ dO + Bs.T @ Z
    al = a
    # coefficient matrices
    A1 = D(s_[None, :], tok[:, None]-1) * (tok[:, None] <= s_[None, :])        # [n', s] d(s, m-1), m<=s
    A2 = D(s_[None, :], s_[:, None]) * (s_[:, None] <= s_[None, :])            # [m, s] d(s, m), m<=s
    dR = (S0 @ dO.T) * D(s_, torch.zeros_like(s_))[None, :] + X @ (A1 * (U @ dO.T)) + W @ (A2 * (v @ dO.T))
    tokn = tok                                                                   # column n -> token s of n
    B1 = D(s_[:, None], tokn[None, :]) * (s_[:, None] >= tokn[None, :])        # [q, n] d(q, s), q>=s
    B2 = D(tok[:, None]-1, tokn[None, :]) * (tok[:, None] > tokn[None, :])     # [n', n] d(q-1, s), q>s
    GU = (dSC @ U.T) * D(torch.full_like(tokn, C), tokn)[None, :] + R @ (B1 * (dO @ U.T)) + Y @ (B2 * (Z @ U.T))
    dX = GU * al[tokn-1][None, :]
    C1 = D(tokn[None, :]-1, tok[:, None]-1) * (tok[:, None] < tokn[None, :])   # [n', n] d(s-1, m-1), m<s
    C2 = D(tokn[None, :]-1, s_[:, None]) * (s_[:, None] < tokn[None, :])       # [m, n] d(s-1, m), m<s
    dY = (S0 @ Z.T) * D(tokn-1, torch.zeros_like(tokn))[None, :] + X @ (C1 * (U @ Z.T)) + W @ (C2 * (v @ Z.T))
    # rows j of G_s
    B1s = D(s_[:, None], s_[None, :]) * (s_[:, None] >= s_[None, :])           # [q, s] d(q,s)
    B2s = D(tok[:, None]-1, s_[None, :]) * (tok[:, None] > s_[None, :])        # [n', s] d(q-1, s)
    Gj = (E @ dSC) * D(torch.full_like(s_, C), s_)[:, None] + (B1s.T * (E @ R)) @ dO + (B2s.T * (E @ Y)) @ Z
    U2 = U[1::2]; U1 = U[0::2]
    dv = b[:, None] * Gj
    db = (Gj * (v - al[:, None] * U2)).sum(1)
    dp = dX[:, 0::2].T
    dh = dY[:, 1::2].T
    dr = dR.T
    # alpha: <G_s, S_{s-1}> + sum_k (x_sk^T G_s) . U_sk
    Zraw = (X.T @ dSC) * D(torch.full_like(tokn, C), tokn)[:, None] + (B1.T * (X.T @ R)) @ dO + (B2.T * (X.T @ Y)) @ Z
    term2 = (Zraw * U).reshape(C, 2, -1).sum((1, 2))
    gc0 = D(torch.full_like(s_, C), s_)                       # C
    gcr = B1s.T                                               # [s, q]
    gcy = B2s.T                                               # [s, n']
    sc0 = D(s_-1, torch.zeros_like(s_))                       # C
    scx = D(s_[:, None]-1, tok[None, :]-1) * (tok[None, :] < s_[:, None])   # [s, n''] d(s-1, m-1), m<s
    scw = D(s_[:, None]-1, s_[None, :]) * (s_[None, :] < s_[:, None])       # [s, m] d(s-1, m), m<s
    p00 = (dSC * S0).sum()
    p0x = ((X.T @ dSC) * U).sum(1)
    p0w = ((W.T @ dSC) * v).sum(1)
    pr0 = ((R.T @ S0) * dO).sum(1)
    prx = (R.T @ X) * (dO @ U.T)
    prw = (R.T @ W) * (dO @ v.T)
    py0 = ((Y.T @ S0) * Z).sum(1)
    pyx = (Y.T @ X) * (Z @ U.T)
    pyw = (Y.T @ W) * (Z @ v.T)
    term1 = gc0 * (p00 * sc0 + scx @ p0x + scw @ p0w) \
        + (gcr * (sc0[:, None] * pr0[None, :] + scx @ prx.T + scw @ prw.T)).sum(1) \
        + (gcy * (sc0[:, None] * py0[None, :] + scx @ pyx.T + scw @ pyw.T)).sum(1)
    da = term1 + term2
    return O, SC, dict(S0=dS0, h=dh, p=dp, r=dr, v=dv, a=da, b=db)

def check(C=7, M=5, V=4, zero_gate=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g)
    S0, h, p, r, v = rnd(M, V), rnd(C, M).abs(), rnd(C, M), rnd(C, M), rnd(C, V)
    a = torch.rand(C, generator=g)*.5+.5; b = torch.rand(C, generator=g)
    if zero_gate:
        a[2] = 0.; b[4] = 0.; a[5] = 1.
    slots = torch.randint(M, (C,), generator=g); slots[3] = slots[2]
    leaves = [x.clone().requires_grad_() for x in (S0, h, p, r, v, a, b)]
    O, SC = token_loop(*leaves, slots)
    dO, dSC = rnd(C, V), rnd(M, V)
    grads = torch.autograd.grad((O, SC), leaves, (dO, dSC))
    O2, SC2, mine = chunk(S0, h, p, r, v, a, b, slots, dO, dSC)
    errs = {'O': float((O-O2).abs().max()), 'SC': float((SC-SC2).abs().max())}
    for name, gref in zip(('S0', 'h', 'p', 'r', 'v', 'a', 'b'), grads):
        errs[name] = float((gref - mine[name]).abs().max() / (gref.abs().max() + 1e-30))
    print(C, M, V, zero_gate, {k: f'{e:.1e}' for k, e in errs.items()})

for zg in (False, True):
    for seed in range(2):
        check(zero_gate=zg, seed=seed)
check(C=16, M=32, V=8, zero_gate=True, seed=3)
