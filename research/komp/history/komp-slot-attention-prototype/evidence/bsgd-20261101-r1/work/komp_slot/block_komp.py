"""Block-KOMP in Triton: chunk maps from key geometry, two-level affine scan, chunk-local backward.

Per chunk (C tokens, M carried atoms, FIFO dictionary = contiguous pool rows):
  Hk=k(Kn,Kold), Hn=k(Kn,Kn), Rk=k(Q,Kold), Rn=k(Q,Kn); g cumsum gives gamma and window decays.
  T=(I+diag(beta)(D o Hn))^-1, WS=-T diag(beta gamma) Hk, WV=T diag(beta)      (new-slot writes W=WS S0+WV V)
  OS=diag(gamma) Rk + (D' o Rn) WS, OV=(D' o Rn) WV                              (outputs O=OS S0+OV V)
  P=(k(Knew,Knew)+ridge I)^-1, J=[sel+P k(Knew,E)]                               (joint projection of evicted atoms)
  FS=gamma_C J_old + J_new diag(dend) WS, FV=J_new diag(dend) WV                  (next state S0'=FS S0+FV V)
Only the M x M Gram inverse is a serial elimination, once per chunk. Everything else is a matrix product.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from .geometry import normalize_kernel, normalize_bwd_kernel
from .geometry_chunk import _sweep_inverse
from .triton_chunk import _inverse
from .recurrence import output_kernel
from .scan import affine_scan, value_offset_kernel, output_offset_kernel
from .utils import zero

ATT = tl.constexpr('tf32')        # attention-like kernel matrices and value products
PROJ = tl.constexpr('tf32x3')     # projection Gram, its inverse products and the UT solve
WARPS = {'maps': 8, 'mapgrad': 4, 'maps_bwd': 8, 'keygrad': 4, 'offset': 4, 'output': 4}


@tr.jit
def _rows(Pool, row0, nrow, R: tl.constexpr, D: tl.constexpr, d0, BDK: tl.constexpr):
    r = tl.arange(0, R)
    d = d0+tl.arange(0, BDK)
    return tl.load(Pool+(row0+r[:, None])*D+d[None, :], (r[:, None] < nrow) & (d[None, :] < D), 0)


@tr.jit
def _ex(cos, TAU: tl.constexpr):
    return tl.exp((cos-1.)/TAU)


@tr.jit
def _chunk_setup(G, Beta, chunk, bh, T: tl.constexpr, H: tl.constexpr, C: tl.constexpr):
    b, head = bh//H, bh % H
    t = tl.arange(0, C)
    t0 = chunk*C
    n = tl.minimum(C, T-t0)
    valid = t < n
    tok = (b*T+t0+t)*H+head
    g = tl.load(G+tok, valid, 0).to(tl.float32)
    beta = tl.load(Beta+tok, valid, 0).to(tl.float32)
    return t, n, valid, tok, g, beta


@tr.jit
def _tiles(QN, Pool, rbase, n, tok, valid, d0, M: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
           BM: tl.constexpr, BDK: tl.constexpr):
    # Column block [d0, d0+BDK) of the old dictionary, new keys, kept atoms and queries.
    d = d0+tl.arange(0, BDK)
    Kold = _rows(Pool, rbase, M, BM, D, d0, BDK)
    Kn = _rows(Pool, rbase+M, n, C, D, d0, BDK)
    Knew = _rows(Pool, rbase+n, M, BM, D, d0, BDK)
    Q = tl.load(QN+tok[:, None]*D+d[None, :], valid[:, None] & (d[None, :] < D), 0)
    return Kold, Kn, Knew, Q


@tr.jit
def _cosines(QN, Pool, rbase, n, tok, valid, M: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
             BM: tl.constexpr, BDK: tl.constexpr):
    hk = tl.zeros((C, BM), tl.float32)
    hn = tl.zeros((C, C), tl.float32)
    rk = tl.zeros((C, BM), tl.float32)
    rn = tl.zeros((C, C), tl.float32)
    kc = tl.zeros((BM, BM), tl.float32)
    nc = tl.zeros((BM, C), tl.float32)
    gk = tl.zeros((BM, BM), tl.float32)
    for d0 in range(0, D, BDK):
        Kold, Kn, Knew, Q = _tiles(QN, Pool, rbase, n, tok, valid, d0, M, D, C, BM, BDK)
        hk += tl.dot(Kn, tl.trans(Kold), input_precision=ATT)
        hn += tl.dot(Kn, tl.trans(Kn), input_precision=ATT)
        rk += tl.dot(Q, tl.trans(Kold), input_precision=ATT)
        rn += tl.dot(Q, tl.trans(Kn), input_precision=ATT)
        kc += tl.dot(Knew, tl.trans(Kold), input_precision=PROJ)
        nc += tl.dot(Knew, tl.trans(Kn), input_precision=PROJ)
        gk += tl.dot(Knew, tl.trans(Knew), input_precision=PROJ)
    return hk, hn, rk, rn, kc, nc, gk


@tr.jit
def _decay(g, valid, n, C: tl.constexpr):
    t = tl.arange(0, C)
    gc = tl.cumsum(g, 0)
    le = t[:, None] >= t[None, :]
    lt = t[:, None] > t[None, :]
    E = tl.exp(tl.where(le, gc[:, None]-gc[None, :], 0.))
    gC = tl.sum(tl.where(t == n-1, gc, 0.), 0)
    dend = tl.where(valid, tl.exp(gC-gc), 0.)
    return gc, tl.exp(gc), E, tl.where(lt, E, 0.), tl.where(le, E, 0.), gC, dend, le, lt


@tr.jit
def block_maps_kernel(QN, Pool, G, Beta, OS, OV, FS, FV, PINV,
                      T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                      NC: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr, TAU: tl.constexpr, RIDGE: tl.constexpr,
                      SAVE: tl.constexpr, BDK: tl.constexpr):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, g, beta = _chunk_setup(G, Beta, chunk, bh, T, H, C)
    gc, gam, E, Dl, Dp, gC, dend, le, lt = _decay(g, valid, n, C)
    a = tl.arange(0, BM)
    am = a < M
    vf = valid.to(tl.float32)
    hk, hn, rk, rn, kc, nc, gk = _cosines(QN, Pool, bh*(T+M)+chunk*C, n, tok, valid, M, D, C, BM, BDK)
    Hk = _ex(hk, TAU)*vf[:, None]
    Hn = _ex(hn, TAU)*vf[:, None]*vf[None, :]
    Rk = _ex(rk, TAU)*vf[:, None]
    Rn = _ex(rn, TAU)*vf[:, None]*vf[None, :]
    Tm = _inverse(-beta[:, None]*(Dl*Hn), C)
    WS = -tl.dot(Tm, (beta*gam)[:, None]*Hk, input_precision=PROJ)
    WV = Tm*beta[None, :]
    Rd = Dp*Rn
    os = gam[:, None]*Rk+tl.dot(Rd, WS, input_precision=PROJ)
    ov = tl.where(le, tl.dot(Rd, WV, input_precision=PROJ), 0.)
    mo = (a < n) & am
    mn = (M+t) < n
    Kc = tl.where(mo[None, :], _ex(kc, TAU), 0.)
    Nc = tl.where(mn[None, :] & valid[None, :], _ex(nc, TAU), 0.)
    valid2 = am[:, None] & am[None, :]
    Gk = tl.where(valid2, _ex(gk, TAU), 0.)+tl.where((a[:, None] == a[None, :]) & valid2, RIDGE, 0.)
    Pinv = _sweep_inverse(Gk, M, BM)
    Jo = (a[None, :] == n+a[:, None]).to(tl.float32)+tl.dot(Pinv, Kc, input_precision=PROJ)
    Jn = ((M+t)[None, :] == n+a[:, None]).to(tl.float32)+tl.dot(Pinv, Nc, input_precision=PROJ)
    fs = tl.exp(gC)*Jo+tl.dot(Jn, dend[:, None]*WS, input_precision=PROJ)
    fv = tl.dot(Jn, dend[:, None]*WV, input_precision=PROJ)
    idx = bh*NC+chunk
    c = tl.arange(0, C)
    tl.store(OS+idx*C*M+c[:, None]*M+a[None, :], os, am[None, :])
    tl.store(OV+idx*C*C+c[:, None]*C+c[None, :], ov)
    tl.store(FS+idx*M*M+a[:, None]*M+a[None, :], fs, valid2)
    tl.store(FV+idx*M*C+a[:, None]*C+c[None, :], fv, am[:, None])
    if SAVE:
        tl.store(PINV+idx*M*M+a[:, None]*M+a[None, :], Pinv, valid2)


@tr.jit
def block_mapgrad_kernel(OV, FV, Value, Tape, DS, DO, DOS, DOV, DFS, DFV, DV,
                         T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
                         NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr):
    # Contract the value axis: dOS=dO S0^T, dOV=dO V^T, dFS=dS S0^T, dFV=dS V^T; and dV.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    a = tl.arange(0, BM)
    c = tl.arange(0, C)
    am = a < M
    idx = bh*NC+chunk
    tok = ((bh//H)*T+chunk*C+c)*H+bh % H
    valid = chunk*C+c < T
    ov = tl.load(OV+idx*C*C+c[:, None]*C+c[None, :]).to(tl.float32)
    fv = tl.load(FV+idx*M*C+a[:, None]*C+c[None, :], am[:, None], 0)
    dos = tl.zeros((C, BM), tl.float32)
    dov = tl.zeros((C, C), tl.float32)
    dfs = tl.zeros((BM, BM), tl.float32)
    dfv = tl.zeros((BM, C), tl.float32)
    for v0 in range(0, V, BV):
        v = v0+tl.arange(0, BV)
        vv = v < V
        smask = am[:, None] & vv[None, :]
        cmask = valid[:, None] & vv[None, :]
        s0 = tl.load(Tape+idx*M*V+a[:, None]*V+v[None, :], smask, 0)
        ds = tl.load(DS+idx*M*V+a[:, None]*V+v[None, :], smask, 0)
        do = tl.load(DO+tok[:, None]*V+v[None, :], cmask, 0).to(tl.float32)
        val = tl.load(Value+tok[:, None]*V+v[None, :], cmask, 0).to(tl.float32)
        dos += tl.dot(do, tl.trans(s0), input_precision=ATT)
        dov += tl.dot(do, tl.trans(val), input_precision=ATT)
        dfs += tl.dot(ds, tl.trans(s0), input_precision=ATT)
        dfv += tl.dot(ds, tl.trans(val), input_precision=ATT)
        dv = tl.dot(tl.trans(ov), do, input_precision=ATT)+tl.dot(tl.trans(fv), ds, input_precision=ATT)
        tl.store(DV+tok[:, None]*V+v[None, :], dv, cmask)
    tl.store(DOS+idx*C*M+c[:, None]*M+a[None, :], dos, am[None, :])
    tl.store(DOV+idx*C*C+c[:, None]*C+c[None, :], dov)
    tl.store(DFS+idx*M*M+a[:, None]*M+a[None, :], dfs, am[:, None] & am[None, :])
    tl.store(DFV+idx*M*C+a[:, None]*C+c[None, :], dfv, am[:, None])


@tr.jit
def block_maps_bwd_kernel(QN, Pool, G, Beta, PINV, DOS, DOV, DFS, DFV, DG, DB, CHK, CHN, CRK, CRN, CKC, CNC, CG,
                          T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                          NC: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr, TAU: tl.constexpr, BDK: tl.constexpr):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, g, beta = _chunk_setup(G, Beta, chunk, bh, T, H, C)
    gc, gam, E, Dl, Dp, gC, dend, le, lt = _decay(g, valid, n, C)
    a = tl.arange(0, BM)
    am = a < M
    vf = valid.to(tl.float32)
    idx = bh*NC+chunk
    valid2 = am[:, None] & am[None, :]
    # ---- recompute forward pieces
    rbase = bh*(T+M)+chunk*C
    hk, hn, rk, rn, kc, nc, gk = _cosines(QN, Pool, rbase, n, tok, valid, M, D, C, BM, BDK)
    Hk = _ex(hk, TAU)*vf[:, None]
    Hn = _ex(hn, TAU)*vf[:, None]*vf[None, :]
    Rk = _ex(rk, TAU)*vf[:, None]
    Rn = _ex(rn, TAU)*vf[:, None]*vf[None, :]
    Tm = _inverse(-beta[:, None]*(Dl*Hn), C)
    bg = beta*gam
    WS = -tl.dot(Tm, bg[:, None]*Hk, input_precision=PROJ)
    WV = Tm*beta[None, :]
    Rd = Dp*Rn
    mo = (a < n) & am
    mn = ((M+t) < n) & valid
    Kc = tl.where(mo[None, :], _ex(kc, TAU), 0.)
    Nc = tl.where(mn[None, :], _ex(nc, TAU), 0.)
    Gk = tl.where(valid2, _ex(gk, TAU), 0.)
    Pinv = tl.load(PINV+idx*M*M+a[:, None]*M+a[None, :], valid2, 0)
    PEo = tl.dot(Pinv, Kc, input_precision=PROJ)
    PEn = tl.dot(Pinv, Nc, input_precision=PROJ)
    Jo = (a[None, :] == n+a[:, None]).to(tl.float32)+PEo
    Jn = ((M+t)[None, :] == n+a[:, None]).to(tl.float32)+PEn
    c = tl.arange(0, C)
    dOS = tl.load(DOS+idx*C*M+c[:, None]*M+a[None, :], am[None, :], 0)
    dOV = tl.load(DOV+idx*C*C+c[:, None]*C+c[None, :])
    dFS = tl.load(DFS+idx*M*M+a[:, None]*M+a[None, :], valid2, 0)
    dFV = tl.load(DFV+idx*M*C+a[:, None]*C+c[None, :], am[:, None], 0)
    # ---- next-state maps: FS = gC Jo + Jn diag(dend) WS ; FV = Jn diag(dend) WV
    egC = tl.exp(gC)
    WSd = dend[:, None]*WS
    WVd = dend[:, None]*WV
    dgC = tl.sum(tl.sum(dFS*Jo, 1), 0)
    dJn = tl.dot(dFS, tl.trans(WSd), input_precision=PROJ)+tl.dot(dFV, tl.trans(WVd), input_precision=PROJ)
    dWSd = tl.dot(tl.trans(Jn), dFS, input_precision=PROJ)
    dWVd = tl.dot(tl.trans(Jn), dFV, input_precision=PROJ)
    ddend = tl.sum(dWSd*WS, 1)+tl.sum(dWVd*WV, 1)
    Zo = tl.dot(Pinv, tl.where(mo[None, :], egC*dFS, 0.), input_precision=PROJ)
    Zn = tl.dot(Pinv, tl.where(mn[None, :], dJn, 0.), input_precision=PROJ)
    dGk = -(tl.dot(Zo, tl.trans(PEo), input_precision=PROJ)+tl.dot(Zn, tl.trans(PEn), input_precision=PROJ))
    cKc = tl.where(mo[None, :], Zo, 0.)*Kc/TAU
    cNc = tl.where(mn[None, :], Zn, 0.)*Nc/TAU
    cG = dGk*Gk/TAU
    # ---- outputs: OS = diag(gam) Rk + Rd WS ; OV = Rd WV
    dWS = dend[:, None]*dWSd+tl.dot(tl.trans(Rd), dOS, input_precision=PROJ)
    dWV = dend[:, None]*dWVd+tl.dot(tl.trans(Rd), dOV, input_precision=PROJ)
    dgam = tl.sum(dOS*Rk, 1)
    dRd = tl.dot(dOS, tl.trans(WS), input_precision=PROJ)+tl.dot(dOV, tl.trans(WV), input_precision=PROJ)
    cRk = gam[:, None]*dOS*Rk/TAU
    cRn = dRd*Dp*Rn/TAU
    dE = tl.where(le, dRd*Rn, 0.)
    # ---- UT transform: WS = -T diag(bg) Hk ; WV = T diag(beta) ; T = (I + diag(beta)(D o Hn))^-1
    TtdWS = tl.dot(tl.trans(Tm), dWS, input_precision=PROJ)
    cHk = -bg[:, None]*TtdWS*Hk/TAU
    dbg = -tl.sum(TtdWS*Hk, 1)
    dbeta = dbg*gam+tl.sum(dWV*Tm, 0)
    dgam += dbg*beta
    dT = -tl.dot(dWS, tl.trans(bg[:, None]*Hk), input_precision=PROJ)+dWV*beta[None, :]
    TtdT = tl.dot(tl.trans(Tm), dT, input_precision=PROJ)
    dL = -tl.where(lt, tl.dot(TtdT, tl.trans(Tm), input_precision=PROJ), 0.)
    dbeta += tl.sum(dL*Dl*Hn, 1)
    cHn = beta[:, None]*dL*Dl*Hn/TAU
    dE += tl.where(lt, beta[:, None]*dL*Hn, 0.)
    # ---- log-decay
    dEE = dE*E
    dGc = dgam*gam+tl.sum(dEE, 1)-tl.sum(dEE, 0)-ddend*dend
    dGC = dgC*egC+tl.sum(ddend*dend, 0)
    dGc += tl.where(t == n-1, dGC, 0.)
    dg = tl.sum(dGc, 0)-tl.cumsum(dGc, 0)+dGc
    tl.store(DG+tok, dg, valid)
    tl.store(DB+tok, dbeta, valid)
    # ---- cosine gradients; the key-width products run in block_keygrad_kernel
    c = tl.arange(0, C)
    tl.store(CHK+idx*C*M+c[:, None]*M+a[None, :], cHk, am[None, :])
    tl.store(CRK+idx*C*M+c[:, None]*M+a[None, :], cRk, am[None, :])
    tl.store(CHN+idx*C*C+c[:, None]*C+c[None, :], cHn+tl.trans(cHn))
    tl.store(CRN+idx*C*C+c[:, None]*C+c[None, :], cRn)
    tl.store(CKC+idx*M*M+a[:, None]*M+a[None, :], cKc, valid2)
    tl.store(CG+idx*M*M+a[:, None]*M+a[None, :], cG+tl.trans(cG), valid2)
    tl.store(CNC+idx*M*C+a[:, None]*C+c[None, :], cNc, am[:, None])


@tr.jit
def block_keygrad_kernel(QN, Pool, CHK, CHN, CRK, CRN, CKC, CNC, CG, DQN, DPool,
                         T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                         NC: tl.constexpr, BM: tl.constexpr, BDK: tl.constexpr):
    chunk, bh, d0 = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2)*BDK
    b, head = bh//H, bh % H
    t = tl.arange(0, C)
    a = tl.arange(0, BM)
    am = a < M
    n = tl.minimum(C, T-chunk*C)
    valid = t < n
    tok = (b*T+chunk*C+t)*H+head
    idx = bh*NC+chunk
    cHk = tl.load(CHK+idx*C*M+t[:, None]*M+a[None, :], am[None, :], 0)
    cRk = tl.load(CRK+idx*C*M+t[:, None]*M+a[None, :], am[None, :], 0)
    cHnS = tl.load(CHN+idx*C*C+t[:, None]*C+t[None, :])
    cRn = tl.load(CRN+idx*C*C+t[:, None]*C+t[None, :])
    cKc = tl.load(CKC+idx*M*M+a[:, None]*M+a[None, :], am[:, None] & am[None, :], 0)
    cGS = tl.load(CG+idx*M*M+a[:, None]*M+a[None, :], am[:, None] & am[None, :], 0)
    cNc = tl.load(CNC+idx*M*C+a[:, None]*C+t[None, :], am[:, None], 0)
    rbase = bh*(T+M)+chunk*C
    Kold, Kn, Knew, Q = _tiles(QN, Pool, rbase, n, tok, valid, d0, M, D, C, BM, BDK)
    d = d0+tl.arange(0, BDK)
    dmask = d[None, :] < D
    dQ = tl.dot(cRk, Kold, input_precision=PROJ)+tl.dot(cRn, Kn, input_precision=PROJ)
    tl.store(DQN+tok[:, None]*D+d[None, :], dQ, valid[:, None] & dmask)
    dKn = (tl.dot(cHk, Kold, input_precision=PROJ)+tl.dot(cHnS, Kn, input_precision=PROJ)
           + tl.dot(tl.trans(cRn), Q, input_precision=PROJ)+tl.dot(tl.trans(cNc), Knew, input_precision=PROJ))
    tl.atomic_add(DPool+(rbase+M+t[:, None])*D+d[None, :], dKn, valid[:, None] & dmask, sem='relaxed')
    dKold = (tl.dot(tl.trans(cHk), Kn, input_precision=PROJ)+tl.dot(tl.trans(cRk), Q, input_precision=PROJ)
             + tl.dot(tl.trans(cKc), Knew, input_precision=PROJ))
    tl.atomic_add(DPool+(rbase+a[:, None])*D+d[None, :], dKold, am[:, None] & dmask, sem='relaxed')
    dKnew = (tl.dot(cKc, Kold, input_precision=PROJ)+tl.dot(cNc, Kn, input_precision=PROJ)
             + tl.dot(cGS, Knew, input_precision=PROJ))
    tl.atomic_add(DPool+(rbase+n+a[:, None])*D+d[None, :], dKnew, am[:, None] & dmask, sem='relaxed')


@tr.jit
def final_rows_kernel(Pool, Final, DPool, DF, T: tl.constexpr, M: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                      BACKWARD: tl.constexpr):
    bh, m = tl.program_id(0).to(tl.int64), tl.program_id(1)
    d = tl.arange(0, BD)
    row = (bh*(T+M)+T+m)*D+d
    if BACKWARD:
        tl.atomic_add(DPool+row, tl.load(DF+(bh*M+m)*D+d, d < D, 0), d < D, sem='relaxed')
    else:
        tl.store(Final+(bh*M+m)*D+d, tl.load(Pool+row, d < D, 0), d < D)


class _BlockKOMP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, keys0, values0, c, tau, ridge, save):
        b, t, h, d = q.shape
        m, width = keys0.shape[-2], v.shape[-1]
        bm, bd = max(16, tr.next_power_of_2(m)), max(16, tr.next_power_of_2(d))
        nc = tr.cdiv(t, c)
        alloc = lambda *s: torch.empty(s, device=q.device, dtype=torch.float32)
        qn, pool = alloc(b, t, h, d), alloc(b, h, t+m, d)
        normalize_kernel[(t+m, b*h)](q, k, keys0, qn, pool, t, h, m, d, bd)
        os, ov, fs, fv = alloc(b, h, nc, c, m), alloc(b, h, nc, c, c), alloc(b, h, nc, m, m), alloc(b, h, nc, m, c)
        pinv = alloc(b, h, nc, m, m) if save else alloc(1)
        block_maps_kernel[(nc, b*h)](qn, pool, g, beta, os, ov, fs, fv, pinv, t, h, m, d, c, nc, bm, bd, tau, ridge, save,
                                     min(bd, 64), num_warps=WARPS['maps'], num_stages=1)
        out = alloc(b, t, h, width)
        final_values = alloc(b, h, m, width)
        tape, off = alloc(b, h, nc, m, width), alloc(b, h, nc, m, width)
        bv = 32
        nv = tr.cdiv(width, bv)
        value_offset_kernel[(nc, b*h, nv)](fv, v, off, t, h, m, width, c, nc, bm, bv, num_warps=WARPS['offset'])
        affine_scan(fs, off, values0, tape, final_values, False, None, bv)
        output_kernel[(nc, b*h, nv)](os, ov, v, tape, out, t, h, m, width, c, nc, bm, bv, num_warps=WARPS['output'])
        final_keys = alloc(b, h, m, d)
        final_rows_kernel[(b*h, m)](pool, final_keys, pool, pool, t, m, d, bd, False)
        ctx.save_for_backward(q, k, v, g, beta, keys0, qn, pool, os, ov, fs, fv, pinv, tape)
        ctx.cfg = (c, tau, values0.dtype)
        return out, final_keys, final_values

    @staticmethod
    @once_differentiable
    def backward(ctx, do, dfk, dfv):
        q, k, v, g, beta, keys0, qn, pool, os, ov, fs, fv, pinv, tape = ctx.saved_tensors
        c, tau, values_dtype = ctx.cfg
        b, t, h, d = q.shape
        m, width = keys0.shape[-2], v.shape[-1]
        bm, bd = max(16, tr.next_power_of_2(m)), max(16, tr.next_power_of_2(d))
        nc = tr.cdiv(t, c)
        do, dfk, dfv = do.contiguous(), dfk.contiguous(), dfv.contiguous()
        alloc = lambda *s: torch.empty(s, device=q.device, dtype=torch.float32)
        ds, off = torch.empty_like(tape), torch.empty_like(tape)
        dvalues0 = torch.empty(dfv.shape, device=q.device, dtype=values_dtype)
        bv = 32
        nv = tr.cdiv(width, bv)
        output_offset_kernel[(nc, b*h, nv)](os, do, off, t, h, m, width, c, nc, bm, bv, num_warps=WARPS['offset'])
        affine_scan(fs, off, dfv, ds, dvalues0, True, None, bv)
        dos, dov, dfs, dfv_map = torch.empty_like(os), torch.empty_like(ov), torch.empty_like(fs), torch.empty_like(fv)
        dv = torch.empty_like(v)
        block_mapgrad_kernel[(nc, b*h)](ov, fv, v, tape, ds, do, dos, dov, dfs, dfv_map, dv,
                                        t, h, m, width, c, nc, bm, 32, num_warps=WARPS['mapgrad'])
        dqn = torch.empty_like(qn)
        dpool = zero(torch.empty_like(pool))
        dg, db = torch.empty_like(g), torch.empty_like(beta)
        cs = [alloc(b, h, nc, *shape) for shape in ((c, m), (c, c), (c, m), (c, c), (m, m), (m, c), (m, m))]
        block_maps_bwd_kernel[(nc, b*h)](qn, pool, g, beta, pinv, dos, dov, dfs, dfv_map, dg, db, *cs,
                                         t, h, m, d, c, nc, bm, bd, tau, min(bd, 64), num_warps=WARPS['maps_bwd'], num_stages=1)
        block_keygrad_kernel[(nc, b*h, tr.cdiv(d, min(bd, 64)))](qn, pool, *cs, dqn, dpool, t, h, m, d, c, nc, bm, min(bd, 64),
                                        num_warps=WARPS['keygrad'], num_stages=1)
        final_rows_kernel[(b*h, m)](pool, pool, dpool, dfk, t, m, d, bd, True)
        dq, dk, dk0 = torch.empty_like(q), torch.empty_like(k), torch.empty_like(keys0)
        normalize_bwd_kernel[(t+m, b*h)](q, k, keys0, dqn, dpool, dq, dk, dk0, t, h, m, d, bd)
        return dq, dk, dv, dg, db, dk0, dvalues0, None, None, None, None


def block_komp(q, k, v, g, beta, initial_state, chunk_size=64, temperature=0.5, ridge=1e-4):
    """q,k:[B,T,H,K], v:[B,T,H,V]; g:[B,T,H] finite log decay (<=0); beta:[B,T,H] in [0,1].

    initial_state=(keys [B,H,M,K], values [B,H,M,V]). Returns output [B,T,H,V] FP32 and the final
    (keys, values) state. chunk_size in {16,32,64}; M <= 64. FIFO dictionary, joint projection at chunk ends.
    """
    keys0, values0 = initial_state
    if chunk_size not in (16, 32, 64) or keys0.shape[-2] > 64:
        raise ValueError('chunk_size must be 16, 32 or 64 and at most 64 slots are supported')
    inputs = [x.contiguous() for x in (q, k, v, g, beta, keys0, values0)]
    save = torch.is_grad_enabled() and any(x.requires_grad for x in inputs)
    with torch.cuda.device(q.device):
        out, keys, values = _BlockKOMP.apply(*inputs, chunk_size, temperature, ridge, save)
    return out, (keys, values)
