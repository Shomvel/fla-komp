"""Block-KOMP, FLA-style pipeline: one focused kernel per stage, chunk-local matrices kept in HBM.

Forward  att: Hk,Hn,Rk,Rn (key-width blocked cosines, tf32) | proj: Kc,Nc,Gk (tf32x3), Gram sweep inverse,
         PEo=P Kc, PEn=P Nc | ut: (I+diag(beta)(D o Hn))^-1 by 16x16 forward substitution and merges |
         compose: OS,OV,FS,FV | two-level affine scan and outputs (shared with the per-token path).
Backward mapgrad: dOS,dOV,dFS,dFV,dV | proj_bwd: cosine gradients of Kc,Nc,Gk and dWS,dWV parts |
         ut_bwd: cosine gradients of Hk,Hn,Rk,Rn, dg, dbeta | keygrad: key-width blocked products and atomics.
Same mathematics as block_komp.py (v0), which stays as the single-kernel reference implementation.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from .geometry import normalize_kernel, normalize_bwd_kernel
from .geometry_chunk import _sweep_inverse
from .recurrence import output_kernel
from .scan import affine_scan, value_offset_kernel, output_offset_kernel
from .block_komp import _rows, _ex, block_mapgrad_kernel, final_rows_kernel
from .utils import zero

ATT = tl.constexpr('tf32')
PROJ = tl.constexpr('tf32x3')
UT = tl.constexpr('ieee')
WARPS = {'att': 4, 'proj': 4, 'ut': 2, 'compose': 4, 'mapgrad': 4, 'proj_bwd': 4, 'ut_bwd': 4, 'keygrad': 4,
         'offset': 4, 'output': 4}
KEY_BLOCK = 32
SCAN = 'tf32'   # state scans and outputs: BF16-level target
SCAN_BV = 64
MAPGRAD_BV = 32


@tr.jit
def _ld(P, idx, R: tl.constexpr, CC: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    r = tl.arange(0, BR)
    c = tl.arange(0, BC)
    return tl.load(P+idx*R*CC+r[:, None]*CC+c[None, :], (r[:, None] < R) & (c[None, :] < CC), 0).to(tl.float32)


@tr.jit
def _st(P, idx, x, R: tl.constexpr, CC: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    r = tl.arange(0, BR)
    c = tl.arange(0, BC)
    tl.store(P+idx*R*CC+r[:, None]*CC+c[None, :], x, (r[:, None] < R) & (c[None, :] < CC))


@tr.jit
def att_kernel(QN, Pool, G, Beta, HK, HN, RK, RN, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
               D: tl.constexpr, C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BDK: tl.constexpr,
               TAU: tl.constexpr):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t = tl.arange(0, C)
    n = tl.minimum(C, T-chunk*C)
    valid = t < n
    tok = ((bh//H)*T+chunk*C+t)*H+bh % H
    rbase = bh*(T+M)+chunk*C
    hk = tl.zeros((C, BM), tl.float32)
    hn = tl.zeros((C, C), tl.float32)
    rk = tl.zeros((C, BM), tl.float32)
    rn = tl.zeros((C, C), tl.float32)
    for d0 in range(0, D, BDK):
        d = d0+tl.arange(0, BDK)
        Kold = _rows(Pool, rbase, M, BM, D, d0, BDK)
        Kn = _rows(Pool, rbase+M, n, C, D, d0, BDK)
        Q = tl.load(QN+tok[:, None]*D+d[None, :], valid[:, None] & (d[None, :] < D), 0)
        hk += tl.dot(Kn, tl.trans(Kold), input_precision=ATT)
        hn += tl.dot(Kn, tl.trans(Kn), input_precision=ATT)
        rk += tl.dot(Q, tl.trans(Kold), input_precision=ATT)
        rn += tl.dot(Q, tl.trans(Kn), input_precision=ATT)
    vf = valid.to(tl.float32)
    idx = bh*NC+chunk
    _st(HK, idx, _ex(hk, TAU)*vf[:, None], C, M, C, BM)
    _st(HN, idx, _ex(hn, TAU)*vf[:, None]*vf[None, :], C, C, C, C)
    _st(RK, idx, _ex(rk, TAU)*vf[:, None], C, M, C, BM)
    _st(RN, idx, _ex(rn, TAU)*vf[:, None]*vf[None, :], C, C, C, C)


@tr.jit
def proj_kernel(Pool, KC, NCM, GK, PINV, PEO, PEN, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
                D: tl.constexpr, C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BDK: tl.constexpr,
                TAU: tl.constexpr, RIDGE: tl.constexpr):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    n = tl.minimum(C, T-chunk*C)
    rbase = bh*(T+M)+chunk*C
    kc = tl.zeros((BM, BM), tl.float32)
    nc = tl.zeros((BM, C), tl.float32)
    gk = tl.zeros((BM, BM), tl.float32)
    for d0 in range(0, D, BDK):
        Kold = _rows(Pool, rbase, M, BM, D, d0, BDK)
        Kn = _rows(Pool, rbase+M, n, C, D, d0, BDK)
        Knew = _rows(Pool, rbase+n, M, BM, D, d0, BDK)
        kc += tl.dot(Knew, tl.trans(Kold), input_precision=PROJ)
        nc += tl.dot(Knew, tl.trans(Kn), input_precision=PROJ)
        gk += tl.dot(Knew, tl.trans(Knew), input_precision=PROJ)
    a = tl.arange(0, BM)
    t = tl.arange(0, C)
    am = a < M
    valid2 = am[:, None] & am[None, :]
    Kc = tl.where(((a < n) & am)[None, :], _ex(kc, TAU), 0.)
    Nc = tl.where((((M+t) < n) & (t < n))[None, :], _ex(nc, TAU), 0.)
    Gk = tl.where(valid2, _ex(gk, TAU), 0.)
    Pinv = _sweep_inverse(Gk+tl.where((a[:, None] == a[None, :]) & valid2, RIDGE, 0.), M, BM)
    idx = bh*NC+chunk
    _st(KC, idx, Kc, M, M, BM, BM)
    _st(NCM, idx, Nc, M, C, BM, C)
    _st(GK, idx, Gk, M, M, BM, BM)
    _st(PINV, idx, Pinv, M, M, BM, BM)
    _st(PEO, idx, tl.dot(Pinv, Kc, input_precision=PROJ), M, M, BM, BM)
    _st(PEN, idx, tl.dot(Pinv, Nc, input_precision=PROJ), M, C, BM, C)


@tr.jit
def _slot_decay(G, Beta, chunk, bh, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
                BM: tl.constexpr, SLOT: tl.constexpr):
    """Per-ring-slot log decay (scalar decay broadcasts to every slot). Atom at sequence position p
    lives in slot p mod M; old atom u of chunk c sits at position cC-M+u, new atom s at cC+s."""
    b, head = bh//H, bh % H
    t = tl.arange(0, C)
    u = tl.arange(0, BM)
    t0 = chunk*C
    n = tl.minimum(C, T-t0)
    valid = t < n
    tok = (b*T+t0+t)*H+head
    beta = tl.load(Beta+tok, valid, 0).to(tl.float32)
    um = u < M
    o = (t0 % M).to(tl.int32)
    r_new = (o+t) % M
    le = t[:, None] >= t[None, :]
    lt = t[:, None] > t[None, :]
    if SLOT:
        g = tl.load(G+tok[:, None]*M+u[None, :], valid[:, None] & um[None, :], 0).to(tl.float32)
        gc = tl.cumsum(g, 0)
        r_old = tl.where(um, (o+u) % M, 0)
        Gam = tl.exp(tl.gather(gc, tl.broadcast_to(r_old[None, :], (C, BM)), 1))
        gcs = tl.gather(gc, tl.broadcast_to(r_new[None, :], (C, C)), 1)
        diag = tl.sum(tl.where(t[:, None] == t[None, :], gcs, 0.), 0)
        Ex = tl.exp(tl.where(le, gcs-diag[None, :], 0.))
        last = t[:, None] == n-1
        dend = tl.where(valid, tl.exp(tl.sum(tl.where(last, gcs, 0.), 0)-diag), 0.)
        gend = tl.sum(tl.where(last, Gam, 0.), 0)
    else:
        # Scalar decay: every slot shares the token's cumulative log decay.
        gc = tl.cumsum(tl.load(G+tok, valid, 0).to(tl.float32), 0)
        Gam = tl.exp(gc)[:, None]+tl.zeros((C, BM), tl.float32)
        Ex = tl.exp(tl.where(le, gc[:, None]-gc[None, :], 0.))
        gC = tl.sum(tl.where(t == n-1, gc, 0.), 0)
        dend = tl.where(valid, tl.exp(gC-gc), 0.)
        gend = tl.exp(gC)+tl.zeros((BM,), tl.float32)
    return t, n, valid, tok, beta, Gam, Ex, tl.where(lt, Ex, 0.), tl.where(le, Ex, 0.), dend, gend, le, lt, o, r_new


@tr.jit
def _blk(P, i: tl.constexpr, j: tl.constexpr, C: tl.constexpr):
    o = tl.arange(0, 16)
    return tl.load(P+(16*i+o[:, None])*C+16*j+o[None, :])


@tr.jit
def _diag_inv(A):
    # (I + A)^-1 for a strictly lower 16x16 block by block doubling (4 steps of 16^3 products).
    o = tl.arange(0, 16)
    L = -tl.where(o[:, None] > o[None, :], A, 0.)
    inv = (o[:, None] == o[None, :]).to(tl.float32)
    for step in tl.static_range(4):
        w = 1 << step
        cross = tl.where((o[:, None]//(2*w) == o[None, :]//(2*w)) & (o[:, None] % (2*w) >= w) & (o[None, :] % (2*w) < w), L, 0.)
        inv += tl.dot(tl.dot(inv, cross, input_precision=PROJ), inv, input_precision=PROJ)
    return inv


@tr.jit
def _put(P, x, i: tl.constexpr, j: tl.constexpr, C: tl.constexpr):
    o = tl.arange(0, 16)
    tl.store(P+(16*i+o[:, None])*C+16*j+o[None, :], x)


@tr.jit
def ut_kernel(G, Beta, HN, AA, TT, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
              NC: tl.constexpr, BM: tl.constexpr, NB: tl.constexpr, SLOT: tl.constexpr):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(G, Beta, chunk, bh, T, H, M, C, BM, SLOT)
    idx = bh*NC+chunk
    Hn = _ld(HN, idx, C, C, C, C)
    _st(AA, idx, beta[:, None]*Dl*Hn, C, C, C, C)
    tl.debug_barrier()
    A = AA+idx*C*C
    O = TT+idx*C*C
    z = tl.zeros((16, 16), tl.float32)
    i11 = _diag_inv(_blk(A, 0, 0, C))
    _put(O, i11, 0, 0, C)
    if NB >= 2:
        i22 = _diag_inv(_blk(A, 1, 1, C))
        i21 = -tl.dot(tl.dot(i22, _blk(A, 1, 0, C), input_precision=UT), i11, input_precision=UT)
        _put(O, i22, 1, 1, C)
        _put(O, i21, 1, 0, C)
        _put(O, z, 0, 1, C)
    if NB == 4:
        i33 = _diag_inv(_blk(A, 2, 2, C))
        i44 = _diag_inv(_blk(A, 3, 3, C))
        a31, a32 = _blk(A, 2, 0, C), _blk(A, 2, 1, C)
        a41, a42, a43 = _blk(A, 3, 0, C), _blk(A, 3, 1, C), _blk(A, 3, 2, C)
        i32 = -tl.dot(tl.dot(i33, a32, input_precision=UT), i22, input_precision=UT)
        i43 = -tl.dot(tl.dot(i44, a43, input_precision=UT), i33, input_precision=UT)
        i31 = -tl.dot(i33, tl.dot(a31, i11, input_precision=UT)+tl.dot(a32, i21, input_precision=UT), input_precision=UT)
        i42 = -tl.dot(i44, tl.dot(a42, i22, input_precision=UT)+tl.dot(a43, i32, input_precision=UT), input_precision=UT)
        i41 = -tl.dot(i44, tl.dot(a41, i11, input_precision=UT)+tl.dot(a42, i21, input_precision=UT)
                      + tl.dot(a43, i31, input_precision=UT), input_precision=UT)
        _put(O, i33, 2, 2, C)
        _put(O, i44, 3, 3, C)
        _put(O, i31, 2, 0, C)
        _put(O, i32, 2, 1, C)
        _put(O, i41, 3, 0, C)
        _put(O, i42, 3, 1, C)
        _put(O, i43, 3, 2, C)
        for j in tl.static_range(4):
            for i in tl.static_range(4):
                if i < j and not (i < 2 and j < 2):
                    _put(O, z, i, j, C)


@tr.jit
def compose_kernel(G, Beta, HK, RK, RN, TT, PEO, PEN, OS, OV, FS, FV, T: tl.constexpr, H: tl.constexpr,
                   M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, SLOT: tl.constexpr):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(G, Beta, chunk, bh, T, H, M, C, BM, SLOT)
    idx = bh*NC+chunk
    a = tl.arange(0, BM)
    Tm = _ld(TT, idx, C, C, C, C)
    WS = -tl.dot(Tm, beta[:, None]*Gam*_ld(HK, idx, C, M, C, BM), input_precision=ATT)
    WV = Tm*beta[None, :]
    Rd = Dp*_ld(RN, idx, C, C, C, C)
    _st(OS, idx, Gam*_ld(RK, idx, C, M, C, BM)+tl.dot(Rd, WS, input_precision=ATT), C, M, C, BM)
    _st(OV, idx, tl.where(le, tl.dot(Rd, WV, input_precision=ATT), 0.), C, C, C, C)
    Jo = (a[None, :] == n+a[:, None]).to(tl.float32)+_ld(PEO, idx, M, M, BM, BM)
    Jn = ((M+t)[None, :] == n+a[:, None]).to(tl.float32)+_ld(PEN, idx, M, C, BM, C)
    _st(FS, idx, Jo*gend[None, :]+tl.dot(Jn, dend[:, None]*WS, input_precision=PROJ), M, M, BM, BM)
    _st(FV, idx, tl.dot(Jn, dend[:, None]*WV, input_precision=PROJ), M, C, BM, C)


@tr.jit
def proj_bwd_a_kernel(G, Beta, HK, TT, PEO, PEN, DFS, DFV, DWSP, DWVP, DJN, VEC,
                      T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                      BM: tl.constexpr, SLOT: tl.constexpr):
    # Next-state maps FS = J_old diag(gend) + J_new diag(dend) WS, FV = J_new diag(dend) WV.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(G, Beta, chunk, bh, T, H, M, C, BM, SLOT)
    idx = bh*NC+chunk
    a = tl.arange(0, BM)
    Tm = _ld(TT, idx, C, C, C, C)
    WS = -tl.dot(Tm, beta[:, None]*Gam*_ld(HK, idx, C, M, C, BM), input_precision=PROJ)
    WV = Tm*beta[None, :]
    Jn = ((M+t)[None, :] == n+a[:, None]).to(tl.float32)+_ld(PEN, idx, M, C, BM, C)
    dFS = _ld(DFS, idx, M, M, BM, BM)
    dFV = _ld(DFV, idx, M, C, BM, C)
    dgend = tl.sum(dFS*((a[None, :] == n+a[:, None]).to(tl.float32)+_ld(PEO, idx, M, M, BM, BM)), 0)
    dWSd = tl.dot(tl.trans(Jn), dFS, input_precision=PROJ)
    dWVd = tl.dot(tl.trans(Jn), dFV, input_precision=PROJ)
    ddend = tl.sum(dWSd*WS, 1)+tl.sum(dWVd*WV, 1)
    _st(DWSP, idx, dend[:, None]*dWSd, C, M, C, BM)
    _st(DWVP, idx, dend[:, None]*dWVd, C, C, C, C)
    tl.store(VEC+idx*(C+BM)+t, ddend*dend)
    tl.store(VEC+idx*(C+BM)+C+a, dgend*gend)
    dJn = tl.dot(dFS, tl.trans(dend[:, None]*WS), input_precision=PROJ)+tl.dot(dFV, tl.trans(dend[:, None]*WV), input_precision=PROJ)
    _st(DJN, idx, dJn, M, C, BM, C)


@tr.jit
def proj_bwd_b_kernel(G, Beta, PEO, PEN, PINV, KC, NCM, GK, DFS, DJN, CKC, CNC, CGS,
                      T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                      BM: tl.constexpr, TAU: tl.constexpr, SLOT: tl.constexpr):
    # Through the Gram inverse: Z = P dJ, dG = -(Z PE^T), then cosine gradients of Kc, Nc, Gk.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(G, Beta, chunk, bh, T, H, M, C, BM, SLOT)
    idx = bh*NC+chunk
    a = tl.arange(0, BM)
    mo = (a < n) & (a < M)
    mn = ((M+t) < n) & valid
    Pinv = _ld(PINV, idx, M, M, BM, BM)
    Zo = tl.dot(Pinv, tl.where(mo[None, :], _ld(DFS, idx, M, M, BM, BM)*gend[None, :], 0.), input_precision=PROJ)
    Zn = tl.dot(Pinv, tl.where(mn[None, :], _ld(DJN, idx, M, C, BM, C), 0.), input_precision=PROJ)
    _st(CKC, idx, tl.where(mo[None, :], Zo, 0.)*_ld(KC, idx, M, M, BM, BM)/TAU, M, M, BM, BM)
    _st(CNC, idx, tl.where(mn[None, :], Zn, 0.)*_ld(NCM, idx, M, C, BM, C)/TAU, M, C, BM, C)
    cG = -(tl.dot(Zo, tl.trans(_ld(PEO, idx, M, M, BM, BM)), input_precision=PROJ)
           + tl.dot(Zn, tl.trans(_ld(PEN, idx, M, C, BM, C)), input_precision=PROJ))*_ld(GK, idx, M, M, BM, BM)/TAU
    _st(CGS, idx, cG+tl.trans(cG), M, M, BM, BM)


@tr.jit
def out_bwd_kernel(G, Beta, HK, RK, RN, TT, DOS, DOV, DWSP, DWVP, DWS, DWV, DGAM, DE, CRK, CRN,
                   T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                   BM: tl.constexpr, TAU: tl.constexpr, SLOT: tl.constexpr):
    # Outputs OS = Gam o Rk + Rd WS, OV = Rd WV (Rd = D' o Rn); completes dWS, dWV.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(G, Beta, chunk, bh, T, H, M, C, BM, SLOT)
    idx = bh*NC+chunk
    Tm = _ld(TT, idx, C, C, C, C)
    WS = -tl.dot(Tm, beta[:, None]*Gam*_ld(HK, idx, C, M, C, BM), input_precision=ATT)
    WV = Tm*beta[None, :]
    dOS = _ld(DOS, idx, C, M, C, BM)
    dOV = _ld(DOV, idx, C, C, C, C)
    dRd = tl.dot(dOS, tl.trans(WS), input_precision=ATT)+tl.dot(dOV, tl.trans(WV), input_precision=ATT)
    Rn = _ld(RN, idx, C, C, C, C)
    Rd = Dp*Rn
    _st(CRN, idx, dRd*Rd/TAU, C, C, C, C)
    _st(DE, idx, tl.where(le, dRd*Rn, 0.), C, C, C, C)
    _st(DWS, idx, _ld(DWSP, idx, C, M, C, BM)+tl.dot(tl.trans(Rd), dOS, input_precision=ATT), C, M, C, BM)
    _st(DWV, idx, _ld(DWVP, idx, C, C, C, C)+tl.dot(tl.trans(Rd), dOV, input_precision=ATT), C, C, C, C)
    Rk = _ld(RK, idx, C, M, C, BM)
    _st(CRK, idx, Gam*dOS*Rk/TAU, C, M, C, BM)
    _st(DGAM, idx, dOS*Rk, C, M, C, BM)


@tr.jit
def ut_bwd_kernel(G, Beta, HK, HN, TT, DWS, DWV, DGAM, DE, CHK, CHNS, DB,
                  T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                  BM: tl.constexpr, TAU: tl.constexpr, SLOT: tl.constexpr):
    # WS = -T (beta Gam Hk), WV = T diag(beta), T = (I + diag(beta)(D o Hn))^-1.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(G, Beta, chunk, bh, T, H, M, C, BM, SLOT)
    idx = bh*NC+chunk
    Tm = _ld(TT, idx, C, C, C, C)
    Hk = _ld(HK, idx, C, M, C, BM)
    dWS = _ld(DWS, idx, C, M, C, BM)
    dWV = _ld(DWV, idx, C, C, C, C)
    dB = -tl.dot(tl.trans(Tm), dWS, input_precision=ATT)
    _st(CHK, idx, dB*beta[:, None]*Gam*Hk/TAU, C, M, C, BM)
    _st(DGAM, idx, _ld(DGAM, idx, C, M, C, BM)+dB*beta[:, None]*Hk, C, M, C, BM)
    dbeta = tl.sum(dB*Gam*Hk, 1)+tl.sum(dWV*Tm, 0)
    dT = -tl.dot(dWS, tl.trans(beta[:, None]*Gam*Hk), input_precision=ATT)+dWV*beta[None, :]
    dL = -tl.where(lt, tl.dot(tl.dot(tl.trans(Tm), dT, input_precision=ATT), tl.trans(Tm), input_precision=ATT), 0.)
    Hn = _ld(HN, idx, C, C, C, C)
    dbeta += tl.sum(dL*Dl*Hn, 1)
    cHn = beta[:, None]*dL*Dl*Hn/TAU
    _st(CHNS, idx, cHn+tl.trans(cHn), C, C, C, C)
    _st(DE, idx, _ld(DE, idx, C, C, C, C)+tl.where(lt, beta[:, None]*dL*Hn, 0.), C, C, C, C)
    tl.store(DB+tok, dbeta, valid)


@tr.jit
def decay_bwd_kernel(G, Beta, DGAM, DE, VEC, DG,
                     T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                     BM: tl.constexpr, SLOT: tl.constexpr):
    # Log decay, scattered to ring slots: old atom u <-> slot (o+u) mod M, new atom s -> (o+s) mod M.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(G, Beta, chunk, bh, T, H, M, C, BM, SLOT)
    idx = bh*NC+chunk
    u = tl.arange(0, BM)
    um = u < M
    Y = _ld(DE, idx, C, C, C, C)*Ex
    vec = VEC+idx*(C+BM)
    Z = tl.load(vec+t)
    W = tl.load(vec+C+u, um, 0)
    if not SLOT:
        dgc = tl.sum(_ld(DGAM, idx, C, M, C, BM)*Gam, 1)+tl.sum(Y, 1)-tl.sum(Y, 0)-Z
        dgc += tl.where(t == n-1, tl.sum(Z, 0)+tl.sum(W, 0), 0.)
        dgc = tl.where(valid, dgc, 0.)
        tl.store(DG+tok, tl.sum(dgc, 0)-tl.cumsum(dgc, 0)+dgc, valid)
        return
    u_of_r = tl.where(um, (u-o+M) % M, 0)
    Rnew = (u[None, :] == r_new[:, None]).to(tl.float32)
    dGc = tl.gather(_ld(DGAM, idx, C, M, C, BM)*Gam, tl.broadcast_to(u_of_r[None, :], (C, BM)), 1)
    dGc += tl.dot(Y, Rnew, input_precision='ieee')-(tl.sum(Y, 0)+Z)[:, None]*Rnew
    rowC = tl.sum(Z[:, None]*Rnew, 0)+tl.gather(W, u_of_r, 0)
    dGc += tl.where(t[:, None] == n-1, rowC[None, :], 0.)
    dGc = tl.where(valid[:, None] & um[None, :], dGc, 0.)
    dg = tl.sum(dGc, 0)[None, :]-tl.cumsum(dGc, 0)+dGc
    tl.store(DG+tok[:, None]*M+u[None, :], dg, valid[:, None] & um[None, :])


@tr.jit
def keygrad_kernel(QN, Pool, CHK, CHN, CRK, CRN, CKC, CNC, CG, DQN, DU,
                   T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                   NC: tl.constexpr, BM: tl.constexpr, BDK: tl.constexpr, BU: tl.constexpr):
    # One key-width block per program: tiles loaded once, factors loaded right before each product.
    # Key-width block is the fastest grid axis: the blocks of one chunk share its factors in L2.
    d0, chunk, bh = tl.program_id(0)*BDK, tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    b, head = bh//H, bh % H
    t = tl.arange(0, C)
    a = tl.arange(0, BM)
    am = a < M
    n = tl.minimum(C, T-chunk*C)
    valid = t < n
    tok = (b*T+chunk*C+t)*H+head
    idx = bh*NC+chunk
    rbase = bh*(T+M)+chunk*C
    d = d0+tl.arange(0, BDK)
    dmask = d[None, :] < D
    Kold = _rows(Pool, rbase, M, BM, D, d0, BDK)
    Kn = _rows(Pool, rbase+M, n, C, D, d0, BDK)
    Knew = _rows(Pool, rbase+n, M, BM, D, d0, BDK)
    Q = tl.load(QN+tok[:, None]*D+d[None, :], valid[:, None] & dmask, 0)
    dq = (tl.dot(_ld(CRK, idx, C, M, C, BM), Kold, input_precision=ATT)
          + tl.dot(_ld(CRN, idx, C, C, C, C), Kn, input_precision=ATT))
    tl.store(DQN+tok[:, None]*D+d[None, :], dq, valid[:, None] & dmask)
    dkn = (tl.dot(_ld(CHK, idx, C, M, C, BM), Kold, input_precision=ATT)
           + tl.dot(_ld(CHN, idx, C, C, C, C), Kn, input_precision=ATT)
           + tl.dot(tl.trans(_ld(CRN, idx, C, C, C, C)), Q, input_precision=ATT)
           + tl.dot(tl.trans(_ld(CNC, idx, M, C, BM, C)), Knew, input_precision=PROJ))
    dko = (tl.dot(tl.trans(_ld(CHK, idx, C, M, C, BM)), Kn, input_precision=ATT)
           + tl.dot(tl.trans(_ld(CRK, idx, C, M, C, BM)), Q, input_precision=ATT)
           + tl.dot(tl.trans(_ld(CKC, idx, M, M, BM, BM)), Knew, input_precision=PROJ))
    dkk = (tl.dot(_ld(CKC, idx, M, M, BM, BM), Kold, input_precision=PROJ)
           + tl.dot(_ld(CNC, idx, M, C, BM, C), Kn, input_precision=PROJ)
           + tl.dot(_ld(CG, idx, M, M, BM, BM), Knew, input_precision=PROJ))
    # Union rows: old dictionary at 0..M, new keys at M..M+n, kept atoms at n..n+M (overlapping).
    w = tl.arange(0, BU)
    du = _place(dkn, w-M, n, C, BU, BDK)
    du += _place(dko, w, M, BM, BU, BDK)
    du += _place(dkk, w-n, M, BM, BU, BDK)
    tl.store(DU+(idx*BU+w[:, None])*D+d[None, :], du, dmask)


@tr.jit
def _place(x, src, rows, R: tl.constexpr, BU: tl.constexpr, BDK: tl.constexpr):
    ok = (src >= 0) & (src < rows)
    idx = tl.broadcast_to(tl.where(ok, src, 0)[:, None], (BU, BDK))
    return tl.where(ok[:, None], tl.gather(x, idx, 0), 0.)


@tr.jit
def pool_grad_kernel(DU, DPool, T: tl.constexpr, M: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                     NC: tl.constexpr, BU: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
    # Deterministic sum over every chunk whose union rows [cC, cC+M+n_c) cover a pool row.
    blk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    r = blk*BR+tl.arange(0, BR)
    d = tl.arange(0, BD)
    rmask = r < T+M
    acc = tl.zeros((BR, BD), tl.float32)
    lo = tl.maximum(blk*BR-M-C+1, 0)//C
    hi = tl.minimum((blk*BR+BR-1)//C, NC-1)
    for c in range(lo, hi+1):
        n = tl.minimum(C, T-c*C)
        rel = r-c*C
        ok = (rel >= 0) & (rel < M+n) & rmask
        acc += tl.load(DU+((bh*NC+c)*BU+rel[:, None])*D+d[None, :], ok[:, None] & (d[None, :] < D), 0)
    tl.store(DPool+(bh*(T+M)+r[:, None])*D+d[None, :], acc, rmask[:, None] & (d[None, :] < D))


class _BlockFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, keys0, values0, c, tau, ridge):
        b, t, h, d = q.shape
        m, width = keys0.shape[-2], v.shape[-1]
        bm, bd = max(16, tr.next_power_of_2(m)), max(16, tr.next_power_of_2(d))
        bdk = min(bd, 64)
        nc = tr.cdiv(t, c)
        alloc = lambda *s: torch.empty(s, device=q.device, dtype=torch.float32)
        mats = lambda *shape: alloc(b*h*nc, *shape)
        # Attention-side matrices follow the input precision (BF16 in, BF16 stored); everything on the
        # projection side (Gram, its inverse, projections, their cosine gradients) and the state maps stay FP32.
        sdt = q.dtype if q.dtype in (torch.bfloat16, torch.float16) else torch.float32
        low = lambda *shape: torch.empty((b*h*nc, *shape), device=q.device, dtype=sdt)
        qn, pool = alloc(b, t, h, d), alloc(b, h, t+m, d)
        normalize_kernel[(t+m, b*h)](q, k, keys0, qn, pool, t, h, m, d, bd)
        hk, hn, rk, rn = low(c, m), low(c, c), low(c, m), low(c, c)
        kc, ncm, gk, pinv, peo, pen = mats(m, m), mats(m, c), mats(m, m), mats(m, m), mats(m, m), mats(m, c)
        aa, tt = mats(c, c), low(c, c)
        os, ov, fs, fv = low(c, m), low(c, c), mats(m, m), mats(m, c)
        slot = g.ndim == 4
        grid = (nc, b*h)
        att_kernel[grid](qn, pool, g, beta, hk, hn, rk, rn, t, h, m, d, c, nc, bm, bdk, tau, num_warps=WARPS['att'], num_stages=1)
        proj_kernel[grid](pool, kc, ncm, gk, pinv, peo, pen, t, h, m, d, c, nc, bm, bdk, tau, ridge,
                          num_warps=WARPS['proj'], num_stages=1)
        ut_kernel[grid](g, beta, hn, aa, tt, t, h, m, c, nc, bm, c//16, slot, num_warps=WARPS['ut'])
        compose_kernel[grid](g, beta, hk, rk, rn, tt, peo, pen, os, ov, fs, fv, t, h, m, c, nc, bm, slot, num_warps=WARPS['compose'])
        out, final_values = torch.empty(v.shape, device=q.device, dtype=v.dtype), alloc(b, h, m, width)
        tape, off = alloc(b, h, nc, m, width), alloc(b, h, nc, m, width)
        bv = SCAN_BV
        nv = tr.cdiv(width, bv)
        value_offset_kernel[(nc, b*h, nv)](fv, v, off, t, h, m, width, c, nc, bm, bv, SCAN, num_warps=WARPS['offset'])
        affine_scan(fs, off, values0, tape, final_values, False, None, bv, SCAN)
        output_kernel[(nc, b*h, nv)](os, ov, v, tape, out, t, h, m, width, c, nc, bm, bv, SCAN, num_warps=WARPS['output'])
        final_keys = alloc(b, h, m, d)
        final_rows_kernel[(b*h, m)](pool, final_keys, pool, pool, t, m, d, bd, False)
        ctx.save_for_backward(q, k, v, g, beta, keys0, qn, pool, hk, hn, rk, rn, kc, ncm, gk, pinv, peo, pen, tt,
                              os, ov, fs, fv, tape)
        ctx.cfg = (c, tau, values0.dtype)
        return out, final_keys, final_values

    @staticmethod
    @once_differentiable
    def backward(ctx, do, dfk, dfv):
        (q, k, v, g, beta, keys0, qn, pool, hk, hn, rk, rn, kc, ncm, gk, pinv, peo, pen, tt,
         os, ov, fs, fv, tape) = ctx.saved_tensors
        c, tau, values_dtype = ctx.cfg
        b, t, h, d = q.shape
        m, width = keys0.shape[-2], v.shape[-1]
        bm, bd = max(16, tr.next_power_of_2(m)), max(16, tr.next_power_of_2(d))
        nc = tr.cdiv(t, c)
        do, dfk, dfv = do.contiguous(), dfk.contiguous(), dfv.contiguous()
        ds, off = torch.empty_like(tape), torch.empty_like(tape)
        dvalues0 = torch.empty(dfv.shape, device=q.device, dtype=values_dtype)
        bv = SCAN_BV
        nv = tr.cdiv(width, bv)
        grid = (nc, b*h)
        slot = g.ndim == 4
        output_offset_kernel[(nc, b*h, nv)](os, do, off, t, h, m, width, c, nc, bm, bv, SCAN, num_warps=WARPS['offset'])
        affine_scan(fs, off, dfv, ds, dvalues0, True, None, bv, SCAN)
        f32 = lambda x: torch.empty(x.shape, device=x.device, dtype=torch.float32)
        dos, dov, dfs, dfv_map = f32(os), f32(ov), torch.empty_like(fs), torch.empty_like(fv)
        dv = torch.empty_like(v)
        block_mapgrad_kernel[grid](ov, fv, v, tape, ds, do, dos, dov, dfs, dfv_map, dv,
                                   t, h, m, width, c, nc, bm, MAPGRAD_BV, num_warps=WARPS['mapgrad'])
        ckc, cnc, cgs = torch.empty_like(kc), torch.empty_like(ncm), torch.empty_like(gk)
        dwsp, dwvp, djn = f32(hk), f32(hn), f32(ncm)
        vec = torch.empty((b*h*nc, c+bm), device=q.device, dtype=torch.float32)
        proj_bwd_a_kernel[grid](g, beta, hk, tt, peo, pen, dfs, dfv_map, dwsp, dwvp, djn, vec,
                                t, h, m, c, nc, bm, slot, num_warps=WARPS['proj_bwd'])
        proj_bwd_b_kernel[grid](g, beta, peo, pen, pinv, kc, ncm, gk, dfs, djn, ckc, cnc, cgs,
                                t, h, m, c, nc, bm, tau, slot, num_warps=WARPS['proj_bwd'])
        chk, chns, crk, crn = torch.empty_like(hk), torch.empty_like(hn), torch.empty_like(rk), torch.empty_like(rn)
        dws, dwv, dgam, de = f32(hk), f32(hn), f32(hk), f32(hn)
        out_bwd_kernel[grid](g, beta, hk, rk, rn, tt, dos, dov, dwsp, dwvp, dws, dwv, dgam, de, crk, crn,
                             t, h, m, c, nc, bm, tau, slot, num_warps=WARPS['ut_bwd'])
        dg, db = torch.empty_like(g), torch.empty_like(beta)
        ut_bwd_kernel[grid](g, beta, hk, hn, tt, dws, dwv, dgam, de, chk, chns, db,
                            t, h, m, c, nc, bm, tau, slot, num_warps=WARPS['ut_bwd'])
        decay_bwd_kernel[grid](g, beta, dgam, de, vec, dg, t, h, m, c, nc, bm, slot, num_warps=4)
        dqn = torch.empty_like(qn)
        kb = min(bd, KEY_BLOCK)
        bu = tr.next_power_of_2(m+c)
        du = torch.empty((b*h*nc, bu, d), device=q.device, dtype=torch.float32)
        keygrad_kernel[(tr.cdiv(d, kb), nc, b*h)](qn, pool, chk, chns, crk, crn, ckc, cnc, cgs, dqn, du,
                                                  t, h, m, d, c, nc, bm, kb, bu, num_warps=WARPS['keygrad'], num_stages=1)
        dpool = torch.empty_like(pool)
        pool_grad_kernel[(tr.cdiv(t+m, 64), b*h)](du, dpool, t, m, d, c, nc, bu, 64, bd, num_warps=4)
        final_rows_kernel[(b*h, m)](pool, pool, dpool, dfk, t, m, d, bd, True)
        dq, dk, dk0 = torch.empty_like(q), torch.empty_like(k), torch.empty_like(keys0)
        normalize_bwd_kernel[(t+m, b*h)](q, k, keys0, dqn, dpool, dq, dk, dk0, t, h, m, d, bd)
        return dq, dk, dv, dg, db, dk0, dvalues0, None, None, None


def block_komp_fast(q, k, v, g, beta, initial_state, chunk_size=64, temperature=0.5, ridge=1e-4):
    """Same contract as block_komp.block_komp; staged kernels.

    g is the log decay: [B,T,H] scalar per head, or [B,T,H,M] per ring slot (the atom at sequence
    position p, initial atoms at -M..-1, lives in slot p mod M, matching cyclic slot routing).
    """
    keys0, values0 = initial_state
    if g.shape not in (q.shape[:3], (*q.shape[:3], keys0.shape[-2])):
        raise ValueError('g must have shape [B,T,H] or [B,T,H,M]')
    if chunk_size not in (16, 32, 64) or keys0.shape[-2] > 64:
        raise ValueError('chunk_size must be 16, 32 or 64 and at most 64 slots are supported')
    inputs = [x.contiguous() for x in (q, k, v, g, beta, keys0, values0)]
    with torch.cuda.device(q.device):
        out, keys, values = _BlockFast.apply(*inputs, chunk_size, temperature, ridge)
    return out, (keys, values)
