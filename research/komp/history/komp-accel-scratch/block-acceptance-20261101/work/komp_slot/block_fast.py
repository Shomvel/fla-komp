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
from .block_komp import _rows, _ex, _chunk_setup, _decay, _tiles, block_mapgrad_kernel, final_rows_kernel
from .utils import zero

ATT = tl.constexpr('tf32')
PROJ = tl.constexpr('tf32x3')
UT = tl.constexpr('ieee')
WARPS = {'att': 4, 'proj': 4, 'ut': 4, 'compose': 4, 'mapgrad': 4, 'proj_bwd': 4, 'ut_bwd': 8, 'keygrad': 4,
         'offset': 4, 'output': 4}
KEY_BLOCK = 64
SCAN = 'tf32'   # state scans and outputs: BF16-level target
SCAN_BV = 64


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
    t, n, valid, tok, g, beta = _chunk_setup(G, Beta, chunk, bh, T, H, C)
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
def _blk(P, i: tl.constexpr, j: tl.constexpr, C: tl.constexpr):
    o = tl.arange(0, 16)
    return tl.load(P+(16*i+o[:, None])*C+16*j+o[None, :])


@tr.jit
def _diag_inv(A):
    # (I + A)^-1 for a strictly lower 16x16 block, row-wise forward substitution.
    o = tl.arange(0, 16)
    Ai = -tl.where(o[:, None] > o[None, :], A, 0.)
    for i in range(2, 16):
        row = tl.sum(tl.where(o[:, None] == i, Ai, 0.), 0)
        row += tl.sum(row[:, None]*Ai, 0)
        Ai = tl.where(o[:, None] == i, row[None, :], Ai)
    return Ai+(o[:, None] == o[None, :]).to(tl.float32)


@tr.jit
def _put(P, x, i: tl.constexpr, j: tl.constexpr, C: tl.constexpr):
    o = tl.arange(0, 16)
    tl.store(P+(16*i+o[:, None])*C+16*j+o[None, :], x)


@tr.jit
def ut_kernel(G, Beta, HN, AA, TT, T: tl.constexpr, H: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
              NB: tl.constexpr):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, g, beta = _chunk_setup(G, Beta, chunk, bh, T, H, C)
    gc, gam, E, Dl, Dp, gC, dend, le, lt = _decay(g, valid, n, C)
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
                   M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, g, beta = _chunk_setup(G, Beta, chunk, bh, T, H, C)
    gc, gam, E, Dl, Dp, gC, dend, le, lt = _decay(g, valid, n, C)
    idx = bh*NC+chunk
    a = tl.arange(0, BM)
    Tm = _ld(TT, idx, C, C, C, C)
    WS = -tl.dot(Tm, (beta*gam)[:, None]*_ld(HK, idx, C, M, C, BM), input_precision=ATT)
    WV = Tm*beta[None, :]
    Rd = Dp*_ld(RN, idx, C, C, C, C)
    _st(OS, idx, gam[:, None]*_ld(RK, idx, C, M, C, BM)+tl.dot(Rd, WS, input_precision=ATT), C, M, C, BM)
    _st(OV, idx, tl.where(le, tl.dot(Rd, WV, input_precision=ATT), 0.), C, C, C, C)
    Jo = (a[None, :] == n+a[:, None]).to(tl.float32)+_ld(PEO, idx, M, M, BM, BM)
    Jn = ((M+t)[None, :] == n+a[:, None]).to(tl.float32)+_ld(PEN, idx, M, C, BM, C)
    _st(FS, idx, tl.exp(gC)*Jo+tl.dot(Jn, dend[:, None]*WS, input_precision=PROJ), M, M, BM, BM)
    _st(FV, idx, tl.dot(Jn, dend[:, None]*WV, input_precision=PROJ), M, C, BM, C)


@tr.jit
def proj_bwd_kernel(G, Beta, HK, TT, PEO, PEN, PINV, KC, NCM, GK, DFS, DFV, CKC, CNC, CGS, DWSP, DWVP, VEC,
                    T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                    BM: tl.constexpr, TAU: tl.constexpr):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, g, beta = _chunk_setup(G, Beta, chunk, bh, T, H, C)
    gc, gam, E, Dl, Dp, gC, dend, le, lt = _decay(g, valid, n, C)
    idx = bh*NC+chunk
    a = tl.arange(0, BM)
    am = a < M
    mo = (a < n) & am
    mn = ((M+t) < n) & valid
    Tm = _ld(TT, idx, C, C, C, C)
    WS = -tl.dot(Tm, (beta*gam)[:, None]*_ld(HK, idx, C, M, C, BM), input_precision=PROJ)
    WV = Tm*beta[None, :]
    WSd = dend[:, None]*WS
    WVd = dend[:, None]*WV
    PEo = _ld(PEO, idx, M, M, BM, BM)
    PEn = _ld(PEN, idx, M, C, BM, C)
    Jn = ((M+t)[None, :] == n+a[:, None]).to(tl.float32)+PEn
    dFS = _ld(DFS, idx, M, M, BM, BM)
    dFV = _ld(DFV, idx, M, C, BM, C)
    egC = tl.exp(gC)
    dgC = egC*tl.sum(tl.sum(dFS*((a[None, :] == n+a[:, None]).to(tl.float32)+PEo), 1), 0)
    dWSd = tl.dot(tl.trans(Jn), dFS, input_precision=PROJ)
    dWVd = tl.dot(tl.trans(Jn), dFV, input_precision=PROJ)
    ddend = tl.sum(dWSd*WS, 1)+tl.sum(dWVd*WV, 1)
    _st(DWSP, idx, dend[:, None]*dWSd, C, M, C, BM)
    _st(DWVP, idx, dend[:, None]*dWVd, C, C, C, C)
    tl.store(VEC+idx*(C+1)+t, ddend*dend)
    tl.store(VEC+idx*(C+1)+C, dgC)
    dJn = tl.dot(dFS, tl.trans(WSd), input_precision=PROJ)+tl.dot(dFV, tl.trans(WVd), input_precision=PROJ)
    Pinv = _ld(PINV, idx, M, M, BM, BM)
    Zo = tl.dot(Pinv, tl.where(mo[None, :], egC*dFS, 0.), input_precision=PROJ)
    Zn = tl.dot(Pinv, tl.where(mn[None, :], dJn, 0.), input_precision=PROJ)
    _st(CKC, idx, tl.where(mo[None, :], Zo, 0.)*_ld(KC, idx, M, M, BM, BM)/TAU, M, M, BM, BM)
    _st(CNC, idx, tl.where(mn[None, :], Zn, 0.)*_ld(NCM, idx, M, C, BM, C)/TAU, M, C, BM, C)
    cG = -(tl.dot(Zo, tl.trans(PEo), input_precision=PROJ)+tl.dot(Zn, tl.trans(PEn), input_precision=PROJ))*_ld(GK, idx, M, M, BM, BM)/TAU
    _st(CGS, idx, cG+tl.trans(cG), M, M, BM, BM)


@tr.jit
def ut_bwd_kernel(G, Beta, HK, HN, RK, RN, TT, DOS, DOV, DWSP, DWVP, VEC, CHK, CHNS, CRK, CRN, DG, DB,
                  T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                  BM: tl.constexpr, TAU: tl.constexpr):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, g, beta = _chunk_setup(G, Beta, chunk, bh, T, H, C)
    gc, gam, E, Dl, Dp, gC, dend, le, lt = _decay(g, valid, n, C)
    idx = bh*NC+chunk
    Tm = _ld(TT, idx, C, C, C, C)
    Hk = _ld(HK, idx, C, M, C, BM)
    bg = beta*gam
    WS = -tl.dot(Tm, bg[:, None]*Hk, input_precision=ATT)
    WV = Tm*beta[None, :]
    Rn = _ld(RN, idx, C, C, C, C)
    Rd = Dp*Rn
    dOS = _ld(DOS, idx, C, M, C, BM)
    dOV = _ld(DOV, idx, C, C, C, C)
    dWS = _ld(DWSP, idx, C, M, C, BM)+tl.dot(tl.trans(Rd), dOS, input_precision=ATT)
    dWV = _ld(DWVP, idx, C, C, C, C)+tl.dot(tl.trans(Rd), dOV, input_precision=ATT)
    Rk = _ld(RK, idx, C, M, C, BM)
    dgam = tl.sum(dOS*Rk, 1)
    _st(CRK, idx, gam[:, None]*dOS*Rk/TAU, C, M, C, BM)
    dRd = tl.dot(dOS, tl.trans(WS), input_precision=ATT)+tl.dot(dOV, tl.trans(WV), input_precision=ATT)
    _st(CRN, idx, dRd*Dp*Rn/TAU, C, C, C, C)
    dE = tl.where(le, dRd*Rn, 0.)
    TtdWS = tl.dot(tl.trans(Tm), dWS, input_precision=ATT)
    _st(CHK, idx, -bg[:, None]*TtdWS*Hk/TAU, C, M, C, BM)
    dbg = -tl.sum(TtdWS*Hk, 1)
    dbeta = dbg*gam+tl.sum(dWV*Tm, 0)
    dgam += dbg*beta
    dT = -tl.dot(dWS, tl.trans(bg[:, None]*Hk), input_precision=ATT)+dWV*beta[None, :]
    dL = -tl.where(lt, tl.dot(tl.dot(tl.trans(Tm), dT, input_precision=ATT), tl.trans(Tm), input_precision=ATT), 0.)
    Hn = _ld(HN, idx, C, C, C, C)
    dbeta += tl.sum(dL*Dl*Hn, 1)
    cHn = beta[:, None]*dL*Dl*Hn/TAU
    _st(CHNS, idx, cHn+tl.trans(cHn), C, C, C, C)
    dE += tl.where(lt, beta[:, None]*dL*Hn, 0.)
    dEE = dE*E
    vec = VEC+idx*(C+1)
    ddd = tl.load(vec+t)
    dGc = dgam*gam+tl.sum(dEE, 1)-tl.sum(dEE, 0)-ddd
    dGC = tl.load(vec+C)+tl.sum(ddd, 0)
    dGc += tl.where(t == n-1, dGC, 0.)
    tl.store(DG+tok, tl.sum(dGc, 0)-tl.cumsum(dGc, 0)+dGc, valid)
    tl.store(DB+tok, dbeta, valid)


@tr.jit
def keygrad_kernel(QN, Pool, CHK, CHN, CRK, CRN, CKC, CNC, CG, DQN, DPool,
                   T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                   NC: tl.constexpr, BM: tl.constexpr, BDK: tl.constexpr, WHICH: tl.constexpr):
    # WHICH: 0 dQ, 1 new keys, 2 old dictionary, 3 kept atoms. Each launch loads only its factors.
    chunk, bh, d0 = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2)*BDK
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
    if WHICH == 0:
        dq = (tl.dot(_ld(CRK, idx, C, M, C, BM), _rows(Pool, rbase, M, BM, D, d0, BDK), input_precision=ATT)
              + tl.dot(_ld(CRN, idx, C, C, C, C), _rows(Pool, rbase+M, n, C, D, d0, BDK), input_precision=ATT))
        tl.store(DQN+tok[:, None]*D+d[None, :], dq, valid[:, None] & dmask)
    elif WHICH == 1:
        q = tl.load(QN+tok[:, None]*D+d[None, :], valid[:, None] & dmask, 0)
        dkn = (tl.dot(_ld(CHK, idx, C, M, C, BM), _rows(Pool, rbase, M, BM, D, d0, BDK), input_precision=ATT)
               + tl.dot(_ld(CHN, idx, C, C, C, C), _rows(Pool, rbase+M, n, C, D, d0, BDK), input_precision=ATT)
               + tl.dot(tl.trans(_ld(CRN, idx, C, C, C, C)), q, input_precision=ATT)
               + tl.dot(tl.trans(_ld(CNC, idx, M, C, BM, C)), _rows(Pool, rbase+n, M, BM, D, d0, BDK), input_precision=PROJ))
        tl.atomic_add(DPool+(rbase+M+t[:, None])*D+d[None, :], dkn, valid[:, None] & dmask, sem='relaxed')
    elif WHICH == 2:
        q = tl.load(QN+tok[:, None]*D+d[None, :], valid[:, None] & dmask, 0)
        dko = (tl.dot(tl.trans(_ld(CHK, idx, C, M, C, BM)), _rows(Pool, rbase+M, n, C, D, d0, BDK), input_precision=ATT)
               + tl.dot(tl.trans(_ld(CRK, idx, C, M, C, BM)), q, input_precision=ATT)
               + tl.dot(tl.trans(_ld(CKC, idx, M, M, BM, BM)), _rows(Pool, rbase+n, M, BM, D, d0, BDK), input_precision=PROJ))
        tl.atomic_add(DPool+(rbase+a[:, None])*D+d[None, :], dko, am[:, None] & dmask, sem='relaxed')
    else:
        dkk = (tl.dot(_ld(CKC, idx, M, M, BM, BM), _rows(Pool, rbase, M, BM, D, d0, BDK), input_precision=PROJ)
               + tl.dot(_ld(CNC, idx, M, C, BM, C), _rows(Pool, rbase+M, n, C, D, d0, BDK), input_precision=PROJ)
               + tl.dot(_ld(CG, idx, M, M, BM, BM), _rows(Pool, rbase+n, M, BM, D, d0, BDK), input_precision=PROJ))
        tl.atomic_add(DPool+(rbase+n+a[:, None])*D+d[None, :], dkk, am[:, None] & dmask, sem='relaxed')


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
        grid = (nc, b*h)
        att_kernel[grid](qn, pool, g, beta, hk, hn, rk, rn, t, h, m, d, c, nc, bm, bdk, tau, num_warps=WARPS['att'], num_stages=1)
        proj_kernel[grid](pool, kc, ncm, gk, pinv, peo, pen, t, h, m, d, c, nc, bm, bdk, tau, ridge,
                          num_warps=WARPS['proj'], num_stages=1)
        ut_kernel[grid](g, beta, hn, aa, tt, t, h, c, nc, c//16, num_warps=WARPS['ut'])
        compose_kernel[grid](g, beta, hk, rk, rn, tt, peo, pen, os, ov, fs, fv, t, h, m, c, nc, bm, num_warps=WARPS['compose'])
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
        output_offset_kernel[(nc, b*h, nv)](os, do, off, t, h, m, width, c, nc, bm, bv, SCAN, num_warps=WARPS['offset'])
        affine_scan(fs, off, dfv, ds, dvalues0, True, None, bv, SCAN)
        f32 = lambda x: torch.empty(x.shape, device=x.device, dtype=torch.float32)
        dos, dov, dfs, dfv_map = f32(os), f32(ov), torch.empty_like(fs), torch.empty_like(fv)
        dv = torch.empty_like(v)
        block_mapgrad_kernel[grid](ov, fv, v, tape, ds, do, dos, dov, dfs, dfv_map, dv,
                                   t, h, m, width, c, nc, bm, 32, num_warps=WARPS['mapgrad'])
        ckc, cnc, cgs = torch.empty_like(kc), torch.empty_like(ncm), torch.empty_like(gk)
        dwsp, dwvp = f32(hk), f32(hn)
        vec = torch.empty((b*h*nc, c+1), device=q.device, dtype=torch.float32)
        proj_bwd_kernel[grid](g, beta, hk, tt, peo, pen, pinv, kc, ncm, gk, dfs, dfv_map, ckc, cnc, cgs, dwsp, dwvp, vec,
                              t, h, m, c, nc, bm, tau, num_warps=WARPS['proj_bwd'])
        chk, chns, crk, crn = torch.empty_like(hk), torch.empty_like(hn), torch.empty_like(rk), torch.empty_like(rn)
        dg, db = torch.empty_like(g), torch.empty_like(beta)
        ut_bwd_kernel[grid](g, beta, hk, hn, rk, rn, tt, dos, dov, dwsp, dwvp, vec, chk, chns, crk, crn, dg, db,
                            t, h, m, c, nc, bm, tau, num_warps=WARPS['ut_bwd'])
        dqn = torch.empty_like(qn)
        dpool = zero(torch.empty_like(pool))
        kb = min(bd, KEY_BLOCK)
        for which in range(4):
            keygrad_kernel[(nc, b*h, tr.cdiv(d, kb))](qn, pool, chk, chns, crk, crn, ckc, cnc, cgs, dqn, dpool,
                                                      t, h, m, d, c, nc, bm, kb, which, num_warps=WARPS['keygrad'], num_stages=1)
        final_rows_kernel[(b*h, m)](pool, pool, dpool, dfk, t, m, d, bd, True)
        dq, dk, dk0 = torch.empty_like(q), torch.empty_like(k), torch.empty_like(keys0)
        normalize_bwd_kernel[(t+m, b*h)](q, k, keys0, dqn, dpool, dq, dk, dk0, t, h, m, d, bd)
        return dq, dk, dv, dg, db, dk0, dvalues0, None, None, None


def block_komp_fast(q, k, v, g, beta, initial_state, chunk_size=64, temperature=0.5, ridge=1e-4):
    """Same contract and mathematics as block_komp.block_komp; staged kernels."""
    keys0, values0 = initial_state
    if chunk_size not in (16, 32, 64) or keys0.shape[-2] > 64:
        raise ValueError('chunk_size must be 16, 32 or 64 and at most 64 slots are supported')
    inputs = [x.contiguous() for x in (q, k, v, g, beta, keys0, values0)]
    with torch.cuda.device(q.device):
        out, keys, values = _BlockFast.apply(*inputs, chunk_size, temperature, ridge)
    return out, (keys, values)
