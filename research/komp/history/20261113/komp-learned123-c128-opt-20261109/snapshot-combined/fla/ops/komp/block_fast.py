# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Block-KOMP, FLA-style pipeline: one focused kernel per stage, chunk-local matrices kept in HBM.

Forward  normalize (row blocks) | [dict_scan: learned eviction tables] | proj: Kc,Nc,Gk (tf32x3), Gram sweep
         inverse, PEo=P Kc, PEn=P Nc | att: Hk,Hn,Rk,Rn (key-width blocked cosines) | ut: (I+diag(beta)(D o Hn))^-1
         by 16x16 blocks (32x32 for C=128) | compose: OS,OV,FS,FV | scan: group compose, group scan, expand + outputs | final rows.
Backward adjoint scan (3 kernels) | mapgrad: dOS,dOV,dFS,dFV,dV | proj_bwd a/b: next-state maps, then through
         the Gram inverse | chunk_bwd: outputs, UT transform and log-decay gradients | keygrad: key-width
         blocked products (FIFO C >= M: Knew folded into Kn) | pool_norm_bwd: pool-row sums + normalize.
Attention side runs on BF16 tensor cores for BF16 inputs; the projection side stays tf32x3 / FP32.
Same mathematics as block_komp.py (v0), which stays as the single-kernel reference implementation.
"""
import math

import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable

from .block_komp import _ex, _rows, block_mapgrad_kernel, final_rows_kernel
from .geometry import normalize_bwd_kernel, normalize_kernel
from .geometry_chunk import _sweep_inverse
from .recurrence import output_kernel
from .scan import _load_map, affine_scan, group_scan_kernel, output_offset_kernel, value_offset_kernel
from .utils import zero

ATT = tl.constexpr('tf32')
PROJ = tl.constexpr('tf32x3')
UT = tl.constexpr('tf32x3')
WARPS = {'att': 4, 'proj': 4, 'ut': 2, 'compose': 4, 'mapgrad': 4, 'proj_bwd': 4, 'ut_bwd': 4, 'keygrad': 4,
         'chunk_fwd': 8, 'proj_bwd_fused': 4, 'chunk_bwd': 4, 'scan': 4,
         'offset': 4, 'output': 4, 'dict': 8, 'ut_slot': 4,
         # chunk_bwd spills ~300 registers per thread with per-slot decay at 4 warps (26 at 8); scalar decay
         # is still faster at 4 warps (2 CTAs/SM). The learned-path variants below spill at 4 warps as well.
         'chunk_bwd_slot': 8, 'proj_ix': 4, 'proj_bwd_ix': 4, 'chol': 1}
KEY_BLOCK = 64
SCAN = 'tf32'   # state scans and outputs: BF16-level target
SCAN_BV = 64
SCAN_GROUP = 0  # chunks per scan group; 0 = about sqrt(NC/2)
SCAN_SERIAL = 64  # one serial pass when batch-heads x value tiles >= this (0: always two-level)
SCAN_STAGES = 3
MAPGRAD_BV = 32
# Fusion switches; the defaults are the fastest combination measured on H100 (see BLOCK_KOMP.md).
FUSE_CHUNK_FWD = False   # att+ut+compose in one kernel (fewer launches, slower: 1 CTA/SM)
FUSE_SCAN = True         # offsets computed inside the scan, outputs written by the expand pass
FUSE_PROJ_BWD = False    # proj_bwd a+b (fused: fewer launches, slightly slower)
FUSE_CHUNK_BWD = True    # out_bwd+ut_bwd+decay_bwd
BF16 = True     # BF16 tensor cores on the attention side when inputs are BF16
RANK1_SWEEP = False  # rank-1 sweep (_rank1_sweep_inverse): 5% faster proj, fails the near-duplicate bound
FUSE_POOL_NORM = True    # pool-gradient sum (fifo) + normalization backward in one row-block kernel
MERGE_KEYGRAD = True     # fifo, C >= M: Knew products folded into Kn products (see _keygrad_merged)
ROW_NORMALIZE = True     # q/k normalization by row blocks instead of one program per row
TAPE_BF16 = True         # BF16 inputs: chunk-entry states and their adjoints stored in BF16 (mapgrad reads them as BF16)
SHARE_GRAM = True        # fifo, C == M: Knew = Kn, so proj also writes Hk, Hn of full chunks and att skips them
SWEEP_FUSED = False      # two-select Gram sweep (bitwise identical; measured slower, kept off)
QN_BF16 = True           # BF16 inputs: normalized queries stored in BF16 (only read by BF16 products; exact)
SKIP_ZERO = True         # SHARE: do not write the all-zero Nc / PEN of full chunks
SLOT_GC_CACHE = True     # per-slot fifo: ut stores the cumulative log decay, later kernels load it
KEYGRAD_DEST = False     # SHARE, T % C == 0: key gradients by destination rows (bitwise identical; measured slower, kept off)
OVERLAP_DICT = True      # learned: dictionary-free att part + ut on a side stream while dict_scan/proj run
DECAY_SLOT_REDUCE = True  # C multiple of M: contract one-hot ring columns by direct summation
DOV_BF16 = True          # BF16 inputs: dOV (read only by BF16 products in chunk_bwd) stored in BF16; exact
# blocked Cholesky + blocked triangular inverse (chol_kernel): measured slower, fails near-duplicate dk
CHOL_INV = False
_SIDE = {}


def _side_stream(device):
    if device not in _SIDE:
        _SIDE[device] = torch.cuda.Stream(device)
    return _SIDE[device]


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
def _dict(Pool, OLD, bh, chunk, nxt, rbase, n, d0, T: tl.constexpr, M: tl.constexpr, BM: tl.constexpr,
          D: tl.constexpr, BDK: tl.constexpr, NC: tl.constexpr, IDX: tl.constexpr):
    """Key block of the dictionary entering the chunk (nxt=0) or kept after it (nxt=1).
    FIFO: contiguous pool rows. IDX: physical-slot table OLD[bh, chunk(+1), slot] of pool rows."""
    if IDX:
        r = tl.arange(0, BM)
        rows = tl.load(OLD+(bh*(NC+1)+chunk+nxt)*M+r, r < M, 0)
        d = d0+tl.arange(0, BDK)
        return tl.load(Pool+(bh*(T+M)+rows[:, None])*D+d[None, :], (r[:, None] < M) & (d[None, :] < D), 0)
    if nxt:
        return _rows(Pool, rbase+n, M, BM, D, d0, BDK)
    return _rows(Pool, rbase, M, BM, D, d0, BDK)


@tr.jit
def _sel(CAND, bh, chunk, n, M: tl.constexpr, C: tl.constexpr, BM: tl.constexpr, NC: tl.constexpr, IDX: tl.constexpr):
    """Selection of kept atoms into next slots and eviction masks over the old slots / new atoms."""
    a = tl.arange(0, BM)
    t = tl.arange(0, C)
    if IDX:
        cand = tl.load(CAND+(bh*NC+chunk)*M+a, a < M, -1)
        selo = (cand[:, None] == a[None, :]) & (a < M)[None, :]
        seln = cand[:, None] == M+t[None, :]
        evo = (a < M) & (tl.sum(selo.to(tl.int32), 0) == 0)
        evn = (t < n) & (tl.sum(seln.to(tl.int32), 0) == 0)
    else:
        selo = a[None, :] == n+a[:, None]
        seln = (M+t)[None, :] == n+a[:, None]
        evo = (a < n) & (a < M)
        evn = ((M+t) < n) & (t < n)
    return selo.to(tl.float32), seln.to(tl.float32), evo, evn


@tr.jit
def att_kernel(QN, Pool, OLD, G, Beta, HK, HN, RK, RN, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
               D: tl.constexpr, C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BDK: tl.constexpr,
               TAU: tl.constexpr, IDX: tl.constexpr, BF: tl.constexpr = False, SHARE: tl.constexpr = False,
               PART: tl.constexpr = 0):
    # PART 0: all four maps; 1: the dictionary-free Hn, Rn; 2: Hk, Rk (needs the dictionary).
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t = tl.arange(0, C)
    n = tl.minimum(C, T-chunk*C)
    own = n < 0
    if SHARE:
        own = n == C   # Hk, Hn of full chunks were written by proj_kernel
    valid = t < n
    tok = ((bh//H)*T+chunk*C+t)*H+bh % H
    rbase = bh*(T+M)+chunk*C
    hk = tl.zeros((C, BM), tl.float32)
    hn = tl.zeros((C, C), tl.float32)
    rk = tl.zeros((C, BM), tl.float32)
    rn = tl.zeros((C, C), tl.float32)
    for d0 in range(0, D, BDK):
        d = d0+tl.arange(0, BDK)
        Kn = _rows(Pool, rbase+M, n, C, D, d0, BDK)
        Q = tl.load(QN+tok[:, None]*D+d[None, :], valid[:, None] & (d[None, :] < D), 0)
        # Runtime branches only where SHARE needs them: a runtime `if` around the accumulators in every
        # iteration made the non-SHARE kernel ~10x slower (8k, C=32 < M=64).
        if PART == 1:
            hn += _adot(Kn, tl.trans(Kn), BF)
            rn += _adot(Q, tl.trans(Kn), BF)
        else:
            Kold = _dict(Pool, OLD, bh, chunk, 0, rbase, n, d0, T, M, BM, D, BDK, NC, IDX)
            if PART == 2:
                hk += _adot(Kn, tl.trans(Kold), BF)
                rk += _adot(Q, tl.trans(Kold), BF)
            else:
                if SHARE:
                    if not own:
                        hk += _adot(Kn, tl.trans(Kold), BF)
                        hn += _adot(Kn, tl.trans(Kn), BF)
                else:
                    hk += _adot(Kn, tl.trans(Kold), BF)
                    hn += _adot(Kn, tl.trans(Kn), BF)
                rk += _adot(Q, tl.trans(Kold), BF)
                rn += _adot(Q, tl.trans(Kn), BF)
    vf = valid.to(tl.float32)
    idx = bh*NC+chunk
    if PART != 1:
        if not own:
            _st(HK, idx, _ex(hk, TAU)*vf[:, None], C, M, C, BM)
        _st(RK, idx, _ex(rk, TAU)*vf[:, None], C, M, C, BM)
    if PART != 2:
        if not own:
            _st(HN, idx, _ex(hn, TAU)*vf[:, None]*vf[None, :], C, C, C, C)
        _st(RN, idx, _ex(rn, TAU)*vf[:, None]*vf[None, :], C, C, C, C)


@tr.jit
def _sweep_inverse_fused(gram, M: tl.constexpr, BM: tl.constexpr):
    # geometry_chunk._sweep_inverse with the pivot row (including the -1/p diagonal) and column written
    # by two selects instead of three plus a mask: the same values, bitwise.
    r = tl.arange(0, BM)
    a = gram
    for j in range(M):
        col = tl.sum(tl.where(r[None, :] == j, a, 0.), 1)
        piv = tl.sum(tl.where(r == j, col, 0.), 0)
        c = col/piv
        rowv = tl.where(r == j, -1./piv, c)
        a = a-col[:, None]*c[None, :]
        a = tl.where(r[:, None] == j, rowv[None, :], tl.where(r[None, :] == j, c[:, None], a))
    valid = (r[:, None] < M) & (r[None, :] < M)
    return tl.where(valid, -a, 0.)


@tr.jit
def _rank1_sweep_inverse(gram, M: tl.constexpr, BM: tl.constexpr):
    # The symmetric sweep written as one rank-1 update per pivot: with x = a[:, j] - e_j and p = a[j, j],
    # a - x x^T / p gives the swept row/column j (a[i, j] / p) and every other entry exactly, and leaves
    # 2 - 1/p instead of -1/p on the pivot. That entry is never read again (later columns do not include
    # it) and is only shifted by later updates, so the +2 offsets are removed once at the end:
    # inverse = 2 I - a. No per-step row/column overwrites.
    r = tl.arange(0, BM)
    a = gram
    for j in range(M):
        col = tl.sum(tl.where(r[None, :] == j, a, 0.), 1)
        piv = tl.sum(tl.where(r == j, col, 0.), 0)
        x = tl.where(r == j, col-1., col)
        a = a-x[:, None]*(x/piv)[None, :]
    valid = (r[:, None] < M) & (r[None, :] < M)
    return tl.where(valid, tl.where(r[:, None] == r[None, :], 2.-a, -a), 0.)


@tr.jit
def _chol_inv16(A):
    # 16x16 SPD tile -> (L, L^-1). Right-looking Cholesky; the inverse is accumulated alongside as
    # X <- M_j^-1 X with M_j the identity whose column j is replaced by l_j (L = M_0 M_1 ... M_15).
    r = tl.arange(0, 16)
    a = A
    L = tl.zeros((16, 16), tl.float32)
    X = (r[:, None] == r[None, :]).to(tl.float32)
    for j in range(16):
        col = tl.sum(tl.where(r[None, :] == j, a, 0.), 1)
        d = tl.sqrt(tl.sum(tl.where(r == j, col, 0.), 0))
        l = tl.where(r >= j, col/d, 0.)
        L = tl.where(r[None, :] == j, l[:, None], L)
        a = a-l[:, None]*l[None, :]
        xj = tl.sum(tl.where(r[:, None] == j, X, 0.), 0)
        X = X-(tl.where(r == j, l-1., l)/d)[:, None]*xj[None, :]
    return L, X


@tr.jit
def _b16(P, i, j, S: tl.constexpr):
    r = tl.arange(0, 16)
    return tl.load(P+(i*16+r[:, None])*S+j*16+r[None, :])


@tr.jit
def _p16(P, x, i, j, S: tl.constexpr):
    r = tl.arange(0, 16)
    tl.store(P+(i*16+r[:, None])*S+j*16+r[None, :], x)


@tr.jit
def chol_kernel(CS, KC, NCM, PINV, PEO, PEN, T: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                BM: tl.constexpr, NB: tl.constexpr, SHARE: tl.constexpr, SKIPZ: tl.constexpr):
    # Pinv = (G + ridge)^-1 for one chunk from the padded BM x BM matrix that proj_kernel wrote to CS[idx, 0]:
    # left-looking 16-blocked Cholesky (Schur updates L L^T stay bounded, unlike explicit block inverses),
    # L_kk^-1 written into the consumed G(k,k) slot, blocked forward substitution for the rest of L^-1,
    # Pinv = L^-T L^-1, then PEo = Pinv Kc (and PEn = Pinv Nc). One warp: every reduction is intra-warp.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    idx = bh*NC+chunk
    n = tl.minimum(C, T-chunk*C)
    S = CS+idx*2*BM*BM
    L = S+BM*BM
    for k in tl.static_range(NB):
        akk = _b16(S, k, k, BM)
        for j in tl.static_range(k):
            lkj = _b16(L, k, j, BM)
            akk -= tl.dot(lkj, tl.trans(lkj), input_precision=PROJ)
        lkk, ikk = _chol_inv16(akk)
        tl.debug_barrier()
        _p16(L, lkk, k, k, BM)
        _p16(S, ikk, k, k, BM)
        for i in tl.static_range(k+1, NB):
            aik = _b16(S, i, k, BM)
            for j in tl.static_range(k):
                aik -= tl.dot(_b16(L, i, j, BM), tl.trans(_b16(L, k, j, BM)), input_precision=PROJ)
            _p16(L, tl.dot(aik, tl.trans(ikk), input_precision=PROJ), i, k, BM)
        tl.debug_barrier()
    for i in tl.static_range(1, NB):
        for k in tl.static_range(i):
            acc = tl.zeros((16, 16), tl.float32)
            for j in tl.static_range(k, i):
                acc += tl.dot(_b16(L, i, j, BM), _b16(S, j, k, BM), input_precision=PROJ)
            _p16(S, -tl.dot(_b16(S, i, i, BM), acc, input_precision=PROJ), i, k, BM)
            tl.debug_barrier()
    a = tl.arange(0, BM)
    t = tl.arange(0, C)  # noqa: F841
    am = a < M
    valid2 = am[:, None] & am[None, :]
    li = tl.load(S+a[:, None]*BM+a[None, :])
    li = tl.where((a[:, None]//16) >= (a[None, :]//16), li, 0.)
    Pinv = tl.where(valid2, tl.dot(tl.trans(li), li, input_precision=PROJ), 0.)
    _st(PINV, idx, Pinv, M, M, BM, BM)
    _st(PEO, idx, tl.dot(Pinv, _ld(KC, idx, M, M, BM, BM), input_precision=PROJ), M, M, BM, BM)
    if SHARE:
        if SKIPZ:
            if n != C:
                _st(PEN, idx, tl.zeros((BM, C), tl.float32), M, C, BM, C)
        else:
            _st(PEN, idx, tl.zeros((BM, C), tl.float32), M, C, BM, C)
    else:
        _st(PEN, idx, tl.dot(Pinv, _ld(NCM, idx, M, C, BM, C), input_precision=PROJ), M, C, BM, C)


@tr.jit
def proj_kernel(Pool, OLD, CAND, KC, NCM, GK, PINV, PEO, PEN, HK, HN, T: tl.constexpr, H: tl.constexpr,
                M: tl.constexpr, D: tl.constexpr, C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr,
                BDK: tl.constexpr, TAU: tl.constexpr, RIDGE: tl.constexpr, IDX: tl.constexpr, R1: tl.constexpr,
                SHARE: tl.constexpr = False, FUSED: tl.constexpr = False, SKIPZ: tl.constexpr = False,
                CS=None, CHOL: tl.constexpr = False):
    # SHARE (fifo, C == M): no new atom is ever evicted, so Nc = 0 and its product is skipped; in a full
    # chunk Knew = Kn, so Kc and Gk are the cosines of Hk and Hn, written here for att_kernel.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    n = tl.minimum(C, T-chunk*C)
    rbase = bh*(T+M)+chunk*C
    kc = tl.zeros((BM, BM), tl.float32)
    nc = tl.zeros((BM, C), tl.float32)
    gk = tl.zeros((BM, BM), tl.float32)
    for d0 in range(0, D, BDK):
        Kold = _dict(Pool, OLD, bh, chunk, 0, rbase, n, d0, T, M, BM, D, BDK, NC, IDX)
        Knew = _dict(Pool, OLD, bh, chunk, 1, rbase, n, d0, T, M, BM, D, BDK, NC, IDX)
        kc += tl.dot(Knew, tl.trans(Kold), input_precision=PROJ)
        if not SHARE:
            Kn = _rows(Pool, rbase+M, n, C, D, d0, BDK)
            nc += tl.dot(Knew, tl.trans(Kn), input_precision=PROJ)
        gk += tl.dot(Knew, tl.trans(Knew), input_precision=PROJ)
    a = tl.arange(0, BM)
    t = tl.arange(0, C)  # noqa: F841
    am = a < M
    valid2 = am[:, None] & am[None, :]
    selo, seln, evo, evn = _sel(CAND, bh, chunk, n, M, C, BM, NC, IDX)
    Kc = tl.where(evo[None, :], _ex(kc, TAU), 0.)
    Nc = tl.where(evn[None, :], _ex(nc, TAU), 0.)
    Gk = tl.where(valid2, _ex(gk, TAU), 0.)
    Gr = Gk+tl.where((a[:, None] == a[None, :]) & valid2, RIDGE, 0.)
    idx = bh*NC+chunk
    if CHOL:
        # chol_kernel inverts later; padding rows/columns carry the identity.
        tl.store(CS+idx*2*BM*BM+a[:, None]*BM+a[None, :], Gr+tl.where((a[:, None] == a[None, :]) & ~am[:, None], 1., 0.))
        Pinv = Gr
    elif R1:
        Pinv = _rank1_sweep_inverse(Gr, M, BM)
    elif FUSED:
        Pinv = _sweep_inverse_fused(Gr, M, BM)
    else:
        Pinv = _sweep_inverse(Gr, M, BM)
    _st(KC, idx, Kc, M, M, BM, BM)
    if not SKIPZ:
        _st(NCM, idx, Nc, M, C, BM, C)   # (SKIPZ: SHARE, where Nc = 0 is never read)
    _st(GK, idx, Gk, M, M, BM, BM)
    if not CHOL:   # (CHOL: Pinv, PEo, PEn are written by chol_kernel)
        _st(PINV, idx, Pinv, M, M, BM, BM)
        _st(PEO, idx, tl.dot(Pinv, Kc, input_precision=PROJ), M, M, BM, BM)
    if SHARE:
        if n == C:
            _st(HK, idx, _ex(kc, TAU), C, M, C, BM)
            _st(HN, idx, Gk, C, C, C, C)
            if not SKIPZ and not CHOL:
                _st(PEN, idx, Nc, M, C, BM, C)
        elif not CHOL:
            _st(PEN, idx, Nc, M, C, BM, C)   # the partial chunk takes compose's general path
    elif not CHOL:
        _st(PEN, idx, tl.dot(Pinv, Nc, input_precision=PROJ), M, C, BM, C)


@tr.jit
def _slot_decay(G, Beta, chunk, bh, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
                BM: tl.constexpr, SLOT: tl.constexpr, IDX: tl.constexpr = False, GCP=None, gidx=0,
                CACHE: tl.constexpr = 0):
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
        if (M == BM) and (C % BM == 0):
            # C a multiple of M: new atom s goes to ring slot (o+s) mod M, i.e. column s mod M of the tile
            # loaded with rotated columns. FIFO old atom u sits in slot (o+u) mod M (the same rotation);
            # IDX old atoms are held by slot (no rotation). Loading g in those orders replaces gathers.
            # CACHE (fifo only): 1 = store the rotated cumulative log decay, 2 = load it instead of the scan.
            if (CACHE == 2) and (not IDX):
                gc = tl.load(GCP+gidx*C*BM+t[:, None]*BM+u[None, :])
            else:
                gc = tl.cumsum(tl.load(G+tok[:, None]*M+(o+u[None, :]) % M, valid[:, None] & um[None, :], 0).to(tl.float32), 0)
                if (CACHE == 1) and (not IDX):
                    tl.store(GCP+gidx*C*BM+t[:, None]*BM+u[None, :], gc)
            if IDX:
                Gam = tl.exp(tl.cumsum(tl.load(G+tok[:, None]*M+u[None, :],
                             valid[:, None] & um[None, :], 0).to(tl.float32), 0))
            else:
                Gam = tl.exp(gc)
            if C == BM:
                gcs = gc
            else:
                gcs = tl.reshape(tl.broadcast_to(gc[:, None, :], (C, C//BM, BM)), (C, C))
        else:
            g = tl.load(G+tok[:, None]*M+u[None, :], valid[:, None] & um[None, :], 0).to(tl.float32)
            gc = tl.cumsum(g, 0)
            # FIFO keeps old atoms in age order (atom u in ring slot (o+u) mod M); IDX holds them by slot.
            r_old = tl.where(um, u if IDX else (o+u) % M, 0)
            Gam = tl.exp(tl.gather(gc, tl.broadcast_to(r_old[None, :], (C, BM)), 1))
            gcs = tl.gather(gc, tl.broadcast_to(r_new[None, :], (C, C)), 1)
        if (CACHE == 2) and (not IDX) and (M == BM) and (C % BM == 0):
            # The diagonal, the last row and gend read straight from the cached (rotated) tile: the same
            # values as the masked reductions below, without the reductions.
            gp = GCP+gidx*C*BM
            diag = tl.load(gp+t*BM+t % BM)
            dend = tl.where(valid, tl.exp(tl.load(gp+(n-1)*BM+t % BM)-diag), 0.)
            gend = tl.exp(tl.load(gp+(n-1)*BM+u))
        else:
            diag = tl.sum(tl.where(t[:, None] == t[None, :], gcs, 0.), 0)
            last = t[:, None] == n-1
            dend = tl.where(valid, tl.exp(tl.sum(tl.where(last, gcs, 0.), 0)-diag), 0.)
            gend = tl.sum(tl.where(last, Gam, 0.), 0)
        Ex = tl.exp(tl.where(le, gcs-diag[None, :], 0.))
    else:
        # Scalar decay: every slot shares the token's cumulative log decay.
        gc = tl.cumsum(tl.load(G+tok, valid, 0).to(tl.float32), 0)
        # [C, 1]: broadcast at use (a materialized [C, BM] copy costs a register tile in every kernel).
        Gam = tl.exp(gc)[:, None]
        Ex = tl.exp(tl.where(le, gc[:, None]-gc[None, :], 0.))
        gC = tl.sum(tl.where(t == n-1, gc, 0.), 0)
        dend = tl.where(valid, tl.exp(gC-gc), 0.)
        gend = tl.exp(gC)+tl.zeros((BM,), tl.float32)
    return t, n, valid, tok, beta, Gam, Ex, tl.where(lt, Ex, 0.), tl.where(le, Ex, 0.), dend, gend, le, lt, o, r_new


@tr.jit
def _blk(P, i: tl.constexpr, j: tl.constexpr, C: tl.constexpr, BS: tl.constexpr = 16):
    o = tl.arange(0, BS)
    return tl.load(P+(BS*i+o[:, None])*C+BS*j+o[None, :])


@tr.jit
def _diag_inv(A, BS: tl.constexpr = 16):
    # Unit-lower block inverse by doubling: 16 rows for C<=64, 32 rows for C=128.
    o = tl.arange(0, BS)
    L = -tl.where(o[:, None] > o[None, :], A, 0.)
    inv = (o[:, None] == o[None, :]).to(tl.float32)
    for step in tl.static_range(5 if BS == 32 else 4):
        w = 1 << step
        cross = tl.where((o[:, None]//(2*w) == o[None, :]//(2*w)) & (o[:, None] %
                         (2*w) >= w) & (o[None, :] % (2*w) < w), L, 0.)
        inv += tl.dot(tl.dot(inv, cross, input_precision=PROJ), inv, input_precision=PROJ)
    return inv


@tr.jit
def _put(P, x, i: tl.constexpr, j: tl.constexpr, C: tl.constexpr, BS: tl.constexpr = 16):
    o = tl.arange(0, BS)
    tl.store(P+(BS*i+o[:, None])*C+BS*j+o[None, :], x)


@tr.jit
def _ut_solve(AA, TT, idx, C: tl.constexpr, NB: tl.constexpr):
    # Keep four blocks for C=128 by doubling the tile width, as in blocked triangular solves.
    # All intermediates remain FP32; only the final TT stores follow the input precision.
    BS: tl.constexpr = 32 if NB == 8 else 16
    A = AA+idx*C*C
    O = TT+idx*C*C
    z = tl.zeros((BS, BS), tl.float32)
    i11 = _diag_inv(_blk(A, 0, 0, C, BS), BS)
    _put(O, i11, 0, 0, C, BS)
    if NB >= 2:
        i22 = _diag_inv(_blk(A, 1, 1, C, BS), BS)
        i21 = -tl.dot(tl.dot(i22, _blk(A, 1, 0, C, BS), input_precision=UT), i11, input_precision=UT)
        _put(O, i22, 1, 1, C, BS)
        _put(O, i21, 1, 0, C, BS)
        _put(O, z, 0, 1, C, BS)
    if NB >= 4:
        i33 = _diag_inv(_blk(A, 2, 2, C, BS), BS)
        i44 = _diag_inv(_blk(A, 3, 3, C, BS), BS)
        a31, a32 = _blk(A, 2, 0, C, BS), _blk(A, 2, 1, C, BS)
        a41, a42, a43 = _blk(A, 3, 0, C, BS), _blk(A, 3, 1, C, BS), _blk(A, 3, 2, C, BS)
        i32 = -tl.dot(tl.dot(i33, a32, input_precision=UT), i22, input_precision=UT)
        i43 = -tl.dot(tl.dot(i44, a43, input_precision=UT), i33, input_precision=UT)
        i31 = -tl.dot(i33, tl.dot(a31, i11, input_precision=UT)+tl.dot(a32, i21, input_precision=UT), input_precision=UT)
        i42 = -tl.dot(i44, tl.dot(a42, i22, input_precision=UT)+tl.dot(a43, i32, input_precision=UT), input_precision=UT)
        i41 = -tl.dot(i44, tl.dot(a41, i11, input_precision=UT)+tl.dot(a42, i21, input_precision=UT)
                      + tl.dot(a43, i31, input_precision=UT), input_precision=UT)
        _put(O, i33, 2, 2, C, BS)
        _put(O, i44, 3, 3, C, BS)
        _put(O, i31, 2, 0, C, BS)
        _put(O, i32, 2, 1, C, BS)
        _put(O, i41, 3, 0, C, BS)
        _put(O, i42, 3, 1, C, BS)
        _put(O, i43, 3, 2, C, BS)
        for j in tl.static_range(4):
            for i in tl.static_range(4):
                if i < j and not (i < 2 and j < 2):
                    _put(O, z, i, j, C, BS)


@tr.jit
def ut_kernel(G, Beta, HN, AA, TT, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
              NC: tl.constexpr, BM: tl.constexpr, NB: tl.constexpr, SLOT: tl.constexpr, IDX: tl.constexpr, GCP=None, CACHE: tl.constexpr = 0):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(
        G, Beta, chunk, bh, T, H, M, C, BM, SLOT, IDX, GCP, bh*NC+chunk, CACHE)
    idx = bh*NC+chunk
    Hn = _ld(HN, idx, C, C, C, C)
    _st(AA, idx, beta[:, None]*Dl*Hn, C, C, C, C)
    tl.debug_barrier()
    _ut_solve(AA, TT, idx, C, NB)


@tr.jit
def chunk_fwd_kernel(QN, Pool, OLD, CAND, G, Beta, HK, HN, RK, RN, AA, TT, PEO, PEN, OS, OV, FS, FV,
                     T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                     NC: tl.constexpr, BM: tl.constexpr, BDK: tl.constexpr, TAU: tl.constexpr, NB: tl.constexpr,
                     SLOT: tl.constexpr, IDX: tl.constexpr, BF: tl.constexpr):
    """Fused att + ut + compose: kernel values, the in-chunk UT inverse and all four chunk maps."""
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(
        G, Beta, chunk, bh, T, H, M, C, BM, SLOT, IDX)
    rbase = bh*(T+M)+chunk*C
    hk = tl.zeros((C, BM), tl.float32)
    hn = tl.zeros((C, C), tl.float32)
    rk = tl.zeros((C, BM), tl.float32)
    rn = tl.zeros((C, C), tl.float32)
    for d0 in range(0, D, BDK):
        d = d0+tl.arange(0, BDK)
        Kold = _dict(Pool, OLD, bh, chunk, 0, rbase, n, d0, T, M, BM, D, BDK, NC, IDX)
        Kn = _rows(Pool, rbase+M, n, C, D, d0, BDK)
        Q = tl.load(QN+tok[:, None]*D+d[None, :], valid[:, None] & (d[None, :] < D), 0)
        hk += _adot(Kn, tl.trans(Kold), BF)
        hn += _adot(Kn, tl.trans(Kn), BF)
        rk += _adot(Q, tl.trans(Kold), BF)
        rn += _adot(Q, tl.trans(Kn), BF)
    vf = valid.to(tl.float32)
    idx = bh*NC+chunk
    Hk = _ex(hk, TAU)*vf[:, None]
    Hn = _ex(hn, TAU)*vf[:, None]*vf[None, :]
    Rk = _ex(rk, TAU)*vf[:, None]
    Rn = _ex(rn, TAU)*vf[:, None]*vf[None, :]
    _st(HK, idx, Hk, C, M, C, BM)
    _st(HN, idx, Hn, C, C, C, C)
    _st(RK, idx, Rk, C, M, C, BM)
    _st(RN, idx, Rn, C, C, C, C)
    _st(AA, idx, beta[:, None]*Dl*Hn, C, C, C, C)
    tl.debug_barrier()
    _ut_solve(AA, TT, idx, C, NB)
    tl.debug_barrier()
    a = tl.arange(0, BM)  # noqa: F841
    Tm = _ld(TT, idx, C, C, C, C)
    WS = -tl.dot(Tm, beta[:, None]*Gam*Hk, input_precision=ATT)
    WV = Tm*beta[None, :]
    Rd = Dp*Rn
    _st(OS, idx, Gam*Rk+tl.dot(Rd, WS, input_precision=ATT), C, M, C, BM)
    _st(OV, idx, tl.where(le, tl.dot(Rd, WV, input_precision=ATT), 0.), C, C, C, C)
    selo, seln, evo, evn = _sel(CAND, bh, chunk, n, M, C, BM, NC, IDX)
    Jn = seln+_ld(PEN, idx, M, C, BM, C)
    _st(FS, idx, (selo+_ld(PEO, idx, M, M, BM, BM))*gend[None, :] +
        tl.dot(Jn, dend[:, None]*WS, input_precision=PROJ), M, M, BM, BM)
    _st(FV, idx, tl.dot(Jn, dend[:, None]*WV, input_precision=PROJ), M, C, BM, C)


@tr.jit
def _adot(a, b, BF: tl.constexpr):
    # Attention-side products: BF16 tensor cores (FP32 accumulate) for BF16 inputs, else TF32.
    if BF:
        r = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        r = tl.dot(a, b, input_precision=ATT)
    return r


@tr.jit
def _offset(FO, Value, idx, bh, block, v, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr,
            C: tl.constexpr, BM: tl.constexpr, REV: tl.constexpr, P: tl.constexpr):
    # Affine offset of a chunk: FV V in the forward scan, OS^T dO in the adjoint scan.
    m = tl.arange(0, BM)
    c = tl.arange(0, C)
    tok = ((bh//H)*T+block*C+c)*H+bh % H
    val = tl.load(Value+tok[:, None]*V+v[None, :], (block*C+c[:, None] < T) & (v[None, :] < V), 0).to(tl.float32)
    if REV:
        f = tl.load(FO+idx*C*M+c[None, :]*M+m[:, None], m[:, None] < M, 0).to(tl.float32)
    else:
        f = tl.load(FO+idx*M*C+m[:, None]*C+c[None, :], m[:, None] < M, 0).to(tl.float32)
    return tl.dot(f, val, input_precision=P)


@tr.jit
def bscan_compose_kernel(F, FO, Value, GA, GC, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr,
                         C: tl.constexpr, NC: tl.constexpr, G: tl.constexpr, NG: tl.constexpr, BM: tl.constexpr,
                         BV: tl.constexpr, REV: tl.constexpr, NV: tl.constexpr, P: tl.constexpr):
    # Group composition with offsets computed in place (no offset buffer).
    g, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    m = tl.arange(0, BM)
    n = tl.minimum(G, NC-g*G)
    out = bh*NG+g
    if tile == NV:
        A = ((m[:, None] == m[None, :]) & (m[:, None] < M)).to(tl.float32)
        for s in range(n):
            block = NC-1-(g*G+s) if REV else g*G+s
            A = tl.dot(_load_map(F, bh*NC+block, M, BM, REV), A, input_precision=P)
        tl.store(GA+out*M*M+m[:, None]*M+m[None, :], A, (m[:, None] < M) & (m[None, :] < M))
    else:
        v = tile*BV+tl.arange(0, BV)
        acc = tl.zeros((BM, BV), tl.float32)
        for s in range(n):
            block = NC-1-(g*G+s) if REV else g*G+s
            idx = bh*NC+block
            acc = tl.dot(_load_map(F, idx, M, BM, REV), acc, input_precision=P) + \
                _offset(FO, Value, idx, bh, block, v, T, H, M, V, C, BM, REV, P)
        tl.store(GC+out*M*V+m[:, None]*V+v[None, :], acc, (m[:, None] < M) & (v[None, :] < V))


@tr.jit
def bscan_expand_kernel(F, FO, Value, Entry, Tape, OS, OV, Out, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
                        V: tl.constexpr, C: tl.constexpr, NC: tl.constexpr, G: tl.constexpr, NG: tl.constexpr,
                        BM: tl.constexpr, BV: tl.constexpr, REV: tl.constexpr, OUT: tl.constexpr, P: tl.constexpr,
                        BF: tl.constexpr, Final=None, FINAL: tl.constexpr = False):
    # Chunk-entry states of one group; the forward pass also writes the chunk outputs OS S0 + OV V.
    # FINAL (single group, G = NC): Entry is the initial state, and the state after the last chunk is
    # written to Final, i.e. a one-pass serial scan.
    g, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    m = tl.arange(0, BM)
    c = tl.arange(0, C)
    v = tile*BV+tl.arange(0, BV)
    vmask = (m[:, None] < M) & (v[None, :] < V)
    state = tl.load(Entry+(bh*NG+g)*M*V+m[:, None]*V+v[None, :], vmask, 0).to(tl.float32)
    n = tl.minimum(G, NC-g*G)
    for s in range(n):
        block = NC-1-(g*G+s) if REV else g*G+s
        idx = bh*NC+block
        tl.store(Tape+idx*M*V+m[:, None]*V+v[None, :], state, vmask)
        if OUT:
            tok = ((bh//H)*T+block*C+c)*H+bh % H
            cm = (block*C+c[:, None] < T) & (v[None, :] < V)
            val = tl.load(Value+tok[:, None]*V+v[None, :], cm, 0).to(tl.float32)
            os = tl.load(OS+idx*C*M+c[:, None]*M+m[None, :], m[None, :] < M, 0).to(tl.float32)
            ov = tl.load(OV+idx*C*C+c[:, None]*C+c[None, :]).to(tl.float32)
            tl.store(Out+tok[:, None]*V+v[None, :], _adot(os, state, BF)+_adot(ov, val, BF), cm)
        if FINAL:  # noqa: SIM114
            state = tl.dot(_load_map(F, idx, M, BM, REV), state, input_precision=P) + \
                _offset(FO, Value, idx, bh, block, v, T, H, M, V, C, BM, REV, P)
        elif s+1 < n:
            state = tl.dot(_load_map(F, idx, M, BM, REV), state, input_precision=P) + \
                _offset(FO, Value, idx, bh, block, v, T, H, M, V, C, BM, REV, P)
    if FINAL:
        tl.store(Final+bh*M*V+m[:, None]*V+v[None, :], state, vmask)


@tr.jit
def _compose_maps(FS, FV, PEN, idx, Jo, seln, WS, WV, dend, gend, M: tl.constexpr, C: tl.constexpr, BM: tl.constexpr):
    # Next-state maps FS = Jo diag(gend) + Jn diag(dend) WS, FV = Jn diag(dend) WV, Jn = seln + PEN.
    Jn = seln+_ld(PEN, idx, M, C, BM, C)
    _st(FS, idx, Jo*gend[None, :]+tl.dot(Jn, dend[:, None]*WS, input_precision=PROJ), M, M, BM, BM)
    _st(FV, idx, tl.dot(Jn, dend[:, None]*WV, input_precision=PROJ), M, C, BM, C)


@tr.jit
def compose_kernel(G, Beta, CAND, HK, RK, RN, TT, PEO, PEN, OS, OV, FS, FV, T: tl.constexpr, H: tl.constexpr,
                   M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, SLOT: tl.constexpr, IDX: tl.constexpr,
                   SHARE: tl.constexpr = False, GCP=None, CACHE: tl.constexpr = 0):
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(
        G, Beta, chunk, bh, T, H, M, C, BM, SLOT, IDX, GCP, bh*NC+chunk, CACHE)
    idx = bh*NC+chunk
    a = tl.arange(0, BM)  # noqa: F841
    Tm = _ld(TT, idx, C, C, C, C)
    WS = -tl.dot(Tm, beta[:, None]*Gam*_ld(HK, idx, C, M, C, BM), input_precision=ATT)
    WV = Tm*beta[None, :]
    Rd = Dp*_ld(RN, idx, C, C, C, C)
    _st(OS, idx, Gam*_ld(RK, idx, C, M, C, BM)+tl.dot(Rd, WS, input_precision=ATT), C, M, C, BM)
    _st(OV, idx, tl.where(le, tl.dot(Rd, WV, input_precision=ATT), 0.), C, C, C, C)
    selo, seln, evo, evn = _sel(CAND, bh, chunk, n, M, C, BM, NC, IDX)
    Jo = selo+_ld(PEO, idx, M, M, BM, BM)
    if SHARE:
        # C == M == BM. Full chunk: PEN = 0 and every new atom is kept in order, so Jn = I. (The shapes
        # only agree when C == BM, so this runtime branch must stay inside the constexpr one.)
        if n == C:
            _st(FS, idx, Jo*gend[None, :]+dend[:, None]*WS, M, M, BM, BM)
            _st(FV, idx, dend[:, None]*WV, M, C, BM, C)
        else:
            _compose_maps(FS, FV, PEN, idx, Jo, seln, WS, WV, dend, gend, M, C, BM)
    else:
        _compose_maps(FS, FV, PEN, idx, Jo, seln, WS, WV, dend, gend, M, C, BM)


@tr.jit
def proj_bwd_a_kernel(G, Beta, CAND, HK, TT, PEO, PEN, DFS, DFV, DWSP, DWVP, DJN, VEC,
                      T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                      BM: tl.constexpr, SLOT: tl.constexpr, IDX: tl.constexpr, SHARE: tl.constexpr = False, GCP=None, CACHE: tl.constexpr = 0):
    # Next-state maps FS = J_old diag(gend) + J_new diag(dend) WS, FV = J_new diag(dend) WV.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(
        G, Beta, chunk, bh, T, H, M, C, BM, SLOT, IDX, GCP, bh*NC+chunk, CACHE)
    idx = bh*NC+chunk
    a = tl.arange(0, BM)
    Tm = _ld(TT, idx, C, C, C, C)
    WS = -tl.dot(Tm, beta[:, None]*Gam*_ld(HK, idx, C, M, C, BM), input_precision=ATT)  # as in compose_kernel
    WV = Tm*beta[None, :]
    selo, seln, evo, evn = _sel(CAND, bh, chunk, n, M, C, BM, NC, IDX)
    dFS = _ld(DFS, idx, M, M, BM, BM)
    dFV = _ld(DFV, idx, M, C, BM, C)
    dgend = tl.sum(dFS*(selo+_ld(PEO, idx, M, M, BM, BM)), 0)
    if SHARE:
        # PEN = 0, so Jn is the selection of new atom t = n+a-M: Jn^T X is X shifted by M-n rows.
        dWSd = _shift(DFS, idx, M, M, t+M-n, a, (t < n)[:, None], False)
        dWVd = _shift(DFV, idx, M, C, t+M-n, t, (t < n)[:, None], False)
    else:
        Jn = seln+_ld(PEN, idx, M, C, BM, C)
        dWSd = tl.dot(tl.trans(Jn), dFS, input_precision=PROJ)
        dWVd = tl.dot(tl.trans(Jn), dFV, input_precision=PROJ)
    ddend = tl.sum(dWSd*WS, 1)+tl.sum(dWVd*WV, 1)
    _st(DWSP, idx, dend[:, None]*dWSd, C, M, C, BM)
    _st(DWVP, idx, dend[:, None]*dWVd, C, C, C, C)
    tl.store(VEC+idx*(C+BM)+t, ddend*dend)
    tl.store(VEC+idx*(C+BM)+C+a, dgend*gend)
    if not SHARE:
        # (SHARE: no new atom is evicted, so proj_bwd_b never reads dJn.)
        dJn = tl.dot(dFS, tl.trans(dend[:, None]*WS), input_precision=PROJ) + \
            tl.dot(dFV, tl.trans(dend[:, None]*WV), input_precision=PROJ)
        _st(DJN, idx, dJn, M, C, BM, C)


@tr.jit
def proj_bwd_b_kernel(G, Beta, CAND, PEO, PEN, PINV, KC, NCM, GK, DFS, DJN, CKC, CNC, CGS,
                      T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                      BM: tl.constexpr, TAU: tl.constexpr, SLOT: tl.constexpr, IDX: tl.constexpr,
                      SHARE: tl.constexpr = False, GCP=None, CACHE: tl.constexpr = 0):
    # Through the Gram inverse: Z = P dJ, dG = -(Z PE^T), then cosine gradients of Kc, Nc, Gk.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(
        G, Beta, chunk, bh, T, H, M, C, BM, SLOT, IDX, GCP, bh*NC+chunk, CACHE)
    idx = bh*NC+chunk
    a = tl.arange(0, BM)  # noqa: F841
    selo, seln, mo, mn = _sel(CAND, bh, chunk, n, M, C, BM, NC, IDX)
    Pinv = _ld(PINV, idx, M, M, BM, BM)
    Zo = tl.dot(Pinv, tl.where(mo[None, :], _ld(DFS, idx, M, M, BM, BM)*gend[None, :], 0.), input_precision=PROJ)
    _st(CKC, idx, tl.where(mo[None, :], Zo, 0.)*_ld(KC, idx, M, M, BM, BM)/TAU, M, M, BM, BM)
    cG = -tl.dot(Zo, tl.trans(_ld(PEO, idx, M, M, BM, BM)), input_precision=PROJ)
    if SHARE:
        # No new atom is evicted: Zn = 0, so CNC = 0 and Gk gets no PEN term.
        _st(CNC, idx, tl.zeros((BM, C), tl.float32), M, C, BM, C)
    else:
        Zn = tl.dot(Pinv, tl.where(mn[None, :], _ld(DJN, idx, M, C, BM, C), 0.), input_precision=PROJ)
        _st(CNC, idx, tl.where(mn[None, :], Zn, 0.)*_ld(NCM, idx, M, C, BM, C)/TAU, M, C, BM, C)
        cG -= tl.dot(Zn, tl.trans(_ld(PEN, idx, M, C, BM, C)), input_precision=PROJ)
    cG = cG*_ld(GK, idx, M, M, BM, BM)/TAU
    _st(CGS, idx, cG+tl.trans(cG), M, M, BM, BM)


@tr.jit
def out_bwd_kernel(G, Beta, HK, RK, RN, TT, DOS, DOV, DWSP, DWVP, DWS, DWV, DGAM, DE, CRK, CRN,
                   T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                   BM: tl.constexpr, TAU: tl.constexpr, SLOT: tl.constexpr, IDX: tl.constexpr, PART: tl.constexpr = 0):
    # Outputs OS = Gam o Rk + Rd WS, OV = Rd WV (Rd = D' o Rn); completes dWS, dWV.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(
        G, Beta, chunk, bh, T, H, M, C, BM, SLOT, IDX)
    idx = bh*NC+chunk
    dOS = _ld(DOS, idx, C, M, C, BM)
    dOV = _ld(DOV, idx, C, C, C, C)
    Rn = _ld(RN, idx, C, C, C, C)
    Rd = Dp*Rn
    # C128 launches each part separately: avoid keeping both orientations of the
    # 128x128 dOV matrix in shared memory alongside Tm/WV and Rd.
    if PART != 2:
        Tm = _ld(TT, idx, C, C, C, C)
        WS = -tl.dot(Tm, beta[:, None]*Gam*_ld(HK, idx, C, M, C, BM), input_precision=ATT)
        WV = Tm*beta[None, :]
        dRd = tl.dot(dOS, tl.trans(WS), input_precision=ATT)+tl.dot(dOV, tl.trans(WV), input_precision=ATT)
        _st(CRN, idx, dRd*Rd/TAU, C, C, C, C)
        _st(DE, idx, tl.where(le, dRd*Rn, 0.), C, C, C, C)
    if PART != 1:
        _st(DWS, idx, _ld(DWSP, idx, C, M, C, BM)+tl.dot(tl.trans(Rd), dOS, input_precision=ATT), C, M, C, BM)
        _st(DWV, idx, _ld(DWVP, idx, C, C, C, C)+tl.dot(tl.trans(Rd), dOV, input_precision=ATT), C, C, C, C)
        Rk = _ld(RK, idx, C, M, C, BM)
        _st(CRK, idx, Gam*dOS*Rk/TAU, C, M, C, BM)
        _st(DGAM, idx, dOS*Rk, C, M, C, BM)


@tr.jit
def ut_bwd_kernel(G, Beta, HK, HN, TT, DWS, DWV, DGAM, DE, CHK, CHNS, DB,
                  T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                  BM: tl.constexpr, TAU: tl.constexpr, SLOT: tl.constexpr, IDX: tl.constexpr):
    # WS = -T (beta Gam Hk), WV = T diag(beta), T = (I + diag(beta)(D o Hn))^-1.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(
        G, Beta, chunk, bh, T, H, M, C, BM, SLOT, IDX)
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
                     BM: tl.constexpr, SLOT: tl.constexpr, IDX: tl.constexpr, REDUCE: tl.constexpr = False):
    # Log decay, scattered to ring slots: old atom u <-> slot (o+u) mod M, new atom s -> (o+s) mod M.
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(
        G, Beta, chunk, bh, T, H, M, C, BM, SLOT, IDX)
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
    u_of_r = tl.where(um, u if IDX else (u-o+M) % M, 0)
    Rnew = (u[None, :] == r_new[:, None]).to(tl.float32)
    dGc = tl.gather(_ld(DGAM, idx, C, M, C, BM)*Gam, tl.broadcast_to(u_of_r[None, :], (C, BM)), 1)
    if REDUCE and M == BM and C % BM == 0:
        # C is a multiple of M, so o = (chunk*C) % M = 0 and r_new[s] = s % M.
        # Y @ one_hot(r_new) sums exactly those columns; no dense FP32 GEMM is needed.
        yr = tl.sum(tl.reshape(Y, (C, C//BM, BM)), 1)
    else:
        yr = tl.dot(Y, Rnew, input_precision='ieee')
    dGc += yr-(tl.sum(Y, 0)+Z)[:, None]*Rnew
    rowC = tl.sum(Z[:, None]*Rnew, 0)+tl.gather(W, u_of_r, 0)
    dGc += tl.where(t[:, None] == n-1, rowC[None, :], 0.)
    dGc = tl.where(valid[:, None] & um[None, :], dGc, 0.)
    dg = tl.sum(dGc, 0)[None, :]-tl.cumsum(dGc, 0)+dGc
    tl.store(DG+tok[:, None]*M+u[None, :], dg, valid[:, None] & um[None, :])


@tr.jit
def proj_bwd_kernel(G, Beta, CAND, HK, TT, PEO, PEN, PINV, KC, NCM, GK, DFS, DFV, DWSP, DWVP, VEC, CKC, CNC, CGS,
                    T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                    BM: tl.constexpr, TAU: tl.constexpr, SLOT: tl.constexpr, IDX: tl.constexpr):
    """Fused proj_bwd_a + proj_bwd_b: next-state maps, then through the Gram inverse."""
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(
        G, Beta, chunk, bh, T, H, M, C, BM, SLOT, IDX)
    idx = bh*NC+chunk
    a = tl.arange(0, BM)
    Tm = _ld(TT, idx, C, C, C, C)
    WS = -tl.dot(Tm, beta[:, None]*Gam*_ld(HK, idx, C, M, C, BM), input_precision=ATT)  # as in compose_kernel
    WV = Tm*beta[None, :]
    selo, seln, mo, mn = _sel(CAND, bh, chunk, n, M, C, BM, NC, IDX)
    PEn = _ld(PEN, idx, M, C, BM, C)
    Jn = seln+PEn
    dFS = _ld(DFS, idx, M, M, BM, BM)
    dFV = _ld(DFV, idx, M, C, BM, C)
    PEo = _ld(PEO, idx, M, M, BM, BM)
    dgend = tl.sum(dFS*(selo+PEo), 0)
    dWSd = tl.dot(tl.trans(Jn), dFS, input_precision=PROJ)
    dWVd = tl.dot(tl.trans(Jn), dFV, input_precision=PROJ)
    ddend = tl.sum(dWSd*WS, 1)+tl.sum(dWVd*WV, 1)
    _st(DWSP, idx, dend[:, None]*dWSd, C, M, C, BM)
    _st(DWVP, idx, dend[:, None]*dWVd, C, C, C, C)
    tl.store(VEC+idx*(C+BM)+t, ddend*dend)
    tl.store(VEC+idx*(C+BM)+C+a, dgend*gend)
    dJn = tl.dot(dFS, tl.trans(dend[:, None]*WS), input_precision=PROJ) + \
        tl.dot(dFV, tl.trans(dend[:, None]*WV), input_precision=PROJ)
    Pinv = _ld(PINV, idx, M, M, BM, BM)
    Zo = tl.dot(Pinv, tl.where(mo[None, :], dFS*gend[None, :], 0.), input_precision=PROJ)
    Zn = tl.dot(Pinv, tl.where(mn[None, :], dJn, 0.), input_precision=PROJ)
    _st(CKC, idx, tl.where(mo[None, :], Zo, 0.)*_ld(KC, idx, M, M, BM, BM)/TAU, M, M, BM, BM)
    _st(CNC, idx, tl.where(mn[None, :], Zn, 0.)*_ld(NCM, idx, M, C, BM, C)/TAU, M, C, BM, C)
    cG = -(tl.dot(Zo, tl.trans(PEo), input_precision=PROJ) +
           tl.dot(Zn, tl.trans(PEn), input_precision=PROJ))*_ld(GK, idx, M, M, BM, BM)/TAU
    _st(CGS, idx, cG+tl.trans(cG), M, M, BM, BM)


@tr.jit
def chunk_bwd_kernel(G, Beta, HK, HN, RK, RN, TT, DOS, DOV, DWSP, DWVP, VEC, CHK, CHNS, CRK, CRN, DG, DB,
                     T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, NC: tl.constexpr,
                     BM: tl.constexpr, TAU: tl.constexpr, SLOT: tl.constexpr, IDX: tl.constexpr, BF: tl.constexpr, GCP=None, CACHE: tl.constexpr = 0):
    """Fused out_bwd + ut_bwd + decay_bwd: outputs, UT transform and log-decay gradients of one chunk."""
    chunk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    t, n, valid, tok, beta, Gam, Ex, Dl, Dp, dend, gend, le, lt, o, r_new = _slot_decay(
        G, Beta, chunk, bh, T, H, M, C, BM, SLOT, IDX, GCP, bh*NC+chunk, CACHE)
    idx = bh*NC+chunk
    u = tl.arange(0, BM)
    um = u < M
    Tm = _ld(TT, idx, C, C, C, C)
    Hk = _ld(HK, idx, C, M, C, BM)
    Bk = beta[:, None]*Gam*Hk
    WS = -tl.dot(Tm, Bk, input_precision=ATT)
    WV = Tm*beta[None, :]
    dOS = _ld(DOS, idx, C, M, C, BM)
    dOV = _ld(DOV, idx, C, C, C, C)
    dRd = _adot(dOS, tl.trans(WS), BF)+_adot(dOV, tl.trans(WV), BF)
    Rn = _ld(RN, idx, C, C, C, C)
    # Dp, Dl are re-formed at use with the condition negated and the branches swapped: the same values,
    # but not CSE-merged with the copies made at the top, which would keep two extra tiles live.
    Rd = tl.where(t[:, None] < t[None, :], 0., Ex)*Rn
    _st(CRN, idx, dRd*Rd/TAU, C, C, C, C)
    dE = tl.where(le, dRd*Rn, 0.)
    dWS = _ld(DWSP, idx, C, M, C, BM)+_adot(tl.trans(Rd), dOS, BF)
    dWV = _ld(DWVP, idx, C, C, C, C)+_adot(tl.trans(Rd), dOV, BF)
    Rk = _ld(RK, idx, C, M, C, BM)
    _st(CRK, idx, Gam*dOS*Rk/TAU, C, M, C, BM)
    dGam = dOS*Rk
    dB = -tl.dot(tl.trans(Tm), dWS, input_precision=ATT)
    _st(CHK, idx, dB*beta[:, None]*Gam*Hk/TAU, C, M, C, BM)
    dGam += dB*beta[:, None]*Hk
    dbeta = tl.sum(dB*Gam*Hk, 1)+tl.sum(dWV*Tm, 0)
    dT = -tl.dot(dWS, tl.trans(Bk), input_precision=ATT)+dWV*beta[None, :]
    dL = -tl.where(lt, tl.dot(tl.dot(tl.trans(Tm), dT, input_precision=ATT), tl.trans(Tm), input_precision=ATT), 0.)
    Hn = _ld(HN, idx, C, C, C, C)
    dbeta += tl.sum(dL*tl.where(t[:, None] <= t[None, :], 0., Ex)*Hn, 1)
    cHn = beta[:, None]*dL*tl.where(t[:, None] <= t[None, :], 0., Ex)*Hn/TAU
    _st(CHNS, idx, cHn+tl.trans(cHn), C, C, C, C)
    dE += tl.where(lt, beta[:, None]*dL*Hn, 0.)
    tl.store(DB+tok, dbeta, valid)
    Y = dE*Ex
    vec = VEC+idx*(C+BM)
    Z = tl.load(vec+t)
    W = tl.load(vec+C+u, um, 0)
    if not SLOT:
        dgc = tl.sum(dGam*Gam, 1)+tl.sum(Y, 1)-tl.sum(Y, 0)-Z
        dgc += tl.where(t == n-1, tl.sum(Z, 0)+tl.sum(W, 0), 0.)
        dgc = tl.where(valid, dgc, 0.)
        tl.store(DG+tok, tl.sum(dgc, 0)-tl.cumsum(dgc, 0)+dgc, valid)
    elif (M == BM) and (C % BM == 0):
        # Rotated slot columns (see _slot_decay): a new atom s adds to column s mod M. FIFO old atoms are
        # in the same rotation and the result is stored through it; IDX old atoms are by slot, so the
        # new-atom part is rotated back with one gather.
        Rrot = (t[:, None] % BM) == u[None, :]
        if C == BM:
            dGr = Y
        else:
            dGr = tl.sum(tl.reshape(Y, (C, C//BM, BM)), 1)
        dGr -= tl.where(Rrot, (tl.sum(Y, 0)+Z)[:, None], 0.)
        dGr += tl.where(t[:, None] == n-1, tl.sum(tl.where(Rrot, Z[:, None], 0.), 0)[None, :], 0.)
        dGc = dGam*Gam+tl.where(t[:, None] == n-1, W[None, :], 0.)
        if IDX:
            dGc += tl.gather(dGr, tl.broadcast_to(((u-o+M) % M)[None, :], (C, BM)), 1)
        else:
            dGc += dGr
        dGc = tl.where(valid[:, None] & um[None, :], dGc, 0.)
        dg = tl.sum(dGc, 0)[None, :]-tl.cumsum(dGc, 0)+dGc
        if IDX:
            tl.store(DG+tok[:, None]*M+u[None, :], dg, valid[:, None] & um[None, :])
        else:
            tl.store(DG+tok[:, None]*M+(o+u[None, :]) % M, dg, valid[:, None] & um[None, :])
    else:
        u_of_r = tl.where(um, u if IDX else (u-o+M) % M, 0)
        Rnew = (u[None, :] == r_new[:, None]).to(tl.float32)
        dGc = tl.gather(dGam*Gam, tl.broadcast_to(u_of_r[None, :], (C, BM)), 1)
        dGc += tl.dot(Y, Rnew, input_precision='ieee')-(tl.sum(Y, 0)+Z)[:, None]*Rnew
        rowC = tl.sum(Z[:, None]*Rnew, 0)+tl.gather(W, u_of_r, 0)
        dGc += tl.where(t[:, None] == n-1, rowC[None, :], 0.)
        dGc = tl.where(valid[:, None] & um[None, :], dGc, 0.)
        dg = tl.sum(dGc, 0)[None, :]-tl.cumsum(dGc, 0)+dGc
        tl.store(DG+tok[:, None]*M+u[None, :], dg, valid[:, None] & um[None, :])


@tr.jit
def _shift(P, idx, R: tl.constexpr, CC: tl.constexpr, rows, cols, ok, TRANS: tl.constexpr):
    # Tile x[i, j] = F[rows_i, cols_j] (F[cols_j, rows_i] if TRANS) of the R x CC factor F at idx,
    # zero where ok is false or the index falls outside F.
    if TRANS:
        r, c = cols[None, :], rows[:, None]
    else:
        r, c = rows[:, None], cols[None, :]
    m = ok & (r >= 0) & (r < R) & (c >= 0) & (c < CC)
    return tl.load(P+idx*R*CC+r*CC+c, m, 0).to(tl.float32)


@tr.jit
def _keygrad_merged(Kold, Kn, Q, CHK, CHN, CRK, CRN, CKC, CNC, CG, DU, idx, n, d, dmask,
                    M: tl.constexpr, C: tl.constexpr, BM: tl.constexpr, BU: tl.constexpr, D: tl.constexpr,
                    BF: tl.constexpr, SHARE: tl.constexpr = False):
    # FIFO with n >= M: the kept atoms are the last M new keys (new-key index t = s+a, s = n-M), so every
    # product with Knew is a product with Kn, and the kept-atom gradient lands on new-key rows. The
    # projection-side factors are shifted into the attention-side ones at load time:
    #   new keys  F_old = CHK + CKC[t-s, :]                                   (x Kold)
    #             F_new = CHN + CNC[t-s, u] + CNC[u-s, t] + CG[t-s, u-s]       (x Kn)
    #             + CRN^T Q
    #   old atoms F_k = CHK^T + CKC[t-s, a]^T                                 (x Kn)   + CRK^T Q
    # Mixed factors run at the projection precision; pure attention-side products keep _adot.
    t = tl.arange(0, C)
    a = tl.arange(0, BM)
    s = n-M
    tv = t < n
    am = a < M
    full = tv[:, None] & tv[None, :]
    f_old = (_shift(CHK, idx, C, M, t, a, tv[:, None] & am[None, :], False)
             + _shift(CKC, idx, M, M, t-s, a, tv[:, None] & am[None, :], False))
    f_new = _shift(CHN, idx, C, C, t, t, full, False)+_shift(CG, idx, M, M, t-s, t-s, full, False)
    if not SHARE:   # (SHARE: CNC = 0)
        y = _shift(CNC, idx, M, C, t-s, t, full, False)
        f_new += y+tl.trans(y)
    dkn = (tl.dot(f_old, Kold, input_precision=PROJ) + tl.dot(f_new, Kn, input_precision=PROJ)
           + _adot(tl.trans(_ld(CRN, idx, C, C, C, C)), Q, BF))
    # F_k is F_old transposed (same CHK and shifted CKC entries).
    f_k = tl.trans(f_old)
    dko = tl.dot(f_k, Kn, input_precision=PROJ) + _adot(tl.trans(_ld(CRK, idx, C, M, C, BM)), Q, BF)
    tl.store(DU+(idx*BU+a[:, None])*D+d[None, :], dko, am[:, None] & dmask)
    tl.store(DU+(idx*BU+M+t[:, None])*D+d[None, :], dkn, tv[:, None] & dmask)


@tr.jit
def keygrad_dest_kernel(QN, Pool, CHK, CHN, CRK, CRN, CKC, CG, DQN, DPool, DF,
                        T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                        NC: tl.constexpr, BDK: tl.constexpr, BF: tl.constexpr):
    # SHARE (fifo, C == M) with full chunks only. Pool rows of chunk j's new keys get dkn(j) (with the
    # kept-atom part, as in _keygrad_merged) and dko(j+1): chunk j+1 evicts exactly these atoms. Program
    # j = -1 covers the initial keys (dko(0) only); the last chunk's rows are the final keys (+ dF).
    # One program per destination block: no union-row blocks and no second summation pass.
    d0, jj, bh = tl.program_id(0)*BDK, tl.program_id(1).to(tl.int64)-1, tl.program_id(2).to(tl.int64)
    b, head = bh//H, bh % H
    t = tl.arange(0, C)
    d = d0+tl.arange(0, BDK)
    dm = d[None, :] < D
    out = tl.zeros((C, BDK), tl.float32)
    if jj >= 0:
        idx = bh*NC+jj
        rbase = bh*(T+M)+jj*C
        Kold = _rows(Pool, rbase, M, C, D, d0, BDK)
        Kn = _rows(Pool, rbase+M, C, C, D, d0, BDK)
        tok = (b*T+jj*C+t)*H+head
        Q = tl.load(QN+tok[:, None]*D+d[None, :], dm, 0)
        dq = _adot(_ld(CRK, idx, C, M, C, C), Kold, BF)+_adot(_ld(CRN, idx, C, C, C, C), Kn, BF)
        tl.store(DQN+tok[:, None]*D+d[None, :], dq, dm)
        f_old = _ld(CHK, idx, C, M, C, C)+_ld(CKC, idx, M, M, C, C)
        f_new = _ld(CHN, idx, C, C, C, C)+_ld(CG, idx, M, M, C, C)
        out = (tl.dot(f_old, Kold, input_precision=PROJ)+tl.dot(f_new, Kn, input_precision=PROJ)
               + _adot(tl.trans(_ld(CRN, idx, C, C, C, C)), Q, BF))
    if jj+1 < NC:
        idx1 = bh*NC+jj+1
        rbase1 = bh*(T+M)+(jj+1)*C
        Kn1 = _rows(Pool, rbase1+M, C, C, D, d0, BDK)
        tok1 = (b*T+(jj+1)*C+t)*H+head
        Q1 = tl.load(QN+tok1[:, None]*D+d[None, :], dm, 0)
        f_k = tl.trans(_ld(CHK, idx1, C, M, C, C)+_ld(CKC, idx1, M, M, C, C))
        out += tl.dot(f_k, Kn1, input_precision=PROJ)+_adot(tl.trans(_ld(CRK, idx1, C, M, C, C)), Q1, BF)
    else:
        out += tl.load(DF+(bh*M+t[:, None])*D+d[None, :], dm, 0)
    tl.store(DPool+(bh*(T+M)+M+jj*C+t[:, None])*D+d[None, :], out, dm)


@tr.jit
def keygrad_kernel(QN, Pool, OLD, CHK, CHN, CRK, CRN, CKC, CNC, CG, DQN, DU, DPool,
                   T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                   NC: tl.constexpr, BM: tl.constexpr, BDK: tl.constexpr, BU: tl.constexpr, IDX: tl.constexpr,
                   BF: tl.constexpr = False, MERGE: tl.constexpr = False, SHARE: tl.constexpr = False):
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
    Kold = _dict(Pool, OLD, bh, chunk, 0, rbase, n, d0, T, M, BM, D, BDK, NC, IDX)
    Kn = _rows(Pool, rbase+M, n, C, D, d0, BDK)
    Q = tl.load(QN+tok[:, None]*D+d[None, :], valid[:, None] & dmask, 0)
    dq = (_adot(_ld(CRK, idx, C, M, C, BM), Kold, BF)
          + _adot(_ld(CRN, idx, C, C, C, C), Kn, BF))
    tl.store(DQN+tok[:, None]*D+d[None, :], dq, valid[:, None] & dmask)
    merged = n < 0
    if MERGE:
        merged = n >= M
    if merged:
        _keygrad_merged(Kold, Kn, Q, CHK, CHN, CRK, CRN, CKC, CNC, CG, DU, idx, n, d, dmask, M, C, BM, BU, D, BF, SHARE)
    else:
        Knew = _dict(Pool, OLD, bh, chunk, 1, rbase, n, d0, T, M, BM, D, BDK, NC, IDX)
        dkn = (_adot(_ld(CHK, idx, C, M, C, BM), Kold, BF)
               + _adot(_ld(CHN, idx, C, C, C, C), Kn, BF)
               + _adot(tl.trans(_ld(CRN, idx, C, C, C, C)), Q, BF)
               + tl.dot(tl.trans(_ld(CNC, idx, M, C, BM, C)), Knew, input_precision=PROJ))
        dko = (_adot(tl.trans(_ld(CHK, idx, C, M, C, BM)), Kn, BF)
               + _adot(tl.trans(_ld(CRK, idx, C, M, C, BM)), Q, BF)
               + tl.dot(tl.trans(_ld(CKC, idx, M, M, BM, BM)), Knew, input_precision=PROJ))
        dkk = (tl.dot(_ld(CKC, idx, M, M, BM, BM), Kold, input_precision=PROJ)
               + tl.dot(_ld(CNC, idx, M, C, BM, C), Kn, input_precision=PROJ)
               + tl.dot(_ld(CG, idx, M, M, BM, BM), Knew, input_precision=PROJ))
        if IDX:
            # Atoms live for data-dependent spans: accumulate into pool rows atomically.
            old = tl.load(OLD+(bh*(NC+1)+chunk)*M+a, am, 0)
            kept = tl.load(OLD+(bh*(NC+1)+chunk+1)*M+a, am, 0)
            prow = bh*(T+M)
            tl.atomic_add(DPool+(rbase+M+t[:, None])*D+d[None, :], dkn, valid[:, None] & dmask, sem='relaxed')
            tl.atomic_add(DPool+(prow+old[:, None])*D+d[None, :], dko, am[:, None] & dmask, sem='relaxed')
            tl.atomic_add(DPool+(prow+kept[:, None])*D+d[None, :], dkk, am[:, None] & dmask, sem='relaxed')
        else:
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
def pool_grad_kernel(DU, DPool, DF, T: tl.constexpr, M: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
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
    fin = (r >= T) & rmask
    acc += tl.load(DF+(bh*M+r[:, None]-T)*D+d[None, :], fin[:, None] & (d[None, :] < D), 0)
    tl.store(DPool+(bh*(T+M)+r[:, None])*D+d[None, :], acc, rmask[:, None] & (d[None, :] < D))


@tr.jit
def normalize_rows_kernel(Q, K, Initial, QN, Pool, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
                          D: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
    # geometry.normalize_kernel by row blocks: pool rows blk*BR.. (initial keys at 0..M, token t at M+t),
    # and query rows (tokens) blk*BR.. .
    blk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    b, head = bh//H, bh % H
    r = blk*BR+tl.arange(0, BR)
    d = tl.arange(0, BD)
    dm = d[None, :] < D
    tokm = (r >= M) & (r < T+M)
    ini = r < M
    tk = (b*T+r-M)*H+head
    k = tl.load(K+tk[:, None]*D+d[None, :], tokm[:, None] & dm, 0).to(tl.float32)
    k += tl.load(Initial+(bh*M+r[:, None])*D+d[None, :], ini[:, None] & dm, 0).to(tl.float32)
    k = k/tl.maximum(tl.sqrt(tl.sum(k*k, 1)), 1.e-6)[:, None]
    tl.store(Pool+(bh*(T+M)+r[:, None])*D+d[None, :], k, (r < T+M)[:, None] & dm)
    qm = r < T
    tq = (b*T+r)*H+head
    q = tl.load(Q+tq[:, None]*D+d[None, :], qm[:, None] & dm, 0).to(tl.float32)
    q = q/tl.maximum(tl.sqrt(tl.sum(q*q, 1)), 1.e-6)[:, None]
    tl.store(QN+tq[:, None]*D+d[None, :], q, qm[:, None] & dm)


@tr.jit
def _norm_grad_rows(raw, g):
    # Row-wise normalize backward (same as geometry._norm_grad for a [rows, D] tile).
    norm = tl.sqrt(tl.sum(raw*raw, 1))
    denom = tl.maximum(norm, 1.e-6)
    nrm = raw/denom[:, None]
    return (g-tl.where((norm >= 1.e-6)[:, None], nrm*tl.sum(nrm*g, 1)[:, None], 0.))/denom[:, None]


@tr.jit
def pool_norm_bwd_kernel(DU, DF, Q, K, Initial, DQN, DQ, DK, DI, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
                         D: tl.constexpr, C: tl.constexpr, NC: tl.constexpr, BU: tl.constexpr, BR: tl.constexpr,
                         BD: tl.constexpr, POOL: tl.constexpr = False):
    # pool_grad_kernel followed by the normalization backward of the same rows (no DPool round trip);
    # the same block index also finishes dq for tokens blk*BR .. blk*BR+BR. POOL: DU is the summed
    # pool gradient itself (indexed path: atomics, final rows already added).
    blk, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    b, head = bh//H, bh % H
    r = blk*BR+tl.arange(0, BR)
    d = tl.arange(0, BD)
    dm = d[None, :] < D
    rmask = r < T+M
    if POOL:
        acc = tl.load(DU+(bh*(T+M)+r[:, None])*D+d[None, :], rmask[:, None] & dm, 0)
    else:
        acc = tl.zeros((BR, BD), tl.float32)
        lo = tl.maximum(blk*BR-M-C+1, 0)//C
        hi = tl.minimum((blk*BR+BR-1)//C, NC-1)
        for c in range(lo, hi+1):
            n = tl.minimum(C, T-c*C)
            rel = r-c*C
            ok = (rel >= 0) & (rel < M+n) & rmask
            acc += tl.load(DU+((bh*NC+c)*BU+rel[:, None])*D+d[None, :], ok[:, None] & dm, 0)
        fin = (r >= T) & rmask
        acc += tl.load(DF+(bh*M+r[:, None]-T)*D+d[None, :], fin[:, None] & dm, 0)
    tokm = (r >= M) & rmask
    ini = r < M
    tk = (b*T+r-M)*H+head
    raw = tl.load(K+tk[:, None]*D+d[None, :], tokm[:, None] & dm, 0).to(tl.float32)
    raw += tl.load(Initial+(bh*M+r[:, None])*D+d[None, :], ini[:, None] & dm, 0).to(tl.float32)
    gk = _norm_grad_rows(raw, acc)
    tl.store(DK+tk[:, None]*D+d[None, :], gk, tokm[:, None] & dm)
    tl.store(DI+(bh*M+r[:, None])*D+d[None, :], gk, ini[:, None] & dm)
    qm = r < T
    tq = (b*T+r)*H+head
    rq = tl.load(Q+tq[:, None]*D+d[None, :], qm[:, None] & dm, 0).to(tl.float32)
    gq = tl.load(DQN+tq[:, None]*D+d[None, :], qm[:, None] & dm, 0)
    tl.store(DQ+tq[:, None]*D+d[None, :], _norm_grad_rows(rq, gq), qm[:, None] & dm)


@tr.jit
def dict_scan_kernel(Scores, Init, OLD, CAND, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
                     NC: tl.constexpr, BM: tl.constexpr, BU: tl.constexpr):
    """Learned-score eviction, value independent: per (b,h) serial over chunks, keep the M highest
    (score, newer) of M old slots + n new atoms; kept new atoms take their positional slot (t mod M)
    when freed, else the remaining freed slots in order. Writes OLD[c, slot] = pool row, CAND[c, slot]."""
    bh = tl.program_id(0).to(tl.int64)
    b, head = bh//H, bh % H
    r = tl.arange(0, BM)
    rm = r < M
    i = tl.arange(0, BU)
    row = tl.where(rm, r, 0)
    score = tl.load(Init+bh*M+r, rm, 0).to(tl.float32)
    born = (r-M).to(tl.float32)
    isold = i < M
    io = tl.where(isold, i, 0)
    s_i = i-M
    # The serial loop is latency bound: the next chunk's scores are loaded one iteration ahead.
    nxt = tl.load(Scores+(b*T+s_i)*H+head, (i >= M) & (s_i < tl.minimum(C, T)), 0).to(tl.float32)
    for c in range(NC):
        tl.store(OLD+(bh*(NC+1)+c)*M+r, row, rm)
        t0 = c*C
        n = tl.minimum(C, T-t0)
        isnew = (i >= M) & (s_i < n)
        valid = isold | isnew
        new_scores = nxt
        nxt = tl.load(Scores+(b*T+t0+C+s_i)*H+head, (i >= M) & (s_i < tl.minimum(C, T-t0-C)), 0).to(tl.float32)
        cs = tl.where(isold, tl.gather(score, io, 0), new_scores)
        cb = tl.where(isold, tl.gather(born, io, 0), (t0+s_i).to(tl.float32))
        crow = tl.where(isold, tl.gather(row, io, 0), (M+t0+s_i).to(tl.int32))
        beats = ((cs[None, :] > cs[:, None]) | ((cs[None, :] == cs[:, None]) & (cb[None, :] > cb[:, None]))) & valid[None, :]
        keep = valid & (tl.sum(beats.to(tl.int32), 1) < M)
        free = rm & (tl.gather((keep & isold).to(tl.int32), r, 0) == 0)
        pos = tl.where(isnew, (t0+s_i) % M, 0)
        knew = keep & isnew
        claim = knew & (tl.gather(free.to(tl.int32), pos, 0) > 0)
        if C > M:
            earlier = (s_i[None, :] < s_i[:, None]) & knew[None, :] & (pos[None, :] == pos[:, None])
            claim = claim & (tl.sum(earlier.to(tl.int32), 1) == 0)
        claimed = tl.sum((claim[None, :] & (pos[None, :] == r[:, None])).to(tl.int32), 1) > 0
        rest_slot = free & ~claimed
        rest_temp = knew & ~claim
        slot_rank = tl.cumsum(rest_slot.to(tl.int32), 0)-1
        temp_rank = tl.cumsum(rest_temp.to(tl.int32), 0)-1
        fill = tl.sum(tl.where(rest_slot[None, :] & (slot_rank[None, :] == temp_rank[:, None]), r[None, :], 0), 1)
        slot = tl.where(isold, i, tl.where(claim, pos, fill))
        cand = tl.sum(tl.where(keep[None, :] & (slot[None, :] == r[:, None]), i[None, :], 0), 1)
        tl.store(CAND+(bh*NC+c)*M+r, cand, rm)
        cidx = tl.where(rm, cand, 0)
        row = tl.gather(crow, cidx, 0)
        score = tl.gather(cs, cidx, 0)
        born = tl.gather(cb, cidx, 0)
    tl.store(OLD+(bh*(NC+1)+NC)*M+r, row, rm)


@tr.jit
def final_idx_kernel(Pool, OLD, Final, DPool, DF, T: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                     NC: tl.constexpr, BD: tl.constexpr, BACKWARD: tl.constexpr):
    bh, m = tl.program_id(0).to(tl.int64), tl.program_id(1)
    d = tl.arange(0, BD)
    row = (bh*(T+M)+tl.load(OLD+(bh*(NC+1)+NC)*M+m))*D+d
    if BACKWARD:
        tl.atomic_add(DPool+row, tl.load(DF+(bh*M+m)*D+d, d < D, 0), d < D, sem='relaxed')
    else:
        tl.store(Final+(bh*M+m)*D+d, tl.load(Pool+row, d < D, 0), d < D)


def _fused_scan(F, FO, value, init, tape, final, os, ov, out, t, h, m, width, c, nc, bm, bv, rev, bf):
    """Three-kernel affine chunk scan with in-place offsets; the forward pass also writes outputs."""
    bh = init.shape[0]*init.shape[1]
    nv = tr.cdiv(width, bv)
    if SCAN_SERIAL and bh*nv >= SCAN_SERIAL:
        # Enough (batch-head, value-tile) programs to fill the GPU: one serial pass over the chunks. The
        # per-chunk loads are pipelined; each stage buffers F, FO (FP32) and the value/OS/OV tiles, so the
        # stage count is capped by shared memory (FP32 inputs at M=64 fit two stages, BF16 three).
        es = value.element_size()
        per_stage = 4*bm*bm+4*bm*c+es*(c*bv)+os.element_size()*(c*bm+c*c)
        # C128 output tiles and FP32 dot scratch exceed H100 shared memory with pipelined stages.
        stages = 1 if c == 128 else max(1, min(SCAN_STAGES, 200_000//per_stage))
        bscan_expand_kernel[(1, bh, nv)](F, FO, value, init, tape, os, ov, out, t, h, m, width, c, nc, nc, 1, bm, bv,
                                         rev, not rev, SCAN, bf, final, True, num_warps=WARPS['scan'],
                                         num_stages=stages)
        return
    group = max(1, min(nc, 1 << max(0, round(math.log2(math.sqrt(nc/2)))) if nc > 1 else 1))
    if SCAN_GROUP:
        group = max(1, min(nc, SCAN_GROUP))
    ng = tr.cdiv(nc, group)
    nv = tr.cdiv(width, bv)
    ga = torch.empty((bh, ng, m, m), device=tape.device, dtype=torch.float32)
    gc = torch.empty((bh, ng, m, width), device=tape.device, dtype=torch.float32)
    entry = torch.empty_like(gc)
    w = WARPS['scan']
    bscan_compose_kernel[(ng, bh, nv+1)](F, FO, value, ga, gc, t, h, m, width, c, nc, group, ng,
                                         bm, bv, rev, nv, SCAN, num_warps=w, num_stages=1 if c == 128 else 3)
    group_scan_kernel[(bh, nv)](ga, gc, init, entry, final, m, width, ng, bm, bv, SCAN, num_warps=w)
    bscan_expand_kernel[(ng, bh, nv)](F, FO, value, entry, tape, os, ov, out, t, h, m, width, c, nc, group, ng, bm, bv,
                                      rev, not rev, SCAN, bf, num_warps=w, num_stages=1 if c == 128 else 3)


class _BlockFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, keys0, values0, scores, init_scores, c, tau, ridge):
        b, t, h, d = q.shape
        m, width = keys0.shape[-2], v.shape[-1]
        bm, bd = max(16, tr.next_power_of_2(m)), max(16, tr.next_power_of_2(d))
        bdk = min(bd, 64)
        nc = tr.cdiv(t, c)
        alloc = lambda *s: torch.empty(s, device=q.device, dtype=torch.float32)  # noqa: E731
        mats = lambda *shape: alloc(b*h*nc, *shape)  # noqa: E731
        # Attention-side matrices follow the input precision (BF16 in, BF16 stored); everything on the
        # projection side (Gram, its inverse, projections, their cosine gradients) and the state maps stay FP32.
        sdt = q.dtype if q.dtype in (torch.bfloat16, torch.float16) else torch.float32
        low = lambda *shape: torch.empty((b*h*nc, *shape), device=q.device, dtype=sdt)  # noqa: E731
        # Normalized queries are only read by attention-side products, which run in BF16 for BF16 inputs.
        qn = torch.empty((b, t, h, d), device=q.device,
                         dtype=torch.bfloat16 if (QN_BF16 and BF16 and q.dtype == torch.bfloat16) else torch.float32)
        pool = alloc(b, h, t+m, d)
        if ROW_NORMALIZE:
            br = 32 if bd > 64 else 64
            normalize_rows_kernel[(tr.cdiv(t+m, br), b*h)](q, k, keys0, qn, pool, t, h, m, d, br, bd, num_warps=4)
        else:
            normalize_kernel[(t+m, b*h)](q, k, keys0, qn, pool, t, h, m, d, bd)
        hk, hn, rk, rn = low(c, m), low(c, c), low(c, m), low(c, c)
        kc, ncm, gk, pinv, peo, pen = mats(m, m), mats(m, c), mats(m, m), mats(m, m), mats(m, m), mats(m, c)
        aa, tt = mats(c, c), low(c, c)
        os, ov, fs, fv = low(c, m), low(c, c), mats(m, m), mats(m, c)
        slot = g.ndim == 4
        ix = scores is not None
        old = torch.empty((b*h, nc+1, m) if ix else (1,), device=q.device, dtype=torch.int32)
        cand = torch.empty((b*h, nc, m) if ix else (1,), device=q.device, dtype=torch.int32)
        grid = (nc, b*h)
        bf = BF16 and q.dtype == torch.bfloat16
        overlap = ix and OVERLAP_DICT and not FUSE_CHUNK_FWD and q.is_cuda
        if overlap:
            # dict_scan is one serial program per batch-head; the dictionary-free maps and the UT solve
            # run beside it (and beside proj) on a side stream, joined before compose.
            cur, side = torch.cuda.current_stream(q.device), _side_stream(q.device)
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                att_kernel[grid](qn, pool, old, g, beta, hk, hn, rk, rn, t, h, m, d, c, nc, bm, bdk, tau, ix, bf, False,
                                 1, num_warps=WARPS['att'], num_stages=1)
                ut_kernel[grid](g, beta, hn, aa, tt, t, h, m, c, nc, bm, c//16, slot, ix, None, 0,
                                num_warps=WARPS['ut_slot' if slot else 'ut'])
        if ix:
            dict_scan_kernel[(b*h,)](scores, init_scores, old, cand, t, h, m, c, nc, bm, tr.next_power_of_2(m+c),
                                     num_warps=WARPS['dict'])
        share = SHARE_GRAM and c == m and not ix and not FUSE_CHUNK_FWD and not FUSE_PROJ_BWD
        cache = (SLOT_GC_CACHE and slot and not ix and m == bm and c % bm == 0 and not FUSE_CHUNK_FWD
                 and not FUSE_PROJ_BWD and FUSE_CHUNK_BWD and c <= 64)
        gcache = torch.empty((b*h*nc, c, bm), device=q.device, dtype=torch.float32) if cache else None
        chol = CHOL_INV and not RANK1_SWEEP and not SWEEP_FUSED
        cs = torch.empty((b*h*nc, 2, bm, bm), device=q.device, dtype=torch.float32) if chol else None
        proj_kernel[grid](pool, old, cand, kc, ncm, gk, pinv, peo, pen, hk, hn, t, h, m, d, c, nc, bm, bdk, tau, ridge,
                          ix, RANK1_SWEEP, share, SWEEP_FUSED, share and SKIP_ZERO, cs, chol,
                          num_warps=WARPS['proj_ix' if ix else 'proj'], num_stages=1)
        if chol:
            chol_kernel[grid](cs, kc, ncm, pinv, peo, pen, t, m, c, nc, bm, bm//16, share, share and SKIP_ZERO,
                              num_warps=WARPS['chol'], num_stages=1)
            del cs
        out, final_values = torch.empty(v.shape, device=q.device, dtype=v.dtype), alloc(b, h, m, width)
        # The tape is read only by mapgrad, which multiplies it in BF16 for BF16 inputs: storing it in BF16
        # there is exact and halves its traffic.
        tape = torch.empty((b, h, nc, m, width), device=q.device,
                           dtype=torch.bfloat16 if (TAPE_BF16 and bf and FUSE_SCAN) else torch.float32)
        bv = SCAN_BV
        nv = tr.cdiv(width, bv)
        if FUSE_CHUNK_FWD:
            chunk_fwd_kernel[grid](qn, pool, old, cand, g, beta, hk, hn, rk, rn, aa, tt, peo, pen, os, ov, fs, fv,
                                   t, h, m, d, c, nc, bm, bdk, tau, c//16, slot, ix, bf, num_warps=WARPS['chunk_fwd'], num_stages=1)
        else:
            if overlap:
                att_kernel[grid](qn, pool, old, g, beta, hk, hn, rk, rn, t, h, m, d, c, nc, bm, bdk, tau, ix, bf, share,
                                 2, num_warps=WARPS['att'], num_stages=1)
                cur.wait_stream(side)
            else:
                att_kernel[grid](qn, pool, old, g, beta, hk, hn, rk, rn, t, h, m, d, c, nc, bm, bdk, tau, ix, bf, share,
                                 num_warps=WARPS['att'], num_stages=1)
                ut_kernel[grid](g, beta, hn, aa, tt, t, h, m, c, nc, bm, c//16, slot, ix, gcache, 1 if cache else 0,
                                num_warps=WARPS['ut_slot' if slot else 'ut'])
            compose_kernel[grid](g, beta, cand, hk, rk, rn, tt, peo, pen, os, ov, fs, fv, t, h, m, c, nc, bm, slot, ix,
                                 share, gcache, 2 if cache else 0, num_warps=WARPS['compose'])
        if FUSE_SCAN:
            _fused_scan(fs, fv, v, values0, tape, final_values, os, ov, out, t, h, m, width, c, nc, bm, bv, False, bf)
        else:
            off = alloc(b, h, nc, m, width)
            value_offset_kernel[(nc, b*h, nv)](fv, v, off, t, h, m, width, c, nc, bm, bv, SCAN, num_warps=WARPS['offset'])
            affine_scan(fs, off, values0, tape, final_values, False, None, bv, SCAN)
            output_kernel[(nc, b*h, nv)](os, ov, v, tape, out, t, h, m, width, c, nc, bm, bv, SCAN, num_warps=WARPS['output'])
        final_keys = alloc(b, h, m, d)
        if ix:
            final_idx_kernel[(b*h, m)](pool, old, final_keys, pool, pool, t, m, d, nc, bd, False)
        else:
            final_rows_kernel[(b*h, m)](pool, final_keys, pool, pool, t, m, d, bd, False)
        ctx.save_for_backward(q, k, v, g, beta, keys0, qn, pool, hk, hn, rk, rn, kc, ncm, gk, pinv, peo, pen, tt,
                              os, ov, fs, fv, tape, old, cand)
        ctx.ix = ix
        ctx.share = share
        ctx.gcache = gcache
        ctx.bf = bf
        ctx.cfg = (c, tau, values0.dtype)
        return out, final_keys, final_values

    @staticmethod
    @once_differentiable
    def backward(ctx, do, dfk, dfv):
        (q, k, v, g, beta, keys0, qn, pool, hk, hn, rk, rn, kc, ncm, gk, pinv, peo, pen, tt,
         os, ov, fs, fv, tape, old, cand) = ctx.saved_tensors
        ix = ctx.ix
        c, tau, values_dtype = ctx.cfg
        b, t, h, d = q.shape
        m, width = keys0.shape[-2], v.shape[-1]
        bm, bd = max(16, tr.next_power_of_2(m)), max(16, tr.next_power_of_2(d))
        nc = tr.cdiv(t, c)
        do, dfk, dfv = do.contiguous(), dfk.contiguous(), dfv.contiguous()
        ds = torch.empty_like(tape)
        bf = ctx.bf
        dvalues0 = torch.empty(dfv.shape, device=q.device, dtype=values_dtype)
        bv = SCAN_BV
        nv = tr.cdiv(width, bv)
        grid = (nc, b*h)
        slot = g.ndim == 4
        if FUSE_SCAN:
            _fused_scan(fs, os, do, dfv, ds, dvalues0, os, ov, do, t, h, m, width, c, nc, bm, bv, True, bf)
        else:
            off = torch.empty_like(tape)
            output_offset_kernel[(nc, b*h, nv)](os, do, off, t, h, m, width, c, nc, bm, bv, SCAN, num_warps=WARPS['offset'])
            affine_scan(fs, off, dfv, ds, dvalues0, True, None, bv, SCAN)

        def f32(x): return torch.empty(x.shape, device=x.device, dtype=torch.float32)
        dos, dov, dfs, dfv_map = f32(os), f32(ov), torch.empty_like(fs), torch.empty_like(fv)
        if DOV_BF16 and bf and FUSE_CHUNK_BWD and c <= 64:
            # chunk_bwd reads dOV only through BF16 products: storing it in BF16 is exact.
            dov = torch.empty(ov.shape, device=ov.device, dtype=torch.bfloat16)
        dv = torch.empty_like(v)
        block_mapgrad_kernel[grid](ov, fv, v, tape, ds, do, dos, dov, dfs, dfv_map, dv,
                                   t, h, m, width, c, nc, bm, MAPGRAD_BV, bf, num_warps=WARPS['mapgrad'])
        ckc, cnc, cgs = torch.empty_like(kc), torch.empty_like(ncm), torch.empty_like(gk)
        dwsp, dwvp, djn = f32(hk), f32(hn), f32(ncm)
        vec = torch.empty((b*h*nc, c+bm), device=q.device, dtype=torch.float32)
        chk, chns, crk, crn = torch.empty_like(hk), torch.empty_like(hn), torch.empty_like(rk), torch.empty_like(rn)
        dg, db = torch.empty_like(g), torch.empty_like(beta)
        if FUSE_PROJ_BWD:
            proj_bwd_kernel[grid](g, beta, cand, hk, tt, peo, pen, pinv, kc, ncm, gk, dfs, dfv_map, dwsp, dwvp, vec, ckc, cnc, cgs,
                                  t, h, m, c, nc, bm, tau, slot, ix, num_warps=WARPS['proj_bwd_fused'])
        else:
            proj_bwd_a_kernel[grid](g, beta, cand, hk, tt, peo, pen, dfs, dfv_map, dwsp, dwvp, djn, vec,
                                    t, h, m, c, nc, bm, slot, ix, ctx.share, ctx.gcache, 0 if ctx.gcache is None else 2,
                                    num_warps=WARPS['proj_bwd_ix' if ix else 'proj_bwd'])
            proj_bwd_b_kernel[grid](g, beta, cand, peo, pen, pinv, kc, ncm, gk, dfs, djn, ckc, cnc, cgs,
                                    t, h, m, c, nc, bm, tau, slot, ix, ctx.share, ctx.gcache, 0 if ctx.gcache is None else 2,
                                    num_warps=WARPS['proj_bwd_ix' if ix else 'proj_bwd'])
        # C128 uses three Triton backward stages to bound live 128x128 tiles.
        if FUSE_CHUNK_BWD and c <= 64:
            chunk_bwd_kernel[grid](g, beta, hk, hn, rk, rn, tt, dos, dov, dwsp, dwvp, vec, chk, chns, crk, crn, dg, db,
                                   t, h, m, c, nc, bm, tau, slot, ix, bf, ctx.gcache, 0 if ctx.gcache is None else 2,
                                   num_warps=WARPS['chunk_bwd_slot' if slot else 'chunk_bwd'])
        else:
            dws, dwv, dgam, de = f32(hk), f32(hn), f32(hk), f32(hn)
            for part in ((1, 2) if c == 128 else (0,)):
                out_bwd_kernel[grid](g, beta, hk, rk, rn, tt, dos, dov, dwsp, dwvp, dws, dwv, dgam, de, crk, crn,
                                     t, h, m, c, nc, bm, tau, slot, ix, part, num_warps=WARPS['ut_bwd'])
            ut_bwd_kernel[grid](g, beta, hk, hn, tt, dws, dwv, dgam, de, chk, chns, db,
                                t, h, m, c, nc, bm, tau, slot, ix, num_warps=WARPS['ut_bwd'])
            decay_bwd_kernel[grid](g, beta, dgam, de, vec, dg, t, h, m, c, nc, bm, slot, ix, REDUCE=DECAY_SLOT_REDUCE, num_warps=4)
        dqn = torch.empty(qn.shape, device=qn.device, dtype=torch.float32)
        kb = min(bd, KEY_BLOCK)
        bu = tr.next_power_of_2(m+c)
        dest = KEYGRAD_DEST and ctx.share and t % c == 0 and FUSE_POOL_NORM and MERGE_KEYGRAD
        if dest:
            dpool = torch.empty_like(pool)
            dq, dk, dk0 = torch.empty_like(q), torch.empty_like(k), torch.empty_like(keys0)
            keygrad_dest_kernel[(tr.cdiv(d, kb), nc+1, b*h)](qn, pool, chk, chns, crk, crn, ckc, cgs, dqn, dpool, dfk,
                                                             t, h, m, d, c, nc, kb, bf, num_warps=WARPS['keygrad'],
                                                             num_stages=1)
            br = 32 if bd > 64 else 64
            pool_norm_bwd_kernel[(tr.cdiv(t+m, br), b*h)](dpool, dfk, q, k, keys0, dqn, dq, dk, dk0, t, h, m, d, c,
                                                          nc, bu, br, bd, True, num_warps=4)
            return dq, dk, dv, dg, db, dk0, dvalues0, None, None, None, None, None
        du = torch.empty((b*h*nc, bu, d) if not ix else (1,), device=q.device, dtype=torch.float32)
        dpool = zero(torch.empty_like(pool)) if ix else (pool[:1, :1, :1] if FUSE_POOL_NORM else torch.empty_like(pool))
        keygrad_kernel[(tr.cdiv(d, kb), nc, b*h)](qn, pool, old, chk, chns, crk, crn, ckc, cnc, cgs, dqn, du, dpool,
                                                  t, h, m, d, c, nc, bm, kb, bu, ix, bf, MERGE_KEYGRAD and c >= m and not ix,
                                                  ctx.share,
                                                  num_warps=WARPS['keygrad'], num_stages=1)
        dq, dk, dk0 = torch.empty_like(q), torch.empty_like(k), torch.empty_like(keys0)
        if ix:
            final_idx_kernel[(b*h, m)](pool, old, pool, dpool, dfk, t, m, d, nc, bd, True)
            if FUSE_POOL_NORM:
                br = 32 if bd > 64 else 64
                pool_norm_bwd_kernel[(tr.cdiv(t+m, br), b*h)](dpool, dfk, q, k, keys0, dqn, dq, dk, dk0, t, h, m, d, c,
                                                              nc, bu, br, bd, True, num_warps=4)
                return dq, dk, dv, dg, db, dk0, dvalues0, None, None, None, None, None
        elif FUSE_POOL_NORM:
            br = 32 if bd > 64 else 64
            pool_norm_bwd_kernel[(tr.cdiv(t+m, br), b*h)](du, dfk, q, k, keys0, dqn, dq, dk, dk0, t, h, m, d, c, nc, bu,
                                                          br, bd, num_warps=4)
            return dq, dk, dv, dg, db, dk0, dvalues0, None, None, None, None, None
        else:
            pool_grad_kernel[(tr.cdiv(t+m, 64), b*h)](du, dpool, dfk, t, m, d, c, nc, bu, 64, bd, num_warps=4)
        normalize_bwd_kernel[(t+m, b*h)](q, k, keys0, dqn, dpool, dq, dk, dk0, t, h, m, d, bd)
        return dq, dk, dv, dg, db, dk0, dvalues0, None, None, None, None, None


def block_komp_fast(q, k, v, g, beta, initial_state, chunk_size=64, temperature=0.5, ridge=1e-4,
                    evict='fifo', scores=None, initial_scores=None):
    """Same contract as block_komp.block_komp; staged kernels.

    g is the log decay: [B,T,H] scalar per head, or [B,T,H,M] per physical slot. evict='fifo' keeps the
    newest M atoms (atom at sequence position p, initial atoms at -M..-1, lives in slot p mod M).
    evict='learned' keeps the M highest scores at each chunk end (ties to the newer atom; scores [B,T,H]
    for new atoms, initial_scores [B,H,M]; no gradient through the selection), matching
    block_flex.block_komp_flex. The final state is in age order for fifo (streamable as an initial
    state) and in physical slot order for learned (carry initial_scores alongside it).
    """
    if evict not in ('fifo', 'learned'):
        raise ValueError("block_komp_fast supports evict='fifo' or 'learned'; use block_flex for the others")
    if evict == 'learned':
        if scores is None or initial_scores is None:
            raise ValueError('learned eviction needs scores and initial_scores')
        scores = scores.detach().float().contiguous()
        initial_scores = initial_scores.detach().float().contiguous()
    else:
        scores = initial_scores = None
    keys0, values0 = initial_state
    if g.shape not in (q.shape[:3], (*q.shape[:3], keys0.shape[-2])):
        raise ValueError('g must have shape [B,T,H] or [B,T,H,M]')
    if chunk_size not in (16, 32, 64, 128) or keys0.shape[-2] > 64:
        raise ValueError('chunk_size must be 16, 32, 64 or 128 and at most 64 slots are supported')
    inputs = [x.contiguous() for x in (q, k, v, g, beta, keys0, values0)]
    with torch.cuda.device(q.device):
        out, keys, values = _BlockFast.apply(*inputs, scores, initial_scores, chunk_size, temperature, ridge)
    return out, (keys, values)

# C128 learned path: independently measured warp configurations.
WARPS.update(compose=8,ut_bwd=8,keygrad=8,proj_bwd_ix=8)
