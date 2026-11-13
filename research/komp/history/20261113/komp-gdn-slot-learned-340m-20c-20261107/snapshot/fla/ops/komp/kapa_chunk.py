# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Chunk-parallel soft-protected KAPA, with all-Triton numeric fwd/bwd.

Parallel key-only direction solves feed rank-one affine chunk maps. Two-level
scans carry boundary states/adjoints; each chunk replays at most C steps in bwd.
Scalar and per-slot log decay use division-free products, including exact resets.
"""
import math

import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable

from .kapa import add_grad, direction_bwd, direction_fwd
from .kapa_geometry import gaussian_features
from .recurrence import output_kernel
from .scan import affine_scan, output_offset_kernel, value_offset_kernel
from .triton_chunk import _inverse, _mul, reduce_kernel
from .utils import zero


@tr.jit
def scalar_prepare(Q, Hist, Write, G, Beta, OS, OV, FS, FV,
                   T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
                   BM: tl.constexpr, NC: tl.constexpr):
    block, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    c, m = tl.arange(0, C), tl.arange(0, BM)
    t = block*C+c
    tok = ((bh//H)*T+t)*H+bh % H
    valid = t < T
    mask = valid[:, None] & (m[None, :] < M)
    q = tl.load(Q+tok[:, None]*M+m[None, :], mask, 0).to(tl.float32)
    h = tl.load(Hist+tok[:, None]*M+m[None, :], mask, 0).to(tl.float32)
    w = tl.load(Write+tok[:, None]*M+m[None, :], mask, 0)
    beta = tl.load(Beta+tok, valid, 0).to(tl.float32)
    alpha = tl.exp(tl.load(G+tok, valid, 0).to(tl.float32))
    gt, ge = c[:, None] > c[None, :], c[:, None] >= c[None, :]
    windows = tl.associative_scan(tl.where(gt, alpha[:, None], 1.), 0, _mul)
    prefix = tl.associative_scan(alpha, 0, _mul)
    lower = tl.where(gt, -beta[:, None]*tl.dot(h, tl.trans(w), input_precision='tf32x3')*windows, 0.)
    inv = _inverse(lower, C)
    us = -tl.dot(inv, (beta*prefix)[:, None]*h, input_precision='tf32x3')
    uv = inv*beta[None, :]
    response = tl.where(ge, tl.dot(q, tl.trans(w), input_precision='tf32x3')*windows, 0.)
    os = prefix[:, None]*q+tl.dot(response, us, input_precision='tf32x3')
    ov = tl.where(ge, tl.dot(response, uv, input_precision='tf32x3'), 0.)
    end = tl.sum(tl.where(c[:, None] == C-1, windows, 0.), 0)
    endw = tl.trans(w*end[:, None])
    pend = tl.sum(tl.where(c == C-1, prefix, 0.), 0)
    fs = pend*(m[:, None] == m[None, :])+tl.dot(endw, us, input_precision='tf32x3')
    fv = tl.dot(endw, uv, input_precision='tf32x3')
    idx = bh*NC+block
    tl.store(OS+idx*C*M+c[:, None]*M+m[None, :], os, m[None, :] < M)
    tl.store(OV+idx*C*C+c[:, None]*C+c[None, :], ov)
    tl.store(FS+idx*M*M+m[:, None]*M+m[None, :], fs, (m[:, None] < M) & (m[None, :] < M))
    tl.store(FV+idx*M*C+m[:, None]*C+c[None, :], fv, m[:, None] < M)


@tr.jit
def slot_pairs(Q, Hist, Write, G, Beta, Lower, Response, BaseS, BaseQ, EndW, EndD,
               T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
               BM: tl.constexpr, NC: tl.constexpr):
    # Each output row has C x M work, avoiding a C x C x M register tensor.
    block, bh, i = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2)
    c, m = tl.arange(0, C), tl.arange(0, BM)
    t = block*C+c
    tok = ((bh//H)*T+t)*H+bh % H
    it = block*C+i
    itok = ((bh//H)*T+it)*H+bh % H
    valid = t < T
    mask = valid[:, None] & (m[None, :] < M)
    w = tl.load(Write+tok[:, None]*M+m[None, :], mask, 0)
    h = tl.load(Hist+itok*M+m, (m < M) & (it < T), 0).to(tl.float32)
    q = tl.load(Q+itok*M+m, (m < M) & (it < T), 0).to(tl.float32)
    beta = tl.load(Beta+itok, it < T, 0).to(tl.float32)
    # Product alpha[j+1]...alpha[i]; zero gates never appear in a denominator.
    shifted = tl.exp(tl.load(G+(tok[:, None]+H)*M+m[None, :],
                             (c[:, None] < i) & (t[:, None]+1 < T) & (m[None, :] < M), 0).to(tl.float32))
    windows = tl.associative_scan(shifted, 0, _mul, reverse=True)
    alpha = tl.exp(tl.load(G+tok[:, None]*M+m[None, :], mask & (c[:, None] <= i), 0).to(tl.float32))
    prefix = tl.reduce(alpha, 0, _mul)
    lower = tl.where(c < i, -beta*tl.sum(h[None, :]*w*windows, 1), 0.)
    response = tl.where(c <= i, tl.sum(q[None, :]*w*windows, 1), 0.)
    idx = bh*NC+block
    tl.store(Lower+idx*C*C+i*C+c, lower)
    tl.store(Response+idx*C*C+i*C+c, response)
    tl.store(BaseS+idx*C*M+i*M+m, -beta*prefix*h, m < M)
    tl.store(BaseQ+idx*C*M+i*M+m, prefix*q, m < M)
    if i == C-1:
        tl.store(EndW+idx*M*C+m[None, :]*C+c[:, None], w*windows, m[None, :] < M)
        tl.store(EndD+idx*M+m, prefix, m < M)


@tr.jit
def slot_prepare(Lower, Response, BaseS, BaseQ, EndW, EndD, Beta, OS, OV, FS, FV,
                 T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
                 BM: tl.constexpr, NC: tl.constexpr):
    block, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    c, m = tl.arange(0, C), tl.arange(0, BM)
    idx = bh*NC+block
    lower = tl.load(Lower+idx*C*C+c[:, None]*C+c[None, :])
    response = tl.load(Response+idx*C*C+c[:, None]*C+c[None, :])
    bs = tl.load(BaseS+idx*C*M+c[:, None]*M+m[None, :], m[None, :] < M, 0)
    bq = tl.load(BaseQ+idx*C*M+c[:, None]*M+m[None, :], m[None, :] < M, 0)
    endw = tl.load(EndW+idx*M*C+m[:, None]*C+c[None, :], m[:, None] < M, 0)
    endd = tl.load(EndD+idx*M+m, m < M, 0)
    t = block*C+c
    tok = ((bh//H)*T+t)*H+bh % H
    beta = tl.load(Beta+tok, t < T, 0).to(tl.float32)
    inv = _inverse(lower, C)
    us = tl.dot(inv, bs, input_precision='tf32x3')
    uv = inv*beta[None, :]
    os = bq+tl.dot(response, us, input_precision='tf32x3')
    ov = tl.where(c[:, None] >= c[None, :], tl.dot(response, uv, input_precision='tf32x3'), 0.)
    fs = endd[:, None]*(m[:, None] == m[None, :])+tl.dot(endw, us, input_precision='tf32x3')
    fv = tl.dot(endw, uv, input_precision='tf32x3')
    tl.store(OS+idx*C*M+c[:, None]*M+m[None, :], os, m[None, :] < M)
    tl.store(OV+idx*C*C+c[:, None]*C+c[None, :], ov)
    tl.store(FS+idx*M*M+m[:, None]*M+m[None, :], fs, (m[:, None] < M) & (m[None, :] < M))
    tl.store(FV+idx*M*C+m[:, None]*C+c[None, :], fv, m[:, None] < M)


@tr.jit
def local_bwd(Q, Hist, Value, G, Beta, Write, Tape, DS, DO, Work, DQ, DH, DV, DG, DB, DW,
              B: tl.constexpr, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr,
              C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr, SLOT: tl.constexpr):
    block, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    m, v = tl.arange(0, BM), tile*BV+tl.arange(0, BV)
    mask = (m[:, None] < M) & (v[None, :] < V)
    idx = bh*NC+block
    state = tl.load(Tape+idx*M*V+m[:, None]*V+v[None, :], mask, 0)
    n = tl.minimum(C, T-block*C)
    for s in range(n):
        tok = ((bh//H)*T+block*C+s)*H+bh % H
        if SLOT:
            a = tl.exp(tl.load(G+tok*M+m, m < M, 0).to(tl.float32))
        else:
            a = tl.full((BM,), 1., tl.float32)*tl.exp(tl.load(G+tok).to(tl.float32))
        pre = a[:, None]*state
        h = tl.load(Hist+tok*M+m, m < M, 0).to(tl.float32)
        w = tl.load(Write+tok*M+m, m < M, 0)
        beta = tl.load(Beta+tok).to(tl.float32)
        val = tl.load(Value+tok*V+v, v < V, 0).to(tl.float32)
        err = val-tl.sum(pre*h[:, None], 0)
        state = pre+beta*w[:, None]*err[None, :]
        tl.store(Work+(idx*C+s)*M*V+m[:, None]*V+v[None, :], pre, mask)
    ds = tl.load(DS+idx*M*V+m[:, None]*V+v[None, :], mask, 0)
    for step in range(n):
        s = n-1-step
        tok = ((bh//H)*T+block*C+s)*H+bh % H
        pre = tl.load(Work+(idx*C+s)*M*V+m[:, None]*V+v[None, :], mask, 0)
        h = tl.load(Hist+tok*M+m, m < M, 0).to(tl.float32)
        w = tl.load(Write+tok*M+m, m < M, 0)
        q = tl.load(Q+tok*M+m, m < M, 0).to(tl.float32)
        beta = tl.load(Beta+tok).to(tl.float32)
        val = tl.load(Value+tok*V+v, v < V, 0).to(tl.float32)
        do = tl.load(DO+tok*V+v, v < V, 0).to(tl.float32)
        err = val-tl.sum(pre*h[:, None], 0)
        state = pre+beta*w[:, None]*err[None, :]
        dq = tl.sum(state*do[None, :], 1)
        ds += q[:, None]*do[None, :]
        selected = tl.sum(w[:, None]*ds, 0)
        dw = beta*tl.sum(ds*err[None, :], 1)
        de = beta*selected
        db = tl.sum(selected*err, 0)
        dh = -tl.sum(pre*de[None, :], 1)
        dpre = ds-h[:, None]*de[None, :]
        part = tile*B*T*H+tok
        if SLOT:
            alpha = tl.exp(tl.load(G+tok*M+m, m < M, 0).to(tl.float32))
            tl.store(DG+part*M+m, tl.sum(dpre*pre, 1), m < M)
        else:
            alpha = tl.full((BM,), 1., tl.float32)*tl.exp(tl.load(G+tok).to(tl.float32))
            tl.store(DG+part, tl.sum(tl.sum(dpre*pre, 1), 0))
        ds = alpha[:, None]*dpre
        tl.store(DQ+part*M+m, dq, m < M)
        tl.store(DH+part*M+m, dh, m < M)
        tl.store(DW+part*M+m, dw, m < M)
        tl.store(DV+tok*V+v, de, v < V)
        tl.store(DB+part, db)


class _ChunkProtected(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, h, v, g, beta, p, initial, anchors, regularization, c):
        b, t, heads, m = h.shape
        width = v.shape[-1]
        r = anchors+1
        nc = tr.cdiv(t, c)
        alloc = lambda *s: torch.empty(s, device=q.device, dtype=torch.float32)  # noqa: E731
        bm = max(16, tr.next_power_of_2(m))
        write = alloc(b, t, heads, m)
        inv = alloc(b*heads, t, r, r)
        coeff = alloc(b*heads, t, r)
        direction_fwd[(t, b*heads)](h, p, write, inv, coeff, t, heads, m, bm, r, regularization, num_warps=4)
        os, ov, fs, fv = alloc(b, heads, nc, c, m), alloc(
            b, heads, nc, c, c), alloc(b, heads, nc, m, m), alloc(b, heads, nc, m, c)
        if g.ndim == 3:
            scalar_prepare[(nc, b*heads)](q, h, write, g, beta, os, ov, fs, fv, t, heads, m, c, bm, nc, num_warps=4)
        else:
            lower, response = alloc(b, heads, nc, c, c), alloc(b, heads, nc, c, c)
            bs, bq = alloc(b, heads, nc, c, m), alloc(b, heads, nc, c, m)
            ew, ed = alloc(b, heads, nc, m, c), alloc(b, heads, nc, m)
            slot_pairs[(nc, b*heads, c)](q, h, write, g, beta, lower, response,
                                         bs, bq, ew, ed, t, heads, m, c, bm, nc, num_warps=4)
            slot_prepare[(nc, b*heads)](lower, response, bs, bq, ew, ed, beta,
                                        os, ov, fs, fv, t, heads, m, c, bm, nc, num_warps=4)
        out, final = torch.empty_like(v), alloc(b, heads, m, width)
        tape, off = alloc(b, heads, nc, m, width), alloc(b, heads, nc, m, width)
        nv = tr.cdiv(width, 32)
        value_offset_kernel[(nc, b*heads, nv)](fv, v, off, t, heads, m, width, c, nc, bm, 32, num_warps=4)
        affine_scan(fs, off, initial, tape, final, False, None, 32)
        output_kernel[(nc, b*heads, nv)](os, ov, v, tape, out, t, heads, m, width, c, nc, bm, 32, num_warps=4)
        ctx.save_for_backward(q, h, v, g, beta, p, write, inv, coeff, tape, os, fs)
        ctx.cfg = (r, c, initial.dtype)
        return out, final

    @staticmethod
    @once_differentiable
    def backward(ctx, do, df):
        q, h, v, g, beta, p, write, inv, coeff, tape, os, fs = ctx.saved_tensors
        r, c, initial_dtype = ctx.cfg
        b, t, heads, m = h.shape
        width = v.shape[-1]
        nc = tr.cdiv(t, c)
        bm = max(16, tr.next_power_of_2(m))
        do, df = do.contiguous(), df.contiguous()
        ds, off = torch.empty_like(tape), torch.empty_like(tape)
        di = torch.empty(df.shape, device=q.device, dtype=initial_dtype)
        output_offset_kernel[(nc, b*heads, tr.cdiv(width, 32))](os, do, off, t, heads, m, width, c, nc, bm, 32, num_warps=4)
        affine_scan(fs, off, df, ds, di, True, None, 32)
        work = torch.empty((b, heads, nc, c, m, width), device=q.device, dtype=torch.float32)
        dq, dv, dg, db = [torch.empty_like(x) for x in (q, v, g, beta)]
        dh_direct = torch.empty(h.shape, device=q.device, dtype=torch.float32)
        dw = torch.empty_like(dh_direct)
        outputs = [dq, dh_direct, dg, db, dw]
        nv, bv = tr.cdiv(width, 64), 64
        partial = outputs if nv == 1 else [torch.empty((nv, *x.shape), device=q.device, dtype=torch.float32) for x in outputs]
        local_bwd[(nc, b*heads, nv)](q, h, v, g, beta, write, tape, ds, do, work, partial[0], partial[1], dv, *partial[2:],
                                     b, t, heads, m, width, c, nc, bm, bv, g.ndim == 4, num_warps=1)
        if nv > 1:
            for x, y in zip(partial, outputs):
                reduce_kernel[(tr.cdiv(y.numel(), 256),)](x, y, y.numel(), nv, tr.next_power_of_2(nv), 256)
        dh_geometry = zero(torch.empty_like(dh_direct))
        dp = zero(torch.empty_like(p))
        dh = torch.empty_like(h)
        direction_bwd[(t, b*heads)](h, p, inv, coeff, dw, dh_geometry, dp, t, heads, m, bm, r, num_warps=4)
        add_grad[(tr.cdiv(h.numel(), 256),)](dh_direct, dh_geometry, dh, h.numel(), 256)
        return dq, dh, dv, dg, db, dp, di, None, None, None


def chunk_protected_kapa(q, h, v, g, beta, inverse_metric, initial_state, *, anchors=2, regularization=.05, chunk_size=32):
    """Same transition/gradients as protected_kapa, with parallel chunks of 16/32/64.

    q/h [B,T,H,M], v [B,T,H,V], g scalar [B,T,H] or slot [B,T,H,M],
    beta [B,T,H], inverse_metric FP32 SPD [B,H,M,M], state [B,H,M,V].
    Fixed dictionary and no cross-call anchor cache; anchors cross chunk boundaries.
    """
    if q.ndim != 4 or q.shape != h.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError('q/h/v shapes disagree')
    b, t, heads, m = q.shape
    width = v.shape[-1]
    if inverse_metric.shape != (b, heads, m, m) or initial_state.shape != (b, heads, m, width):
        raise ValueError('metric/state shape mismatch')
    if beta.shape != (b, t, heads) or g.shape not in ((b, t, heads), (b, t, heads, m)):
        raise ValueError('gate shape mismatch')
    if min(b, t, heads, m, width) < 1 or m > 64 or width > 128:
        raise ValueError('nonempty tensors, M<=64 and V<=128 required')
    if not isinstance(anchors, int) or not 0 <= anchors <= 7:
        raise ValueError('anchors must be 0..7')
    if chunk_size not in (16, 32, 64):
        raise ValueError('chunk_size must be 16/32/64')
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError('regularization must be positive and finite')
    xs = (q, h, v, g, beta, inverse_metric, initial_state)
    if q.device.type != 'cuda' or any(x.device != q.device or x.dtype not in (torch.float32, torch.bfloat16) for x in xs):
        raise ValueError('one CUDA device and FP32/BF16 tensors required')
    if inverse_metric.dtype != torch.float32:
        raise ValueError('inverse_metric must be FP32')
    with torch.cuda.device(q.device):
        return _ChunkProtected.apply(*[x.contiguous() for x in xs], anchors, regularization, chunk_size)


def chunk_gaussian_protected_kapa(q, k, v, g, beta, centers, initial_state, *, temperature=.5, gram_ridge=.01,
                                  anchors=2, regularization=.05, chunk_size=32):
    """Raw-key all-Triton KAPA, including Gaussian and inverse-Gram backward."""
    qf, h, p = gaussian_features(q, k, centers, temperature=temperature, gram_ridge=gram_ridge)
    return chunk_protected_kapa(qf, h, v, g, beta, p, initial_state, anchors=anchors,
                                regularization=regularization, chunk_size=chunk_size)
