# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton Gaussian features and differentiable inverse Gram for fixed KAPA centers.

Numeric forward/backward run on device; PyTorch only allocates and dispatches.
The inverse and accumulation use FP32. This prototype supports M<=64, D<=128.
"""
import math

import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable

from .geometry_chunk import _sweep_inverse


@tr.jit
def features_fwd(Q, K, U, QF, HF, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                 BM: tl.constexpr, BD: tl.constexpr, TAU: tl.constexpr):
    t, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    tok = ((bh//H)*T+t)*H+bh % H
    m, d = tl.arange(0, BM), tl.arange(0, BD)
    u = tl.load(U+bh*M*D+m[:, None]*D+d[None, :], (m[:, None] < M) & (d[None, :] < D), 0).to(tl.float32)
    q = tl.load(Q+tok*D+d, d < D, 0).to(tl.float32)
    k = tl.load(K+tok*D+d, d < D, 0).to(tl.float32)
    qf = tl.exp(-tl.sum((q[None, :]-u)*(q[None, :]-u), 1)/(2*TAU))
    hf = tl.exp(-tl.sum((k[None, :]-u)*(k[None, :]-u), 1)/(2*TAU))
    tl.store(QF+tok*M+m, qf, m < M)
    tl.store(HF+tok*M+m, hf, m < M)


@tr.jit
def gram_fwd(U, G, M: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr, TAU: tl.constexpr):
    i, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    m, d = tl.arange(0, BM), tl.arange(0, BD)
    u = tl.load(U+bh*M*D+m[:, None]*D+d[None, :], (m[:, None] < M) & (d[None, :] < D), 0).to(tl.float32)
    ui = tl.load(U+(bh*M+i)*D+d, d < D, 0).to(tl.float32)
    diff = ui[None, :]-u
    g = tl.exp(-tl.sum(diff*diff, 1)/(2*TAU))
    tl.store(G+bh*M*M+i*M+m, g, m < M)


@tr.jit
def inverse_fwd(G, P, M: tl.constexpr, BM: tl.constexpr, RIDGE: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    m = tl.arange(0, BM)
    mask = (m[:, None] < M) & (m[None, :] < M)
    a = tl.load(G+bh*M*M+m[:, None]*M+m[None, :], mask, 0)
    a += tl.where(m[:, None] == m[None, :], RIDGE, 0.)
    p = _sweep_inverse(a, M, BM)
    tl.store(P+bh*M*M+m[:, None]*M+m[None, :], p, mask)


@tr.jit
def inverse_bwd(P, DP, DG, M: tl.constexpr, BM: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    m = tl.arange(0, BM)
    mask = (m[:, None] < M) & (m[None, :] < M)
    p = tl.load(P+bh*M*M+m[:, None]*M+m[None, :], mask, 0)
    dp = tl.load(DP+bh*M*M+m[:, None]*M+m[None, :], mask, 0)
    tmp = tl.dot(tl.trans(p), dp, input_precision='tf32x3')
    dg = -tl.dot(tmp, tl.trans(p), input_precision='tf32x3')
    tl.store(DG+bh*M*M+m[:, None]*M+m[None, :], dg, mask)


@tr.jit
def gram_bwd(U, G, DG, DU, M: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr, TAU: tl.constexpr):
    i, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    m, d = tl.arange(0, BM), tl.arange(0, BD)
    u = tl.load(U+bh*M*D+m[:, None]*D+d[None, :], (m[:, None] < M) & (d[None, :] < D), 0).to(tl.float32)
    ui = tl.load(U+(bh*M+i)*D+d, d < D, 0).to(tl.float32)
    g = tl.load(G+bh*M*M+i*M+m, m < M, 0)
    dg = tl.load(DG+bh*M*M+i*M+m, m < M, 0)+tl.load(DG+bh*M*M+m*M+i, m < M, 0)
    du = -tl.sum((dg*g/TAU)[:, None]*(ui[None, :]-u), 0)
    tl.store(DU+(bh*M+i)*D+d, du, d < D)


@tr.jit
def features_bwd(Q, K, U, QF, HF, DQF, DHF, DQ, DK, DU, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                 BM: tl.constexpr, BD: tl.constexpr, TAU: tl.constexpr):
    t, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    tok = ((bh//H)*T+t)*H+bh % H
    m, d = tl.arange(0, BM), tl.arange(0, BD)
    mask = (m[:, None] < M) & (d[None, :] < D)
    u = tl.load(U+bh*M*D+m[:, None]*D+d[None, :], mask, 0).to(tl.float32)
    q = tl.load(Q+tok*D+d, d < D, 0).to(tl.float32)
    k = tl.load(K+tok*D+d, d < D, 0).to(tl.float32)
    aq = tl.load(QF+tok*M+m, m < M, 0)*tl.load(DQF+tok*M+m, m < M, 0)/TAU
    ak = tl.load(HF+tok*M+m, m < M, 0)*tl.load(DHF+tok*M+m, m < M, 0)/TAU
    dq = aq[:, None]*(u-q[None, :])
    dk = ak[:, None]*(u-k[None, :])
    tl.store(DQ+tok*D+d, tl.sum(dq, 0), d < D)
    tl.store(DK+tok*D+d, tl.sum(dk, 0), d < D)
    tl.atomic_add(DU+bh*M*D+m[:, None]*D+d[None, :], -dq-dk, mask, sem='relaxed')


@tr.jit
def cast_grad(S, D, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0, BLOCK)
    tl.store(D+i, tl.load(S+i, i < N, 0), i < N)


class _Gaussian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, u, temperature, ridge):
        b, t, h, d = q.shape
        m = u.shape[-2]
        alloc = lambda *s: torch.empty(s, device=q.device, dtype=torch.float32)  # noqa: E731
        qf, hf = alloc(b, t, h, m), alloc(b, t, h, m)
        gram, p = alloc(b, h, m, m), alloc(b, h, m, m)
        bm, bd = max(16, tr.next_power_of_2(m)), tr.next_power_of_2(d)
        features_fwd[(t, b*h)](q, k, u, qf, hf, t, h, m, d, bm, bd, temperature)
        gram_fwd[(m, b*h)](u, gram, m, d, bm, bd, temperature)
        inverse_fwd[(b*h,)](gram, p, m, bm, ridge)
        ctx.save_for_backward(q, k, u, qf, hf, gram, p)
        ctx.temperature = temperature
        return qf, hf, p

    @staticmethod
    @once_differentiable
    def backward(ctx, dqf, dhf, dp):
        q, k, u, qf, hf, gram, p = ctx.saved_tensors
        b, t, h, d = q.shape
        m = u.shape[-2]
        bm, bd = max(16, tr.next_power_of_2(m)), tr.next_power_of_2(d)
        dq, dk, du = [torch.empty_like(x) for x in (q, k, u)]
        dg = torch.empty_like(p)
        du32 = torch.empty(u.shape, device=u.device, dtype=torch.float32)
        inverse_bwd[(b*h,)](p, dp.contiguous(), dg, m, bm)
        gram_bwd[(m, b*h)](u, gram, dg, du32, m, d, bm, bd, ctx.temperature)
        features_bwd[(t, b*h)](q, k, u, qf, hf, dqf.contiguous(), dhf.contiguous(),
                               dq, dk, du32, t, h, m, d, bm, bd, ctx.temperature)
        cast_grad[(tr.cdiv(u.numel(), 256),)](du32, du, u.numel(), 256)
        return dq, dk, du, None, None


def gaussian_features(q, k, centers, *, temperature=.5, gram_ridge=.01):
    """Return FP32 RBF q/k features and inverse Gram, with raw-input gradients.

    q/k [B,T,H,D], centers [B,H,M,D]; FP32/BF16, one CUDA device.
    No normalization or stop-gradient. Centers are fixed within the sequence.
    FP32 inversion requires a well-resolved positive ridge (tested >= 1e-3).
    Higher-order differentiation is not supported.
    """
    if q.ndim != 4 or q.shape != k.shape or centers.ndim != 4:
        raise ValueError('q/k must match [B,T,H,D]; centers must be [B,H,M,D]')
    b, t, h, d = q.shape
    m = centers.shape[-2]
    if centers.shape != (b, h, m, d) or min(b, t, h, d, m) < 1 or m > 64 or d > 128:
        raise ValueError('matching nonempty shapes, M<=64 and D<=128 required')
    if any(x.device != q.device or x.dtype not in (torch.float32, torch.bfloat16) for x in (q, k, centers)) or q.device.type != 'cuda':
        raise ValueError('FP32/BF16 tensors on one CUDA device required')
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(gram_ridge) or gram_ridge <= 0:
        raise ValueError('temperature and Gram ridge must be positive and finite')
    with torch.cuda.device(q.device):
        return _Gaussian.apply(q.contiguous(), k.contiguous(), centers.contiguous(), temperature, gram_ridge)
