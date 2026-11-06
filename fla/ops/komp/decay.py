# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""GDN's learnable scalar/slot timescale, with FP32 Triton gate arithmetic."""
import math

import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from triton.language.extra.cuda import libdevice


def matched_dt_bias(alpha=.98, rate=1.):
    """Bias giving the requested decay at zero logits and exp(A_log)=rate."""
    if not 0. < alpha < 1. or not math.isfinite(rate) or rate <= 0.:
        raise ValueError('alpha must be in (0,1) and rate must be positive and finite')
    step = -math.log(alpha) / rate
    return step + math.log(-math.expm1(-step))


@tr.jit
def _values(x, log_rate):
    # Work with log(rate * softplus(x)), avoiding inf * 0 for extreme inputs.
    # Below -20, log(softplus(x)) = x to FP32 accuracy (error < 1.1e-9).
    sp = tl.maximum(x, 0.) + libdevice.log1p(tl.exp(-tl.abs(x)))
    log_sp = tl.where(x < -20., x, tl.log(sp))
    # z >= exp(20) gives exactly zero alpha and derivatives in FP32.
    z = tl.exp(tl.minimum(log_rate + log_sp, 20.))
    alpha = tl.exp(-z)
    log_sp_grad = tl.where(x < -20., 1., tl.sigmoid(x) / sp)
    return alpha, -alpha * z, log_sp_grad


@tr.jit
def _forward(X, A, Bias, Out, N: tl.constexpr, P: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    p = i % P
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    a = tl.load(A + p).to(tl.float32)
    bias = tl.load(Bias + p).to(tl.float32)
    alpha, _, _ = _values(x + bias, a)
    tl.store(Out + i, alpha, i < N)


@tr.jit
def _backward(X, A, Bias, DOut, DX, PA, PB, R: tl.constexpr, P: tl.constexpr,
              BR: tl.constexpr, BP: tl.constexpr):
    block = tl.program_id(0).to(tl.int64)
    r = block * BR + tl.arange(0, BR)
    p = tl.program_id(1).to(tl.int64) * BP + tl.arange(0, BP)
    mask = (r[:, None] < R) & (p[None, :] < P)
    x = tl.load(X + r[:, None] * P + p[None, :], mask, 0).to(tl.float32)
    a = tl.load(A + p, p < P, 0).to(tl.float32)
    bias = tl.load(Bias + p, p < P, 0).to(tl.float32)
    dout = tl.load(DOut + r[:, None] * P + p[None, :], mask, 0).to(tl.float32)
    _, da, ratio = _values(x + bias[None, :], a[None, :])
    da = dout * da
    dx = da * ratio
    tl.store(DX + r[:, None] * P + p[None, :], dx, mask)
    tl.store(PA + block * P + p, tl.sum(tl.where(mask, da, 0.), 0), p < P)
    tl.store(PB + block * P + p, tl.sum(tl.where(mask, dx, 0.), 0), p < P)


@tr.jit
def _reduce(PA, PB, DA, DB, P: tl.constexpr, NR: tl.constexpr,
            RR: tl.constexpr, BP: tl.constexpr):
    p = tl.program_id(0).to(tl.int64) * BP + tl.arange(0, BP)
    r = tl.arange(0, RR)
    mask = (r[:, None] < NR) & (p[None, :] < P)
    da = tl.load(PA + r[:, None] * P + p[None, :], mask, 0)
    db = tl.load(PB + r[:, None] * P + p[None, :], mask, 0)
    tl.store(DA + p, tl.sum(da, 0), p < P)
    tl.store(DB + p, tl.sum(db, 0), p < P)


class _GDNDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, A_log, dt_bias):
        alpha = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
        _forward[(tr.cdiv(logits.numel(), 256),)](
            logits, A_log, dt_bias, alpha, logits.numel(), A_log.numel(), 256)
        ctx.save_for_backward(logits, A_log, dt_bias)
        return alpha

    @staticmethod
    @once_differentiable
    def backward(ctx, dalpha):
        logits, A_log, dt_bias = ctx.saved_tensors
        rows, params = logits.shape[0] * logits.shape[1], A_log.numel()
        nr = tr.cdiv(rows, 128)
        bp = min(32, tr.next_power_of_2(params))
        pa = torch.empty((nr, params), dtype=torch.float32, device=logits.device)
        pb = torch.empty_like(pa)
        dx, da, db = [torch.empty_like(x) for x in (logits, A_log, dt_bias)]
        with torch.cuda.device(logits.device):
            _backward[(nr, tr.cdiv(params, bp))](
                logits, A_log, dt_bias, dalpha.contiguous(), dx, pa, pb, rows, params, 128, bp)
            _reduce[(tr.cdiv(params, bp),)](pa, pb, da, db, params, nr, tr.next_power_of_2(nr), bp)
        return dx, da, db


def gdn_decay(logits, A_log, dt_bias):
    """Return FP32 exp(-exp(A_log)*softplus(logits+dt_bias)).

    logits: CUDA [B,T,H] or [B,T,H,M], FP32/FP16/BF16.
    A_log, dt_bias: CUDA tensors of shape logits.shape[2:], normally FP32.
    Parameters broadcast over B/T; gradients reduce those axes in Triton.
    Finite model inputs are assumed; no host synchronization is introduced.
    """
    if logits.ndim not in (3, 4) or any(d == 0 for d in logits.shape):
        raise ValueError('logits must be a nonempty [B,T,H] or [B,T,H,M] tensor')
    if A_log.shape != logits.shape[2:] or dt_bias.shape != A_log.shape:
        raise ValueError('A_log and dt_bias must have shape logits.shape[2:]')
    for x in (logits, A_log, dt_bias):
        if not x.is_cuda or x.device != logits.device:
            raise ValueError('all gate tensors must be on the same CUDA device')
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError('gate tensors must be FP32, FP16 or BF16')
    with torch.cuda.device(logits.device):
        return _GDNDecay.apply(logits.contiguous(), A_log.contiguous(), dt_bias.contiguous())


@tr.jit
def _gsa_forward(X, Out, N: tl.constexpr, NORMALIZER: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    log_sigmoid = tl.minimum(x, 0.) - libdevice.log1p(tl.exp(-tl.abs(x)))
    tl.store(Out + i, tl.exp(log_sigmoid / NORMALIZER), i < N)


@tr.jit
def _gsa_backward(X, Alpha, DOut, DX, N: tl.constexpr, NORMALIZER: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    alpha = tl.load(Alpha + i, i < N, 0)
    dout = tl.load(DOut + i, i < N, 0).to(tl.float32)
    tl.store(DX + i, dout * alpha * tl.sigmoid(-x) / NORMALIZER, i < N)


class _GSADecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, normalizer):
        alpha = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
        _gsa_forward[(tr.cdiv(logits.numel(), 256),)](logits, alpha, logits.numel(), normalizer, 256)
        ctx.save_for_backward(logits, alpha)
        ctx.normalizer = normalizer
        return alpha

    @staticmethod
    @once_differentiable
    def backward(ctx, dalpha):
        logits, alpha = ctx.saved_tensors
        dx = torch.empty_like(logits)
        with torch.cuda.device(logits.device):
            _gsa_backward[(tr.cdiv(logits.numel(), 256),)](
                logits, alpha, dalpha.contiguous(), dx, logits.numel(), ctx.normalizer, 256)
        return dx, None


def gsa_decay(logits, normalizer=8.):
    """Return FP32 exp(logsigmoid(logits)/normalizer), with Triton backward.

    Accepts any nonempty CUDA FP32/FP16/BF16 shape; normalizer is a positive
    finite Python scalar, not a trainable tensor. At normalizer=1 this is sigmoid.
    """
    if not isinstance(normalizer, (int, float)) or not math.isfinite(normalizer) or normalizer <= 0.:
        raise ValueError('normalizer must be a positive finite Python scalar')
    if not logits.is_cuda or logits.numel() == 0:
        raise ValueError('logits must be a nonempty CUDA tensor')
    if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError('gate logits must be FP32, FP16 or BF16')
    with torch.cuda.device(logits.device):
        return _GSADecay.apply(logits.contiguous(), float(normalizer))
