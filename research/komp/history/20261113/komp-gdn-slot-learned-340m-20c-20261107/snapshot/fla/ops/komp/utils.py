# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton initialization and input checks; Python only manages storage and shapes."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def zero_kernel(X, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0, BLOCK)
    tl.store(X+i, 0, i < N)


def zero(x):
    zero_kernel[(tr.cdiv(x.numel(), 256),)](x, x.numel(), 256)
    return x


@tr.jit
def _or(a, b):
    return a | b


@tr.jit
def check_kernel(Q, K, V, A, B, KI, VI, S, Flags,
                 NQ: tl.constexpr, NV: tl.constexpr, NA: tl.constexpr, NB: tl.constexpr, NS: tl.constexpr, NKI: tl.constexpr, NVI: tl.constexpr,
                 M: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0, BLOCK)
    q = tl.load(Q+i, i < NQ, 0).to(tl.float32)
    k = tl.load(K+i, i < NQ, 0).to(tl.float32)
    v = tl.load(V+i, i < NV, 0).to(tl.float32)
    a = tl.load(A+i, i < NA, 0).to(tl.float32)
    b = tl.load(B+i, i < NB, 0).to(tl.float32)
    ki = tl.load(KI+i, i < NKI, 0).to(tl.float32)
    vi = tl.load(VI+i, i < NVI, 0).to(tl.float32)
    s = tl.load(S+i, i < NS, 0)
    finite = (tl.abs(q) < float('inf')) & (tl.abs(k) < float('inf')) & (tl.abs(v) < float('inf'))
    finite &= (tl.abs(a) < float('inf')) & (tl.abs(b) < float('inf')) & (
        tl.abs(ki) < float('inf')) & (tl.abs(vi) < float('inf'))
    flag = (~finite).to(tl.int32) | (((s < 0) | (s >= M)).to(tl.int32) *
                                     2) | (((a < 0) | (a > 1) | (b < 0) | (b > 1)).to(tl.int32)*4)
    tl.store(Flags+tl.program_id(0), tl.reduce(flag, 0, _or))


@tr.jit
def flags_kernel(Flags, Result, N: tl.constexpr, BN: tl.constexpr):
    i = tl.arange(0, BN)
    f = tl.load(Flags+i, i < N, 0)
    tl.store(Result, tl.reduce(f, 0, _or))


def validate_inputs(q, k, v, a, b, ki, vi, slots):
    sizes = (q.numel(), v.numel(), a.numel(), b.numel(), slots.numel(), ki.numel(), vi.numel())
    n = tr.cdiv(max(sizes), 512)
    flags = torch.empty((n,), device=q.device, dtype=torch.int32)
    result = torch.empty((), device=q.device, dtype=torch.int32)
    with torch.cuda.device(q.device):
        check_kernel[(n,)](q, k, v, a, b, ki, vi, slots, flags, *sizes, ki.shape[-2], 512)
        flags_kernel[(1,)](flags, result, n, tr.next_power_of_2(n))
    code = result.item()
    if code & 2:
        raise ValueError('slot index out of range')
    if code & 1:
        raise ValueError('Inputs must be finite')
    if code & 4:
        raise ValueError('alpha and beta must lie in [0,1]')
