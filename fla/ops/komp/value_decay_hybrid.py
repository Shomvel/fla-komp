# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton value-state scan/adjoint with differentiable PyTorch geometry.

Hybrid prototype: normalization, kernels and batched Gram solves use PyTorch;
value recurrence and its backward use Triton. Only chunk entries and writes
are retained, not a token-by-dictionary-by-value state tape.
"""

import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@tr.jit
def value_decay_fwd_kernel(
    AK,
    AQ,
    P,
    V,
    G,
    Beta,
    S0,
    O,
    SF,
    Entry,
    W,
    GC,
    NC: tl.constexpr,
    C: tl.constexpr,
    M: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    m = tl.arange(0, BM)
    r = tl.arange(0, BR)
    v = tile * BV + tl.arange(0, BV)
    state = tl.load(S0 + (bh * M + m[:, None]) * D + v[None, :], (m[:, None] < M) & (v[None, :] < D), 0).to(tl.float32)
    for ch in range(NC):
        idx = bh * NC + ch
        tl.store(Entry + (idx * M + m[:, None]) * D + v[None, :], state, (m[:, None] < M) & (v[None, :] < D))
        # keep live state in registers; the entry tape is not a cross-layout communication buffer.
        x = tl.gather(state, tl.broadcast_to(tl.minimum(r, M - 1)[:, None], (BR, BV)), 0)
        x = tl.where(r[:, None] < M, x, 0.0)
        cum = tl.full((BV,), 0, tl.float32)
        for t in range(C):
            g = tl.load(G + ((idx * C + t) * D + v), v < D, 0).to(tl.float32)
            cum += g
            tl.store(GC + ((idx * C + t) * D + v), cum, v < D)
            x = x * tl.exp(g)[None, :]
            ak = tl.load(AK + (idx * C + t) * R + r, r < R, 0).to(tl.float32)
            aq = tl.load(AQ + (idx * C + t) * R + r, r < R, 0).to(tl.float32)
            val = tl.load(V + (idx * C + t) * D + v, v < D, 0).to(tl.float32)
            beta = tl.load(Beta + idx * C + t).to(tl.float32)
            write = beta * (val - tl.sum(ak[:, None] * x, 0))
            tl.store(W + (idx * C + t) * D + v, write, v < D)
            x = tl.where(r[:, None] == M + t, write[None, :], x)
            out = tl.sum(aq[:, None] * x, 0)
            tl.store(O + (idx * C + t) * D + v, out, v < D)
        proj = tl.load(P + (idx * M + m[:, None]) * R + r[None, :], (m[:, None] < M) & (r[None, :] < R), 0).to(tl.float32)
        state = tl.dot(proj, x, input_precision="tf32x3")
    tl.store(SF + (bh * M + m[:, None]) * D + v[None, :], state, (m[:, None] < M) & (v[None, :] < D))


@tr.jit
def value_decay_bwd_kernel(
    AK,
    AQ,
    P,
    V,
    G,
    Beta,
    Entry,
    W,
    GC,
    DO,
    DSF,
    DAK,
    DAQ,
    DP,
    DV,
    DG,
    DB,
    DS0,
    NC: tl.constexpr,
    C: tl.constexpr,
    M: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    m = tl.arange(0, BM)
    r = tl.arange(0, BR)
    v = tile * BV + tl.arange(0, BV)
    adj = tl.load(DSF + (bh * M + m[:, None]) * D + v[None, :], (m[:, None] < M) & (v[None, :] < D), 0).to(tl.float32)
    for back in range(NC):
        ch = NC - 1 - back
        idx = bh * NC + ch
        entry = tl.load(Entry + (idx * M + r[:, None]) * D + v[None, :], (r[:, None] < M) & (v[None, :] < D), 0).to(tl.float32)
        j = r - M
        writes = tl.load(
            W + (idx * C + j[:, None]) * D + v[None, :], (j[:, None] >= 0) & (j[:, None] < C) & (v[None, :] < D), 0
        ).to(tl.float32)
        born = tl.load(
            GC + (idx * C + j[:, None]) * D + v[None, :], (j[:, None] >= 0) & (j[:, None] < C) & (v[None, :] < D), 0
        ).to(tl.float32)
        last = tl.load(GC + (idx * C + C - 1) * D + v, v < D, 0).to(tl.float32)
        xend = tl.where(r[:, None] < M, entry, writes) * tl.exp(last[None, :] - born)
        proj = tl.load(P + (idx * M + m[:, None]) * R + r[None, :], (m[:, None] < M) & (r[None, :] < R), 0).to(tl.float32)
        dp = tl.dot(adj, tl.trans(xend), input_precision="tf32x3")
        tl.atomic_add(DP + (idx * M + m[:, None]) * R + r[None, :], dp, (m[:, None] < M) & (r[None, :] < R), sem="relaxed")
        a = tl.dot(tl.trans(proj), adj, input_precision="tf32x3")
        for step in range(C):
            t = C - 1 - step
            cum = tl.load(GC + (idx * C + t) * D + v, v < D, 0).to(tl.float32)
            active = r < M + t
            exponent = tl.where(active[:, None], cum[None, :] - born, 0.0)
            pre = tl.where(active[:, None], tl.where(r[:, None] < M, entry, writes) * tl.exp(exponent), 0.0)
            write = tl.load(W + (idx * C + t) * D + v, v < D, 0).to(tl.float32)
            after = pre + tl.where(r[:, None] == M + t, write[None, :], 0.0)
            do = tl.load(DO + (idx * C + t) * D + v, v < D, 0).to(tl.float32)
            aq = tl.load(AQ + (idx * C + t) * R + r, r < R, 0).to(tl.float32)
            ak = tl.load(AK + (idx * C + t) * R + r, r < R, 0).to(tl.float32)
            daq = tl.sum(after * do[None, :], 1)
            tl.atomic_add(DAQ + (idx * C + t) * R + r, daq, r < R, sem="relaxed")
            a += aq[:, None] * do[None, :]
            dw = tl.sum(tl.where(r[:, None] == M + t, a, 0.0), 0)
            beta = tl.load(Beta + idx * C + t).to(tl.float32)
            val = tl.load(V + (idx * C + t) * D + v, v < D, 0).to(tl.float32)
            pred = tl.sum(ak[:, None] * pre, 0)
            db = tl.sum(tl.where(v < D, dw * (val - pred), 0.0), 0)
            tl.atomic_add(DB + idx * C + t, db, sem="relaxed")
            tl.store(DV + (idx * C + t) * D + v, beta * dw, v < D)
            dak = -tl.sum(pre * (beta * dw)[None, :], 1)
            tl.atomic_add(DAK + (idx * C + t) * R + r, dak, r < R, sem="relaxed")
            a = tl.where(active[:, None], a - ak[:, None] * (beta * dw)[None, :], 0.0)
            tl.store(DG + (idx * C + t) * D + v, tl.sum(a * pre, 0), v < D)
            g = tl.load(G + (idx * C + t) * D + v, v < D, 0).to(tl.float32)
            a *= tl.exp(g)[None, :]
        # gather the entry adjoint even when BR differs from BM.
        adj = tl.gather(a, tl.broadcast_to(m[:, None], (BM, BV)), 0)
    tl.store(DS0 + (bh * M + m[:, None]) * D + v[None, :], adj, (m[:, None] < M) & (v[None, :] < D))


class _Scan(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, ak, aq, p, v, g, beta, s0):
        bh, nc, c, r = ak.shape
        m = s0.shape[-2]
        d = v.shape[-1]
        o = torch.empty_like(v)
        sf = torch.empty_like(s0)
        entry = torch.empty((bh, nc, m, d), device=v.device, dtype=torch.float32)
        w = torch.empty_like(v)
        gc = torch.empty_like(v)
        opts = (nc, c, m, r, d, max(16, tr.next_power_of_2(m)), max(16, tr.next_power_of_2(r)), 16)
        value_decay_fwd_kernel[(bh, tr.cdiv(d, 16))](ak, aq, p, v, g, beta, s0, o, sf, entry, w, gc, *opts, num_warps=4)
        ctx.save_for_backward(ak, aq, p, v, g, beta, entry, w, gc)
        ctx.opts = opts
        return o, sf

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    @once_differentiable
    def backward(ctx, do, dsf):
        ak, aq, p, v, g, beta, entry, w, gc = ctx.saved_tensors
        bh = ak.shape[0]
        nc, c, m, r, d, *_ = ctx.opts
        do = torch.zeros_like(v) if do is None else do.contiguous()
        dsf = torch.zeros((bh, m, d), device=v.device, dtype=torch.float32) if dsf is None else dsf.contiguous()
        dak, daq, dp, db = [torch.zeros_like(x) for x in (ak, aq, p, beta)]
        dv, dg = torch.empty_like(v), torch.empty_like(g)
        ds0 = torch.empty_like(dsf)
        value_decay_bwd_kernel[(bh, tr.cdiv(d, 16))](
            ak, aq, p, v, g, beta, entry, w, gc, do, dsf, dak, daq, dp, dv, dg, db, ds0, *ctx.opts, num_warps=4
        )
        return dak, daq, dp, dv, dg, db, ds0


def _segment(q, k, v, g, beta, keys, coef, c, temperature, ridge):
    """All chunks have equal length; build FIFO geometry without a state loop."""
    b, h, t, dk = q.shape
    m = keys.shape[-2]
    dv = v.shape[-1]
    nc = t // c
    pool = torch.cat((keys, k), -2)
    indices = torch.arange(nc, device=q.device)[:, None] * c + torch.arange(m + c, device=q.device)[None, :]
    dictionary = pool[:, :, indices, :]
    qc = q.reshape(b, h, nc, c, dk)
    kc = k.reshape(b, h, nc, c, dk)

    def kern(x, y):
        return torch.exp(((x @ y.transpose(-1, -2)) - 1) / temperature)

    ak, aq = kern(kc, dictionary), kern(qc, dictionary)
    kept, evicted = dictionary[..., c:, :], dictionary[..., :c, :]
    gram = kern(kept, kept) + ridge * torch.eye(m, device=q.device)
    proj = torch.linalg.solve(gram, kern(kept, evicted))
    projection = torch.cat((proj, torch.eye(m, device=q.device).expand(b, h, nc, m, m)), -1)
    shape = (b * h, nc, c)
    o, sf = _Scan.apply(
        ak.reshape(*shape, m + c).contiguous(),
        aq.reshape(*shape, m + c).contiguous(),
        projection.reshape(b * h, nc, m, m + c).contiguous(),
        v.reshape(*shape, dv).contiguous(),
        g.reshape(*shape, dv).contiguous(),
        beta.reshape(*shape).contiguous(),
        coef.reshape(b * h, m, dv).contiguous(),
    )
    return o.reshape(b, h, t, dv), pool[..., -m:, :], sf.reshape(b, h, m, dv)


def block_komp_value_decay_triton(q, k, v, g, beta, initial_state, *, chunk_size=128, temperature=0.5, ridge=1e-4):
    from fla.ops.komp.value_decay import _check

    _check(q, k, v, g, beta, initial_state, chunk_size, temperature, ridge)
    if not q.is_cuda or any(x.dtype == torch.float64 for x in (q, k, v, g, beta, *initial_state)):
        raise ValueError("Triton requires CUDA and FP32/BF16/FP16 inputs; use recurrent for FP64")
    if chunk_size > 128 or initial_state[0].shape[-2] > 64:
        raise ValueError("prototype supports chunk_size<=128 and M<=64")
    dtype = v.dtype
    with torch.autocast("cuda", enabled=False):
        q, k = [F.normalize(x.float(), dim=-1, eps=1e-6).transpose(1, 2) for x in (q, k)]
        v, g, beta = [x.float().transpose(1, 2) for x in (v, g, beta)]
        keys = F.normalize(initial_state[0].float(), dim=-1, eps=1e-6)
        coef = initial_state[1].float()
        full = q.shape[2] // chunk_size * chunk_size
        outs = []
        for start, end, c in ((0, full, chunk_size), (full, q.shape[2], q.shape[2] - full)):
            if start == end:
                continue
            out, keys, coef = _segment(
                q[:, :, start:end],
                k[:, :, start:end],
                v[:, :, start:end],
                g[:, :, start:end],
                beta[:, :, start:end],
                keys,
                coef,
                c,
                temperature,
                ridge,
            )
            outs.append(out)
        return torch.cat(outs, 2).transpose(1, 2).to(dtype), (keys, coef)
