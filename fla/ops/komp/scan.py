# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Two-level affine scans over chunk maps: state_{i+1} = F_i state_i + b_i.

The forward recurrence and its boundary adjoint are both affine chunk recursions. Groups of
G consecutive chunks are first composed in parallel; one short serial scan runs over groups;
each group then expands its chunk-entry states in parallel. Serial depth is G + NC/G + G
instead of NC. The adjoint uses transposed maps in reverse order.
"""
import math

import torch
import triton as tr
import triton.language as tl

PREC = tl.constexpr('tf32x3')
WARPS = {'compose': 2, 'scan': 2, 'expand': 2}


@tr.jit
def _dot(a, b, P: tl.constexpr = 'tf32x3'):
    return tl.dot(a, b, input_precision=P)


@tr.jit
def _load_map(F, idx, M: tl.constexpr, BM: tl.constexpr, REV: tl.constexpr):
    m = tl.arange(0, BM)
    mask = (m[:, None] < M) & (m[None, :] < M)
    if REV:
        return tl.load(F+idx*M*M+m[None, :]*M+m[:, None], mask, 0)
    return tl.load(F+idx*M*M+m[:, None]*M+m[None, :], mask, 0)


@tr.jit
def value_offset_kernel(FV, Value, Off, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr,
                        C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr, P: tl.constexpr = 'tf32x3'):
    # b_i = FV_i v_i  (M x C)(C x V)
    block, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    m = tl.arange(0, BM)
    c = tl.arange(0, C)
    v = tile*BV+tl.arange(0, BV)
    idx = bh*NC+block
    fv = tl.load(FV+idx*M*C+m[:, None]*C+c[None, :], m[:, None] < M, 0)
    tok = ((bh//H)*T+block*C+c)*H+bh % H
    value = tl.load(Value+tok[:, None]*V+v[None, :], (block*C+c[:, None] < T) & (v[None, :] < V), 0).to(tl.float32)
    tl.store(Off+idx*M*V+m[:, None]*V+v[None, :], _dot(fv, value, P), (m[:, None] < M) & (v[None, :] < V))


@tr.jit
def output_offset_kernel(OS, DO, Off, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr,
                         C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr, P: tl.constexpr = 'tf32x3'):
    # b_i = OS_i^T dO_i  (M x C)(C x V)
    block, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    m = tl.arange(0, BM)
    c = tl.arange(0, C)
    v = tile*BV+tl.arange(0, BV)
    idx = bh*NC+block
    ost = tl.load(OS+idx*C*M+c[None, :]*M+m[:, None], m[:, None] < M, 0).to(tl.float32)
    tok = ((bh//H)*T+block*C+c)*H+bh % H
    do = tl.load(DO+tok[:, None]*V+v[None, :], (block*C+c[:, None] < T) & (v[None, :] < V), 0).to(tl.float32)
    tl.store(Off+idx*M*V+m[:, None]*V+v[None, :], _dot(ost, do, P), (m[:, None] < M) & (v[None, :] < V))


@tr.jit
def group_compose_kernel(F, Off, GA, GC, M: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, G: tl.constexpr,
                         NG: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr, REV: tl.constexpr, NV: tl.constexpr,
                         P: tl.constexpr = 'tf32x3'):
    # Compose the affine maps of one group in application order. Tile NV composes the linear
    # part once; value tiles 0..NV-1 only carry their offsets.
    g, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    m = tl.arange(0, BM)
    n = tl.minimum(G, NC-g*G)
    out = bh*NG+g
    if tile == NV:
        A = ((m[:, None] == m[None, :]) & (m[:, None] < M)).to(tl.float32)
        for s in range(n):
            block = NC-1-(g*G+s) if REV else g*G+s
            A = _dot(_load_map(F, bh*NC+block, M, BM, REV), A, P)
        tl.store(GA+out*M*M+m[:, None]*M+m[None, :], A, (m[:, None] < M) & (m[None, :] < M))
    else:
        v = tile*BV+tl.arange(0, BV)
        vmask = (m[:, None] < M) & (v[None, :] < V)
        acc = tl.zeros((BM, BV), tl.float32)
        for s in range(n):
            block = NC-1-(g*G+s) if REV else g*G+s
            idx = bh*NC+block
            acc = _dot(_load_map(F, idx, M, BM, REV), acc, P)+tl.load(Off+idx*M*V+m[:, None]*V+v[None, :], vmask, 0)
        tl.store(GC+out*M*V+m[:, None]*V+v[None, :], acc, vmask)


@tr.jit
def group_scan_kernel(GA, GC, Init, Entry, Final, M: tl.constexpr, V: tl.constexpr, NG: tl.constexpr,
                      BM: tl.constexpr, BV: tl.constexpr, P: tl.constexpr = 'tf32x3'):
    bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    m = tl.arange(0, BM)
    v = tile*BV+tl.arange(0, BV)
    vmask = (m[:, None] < M) & (v[None, :] < V)
    state = tl.load(Init+bh*M*V+m[:, None]*V+v[None, :], vmask, 0).to(tl.float32)
    for g in range(NG):
        out = bh*NG+g
        tl.store(Entry+out*M*V+m[:, None]*V+v[None, :], state, vmask)
        a = tl.load(GA+out*M*M+m[:, None]*M+m[None, :], (m[:, None] < M) & (m[None, :] < M), 0)
        state = _dot(a, state, P)+tl.load(GC+out*M*V+m[:, None]*V+v[None, :], vmask, 0)
    tl.store(Final+bh*M*V+m[:, None]*V+v[None, :], state, vmask)


@tr.jit
def group_expand_kernel(F, Off, Entry, Tape, M: tl.constexpr, V: tl.constexpr, NC: tl.constexpr, G: tl.constexpr,
                        NG: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr, REV: tl.constexpr, P: tl.constexpr = 'tf32x3'):
    # Tape[i] is the state before chunk i is applied (in application order).
    g, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    m = tl.arange(0, BM)
    v = tile*BV+tl.arange(0, BV)
    vmask = (m[:, None] < M) & (v[None, :] < V)
    state = tl.load(Entry+(bh*NG+g)*M*V+m[:, None]*V+v[None, :], vmask, 0)
    n = tl.minimum(G, NC-g*G)
    for s in range(n):
        if REV:
            block = NC-1-(g*G+s)
        else:
            block = g*G+s
        idx = bh*NC+block
        tl.store(Tape+idx*M*V+m[:, None]*V+v[None, :], state, vmask)
        if s+1 < n:
            f = _load_map(F, idx, M, BM, REV)
            state = _dot(f, state, P)+tl.load(Off+idx*M*V+m[:, None]*V+v[None, :], vmask, 0)


def affine_scan(F, off, init, tape, final, rev, group=None, bv=32, prec='tf32x3'):
    """F: [BH*NC,M,M] maps, off: [BH*NC,M,V]; writes tape (state before each chunk) and final."""
    bh, nc, m, width = tape.shape[0]*tape.shape[1], tape.shape[2], tape.shape[3], tape.shape[4]
    if group is None:
        # Minimises 2G + NC/G for power-of-two G.
        group = 1 << max(0, round(math.log2(math.sqrt(nc/2)))) if nc > 1 else 1
        group = max(1, min(group, nc))
    ng = tr.cdiv(nc, group)
    bm = max(16, tr.next_power_of_2(m))
    nv = tr.cdiv(width, bv)
    ga = torch.empty((bh, ng, m, m), device=tape.device, dtype=torch.float32)
    gc = torch.empty((bh, ng, m, width), device=tape.device, dtype=torch.float32)
    entry = torch.empty_like(gc)
    group_compose_kernel[(ng, bh, nv+1)](F, off, ga, gc, m, width, nc, group,
                                         ng, bm, bv, rev, nv, prec, num_warps=WARPS['compose'])
    group_scan_kernel[(bh, nv)](ga, gc, init, entry, final, m, width, ng, bm, bv, prec, num_warps=WARPS['scan'])
    group_expand_kernel[(ng, bh, nv)](F, off, entry, tape, m, width, nc, group,
                                      ng, bm, bv, rev, prec, num_warps=WARPS['expand'])
