# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Chunk boundary adjoint, local rank-two adjoint, and parallel state recomputation."""
import triton as tr
import triton.language as tl


@tr.jit
def outputs(Qmap, Omap, Entry, V, O, T: tl.constexpr, M: tl.constexpr, DV: tl.constexpr,
            C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr):
    ch, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    c = tl.arange(0, C)
    m = tl.arange(0, BM)
    v = tile*BV+tl.arange(0, BV)
    idx = bh*NC+ch
    t = ch*C+c
    qm = tl.load(Qmap+idx*C*M+c[:, None]*M+m[None, :], m[None, :] < M, 0)
    om = tl.load(Omap+idx*C*C+c[:, None]*C+c[None, :])
    entry = tl.load(Entry+idx*M*DV+m[:, None]*DV+v[None, :], (m[:, None] < M) & (v[None, :] < DV), 0)
    val = tl.load(V+(bh*T+t[:, None])*DV+v[None, :], (t[:, None] < T) & (v[None, :] < DV), 0).to(tl.float32)
    out = tl.dot(qm, entry, input_precision='tf32x3')+tl.dot(om, val, input_precision='tf32x3')
    tl.store(O+(bh*T+t[:, None])*DV+v[None, :], out, (t[:, None] < T) & (v[None, :] < DV))


@tr.jit
def boundary_source(Qmap, DO, J, T: tl.constexpr, M: tl.constexpr, DV: tl.constexpr,
                    C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr):
    ch, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    c = tl.arange(0, C)
    m = tl.arange(0, BM)
    v = tile*BV+tl.arange(0, BV)
    idx = bh*NC+ch
    t = ch*C+c
    qm = tl.load(Qmap+idx*C*M+c[:, None]*M+m[None, :], m[None, :] < M, 0)
    do = tl.load(DO+(bh*T+t[:, None])*DV+v[None, :], (t[:, None] < T) & (v[None, :] < DV), 0).to(tl.float32)
    j = tl.dot(tl.trans(qm), do, input_precision='tf32x3')
    tl.store(J+idx*M*DV+m[:, None]*DV+v[None, :], j, (m[:, None] < M) & (v[None, :] < DV))


@tr.jit
def boundary_scan(F, J, DSF, Exit, DS0, M: tl.constexpr, DV: tl.constexpr, NC: tl.constexpr,
                  BM: tl.constexpr, BV: tl.constexpr):
    bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    m = tl.arange(0, BM)
    v = tile*BV+tl.arange(0, BV)
    mask = (m[:, None] < M) & (v[None, :] < DV)
    adj = tl.load(DSF+bh*M*DV+m[:, None]*DV+v[None, :], mask, 0).to(tl.float32)
    for step in range(NC):
        ch = NC-1-step
        idx = bh*NC+ch
        tl.store(Exit+idx*M*DV+m[:, None]*DV+v[None, :], adj, mask)
        f = tl.load(F+idx*M*M+m[:, None]*M+m[None, :], (m[:, None] < M) & (m[None, :] < M), 0)
        j = tl.load(J+idx*M*DV+m[:, None]*DV+v[None, :], mask, 0)
        adj = tl.dot(tl.trans(f), adj, input_precision='tf32x3')+j
    tl.store(DS0+bh*M*DV+m[:, None]*DV+v[None, :], adj, mask)


@tr.jit
def local_adjoint(P, H, R, N, BE, G, DropSlot, DO, Exit, DI, DC, DVout,
                  T: tl.constexpr, M: tl.constexpr, DV: tl.constexpr, C: tl.constexpr,
                  NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr):
    ch, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    m = tl.arange(0, BM)
    v = tile*BV+tl.arange(0, BV)
    mask = (m[:, None] < M) & (v[None, :] < DV)
    adj = tl.load(Exit+(bh*NC+ch)*M*DV+m[:, None]*DV+v[None, :], mask, 0)
    for step in range(C):
        t = ch*C+C-1-step
        idx = bh*T+t
        if t < T:
            p = tl.load(P+idx*M+m, m < M, 0).to(tl.float32)
            h = tl.load(H+idx*M+m, m < M, 0).to(tl.float32)
            r = tl.load(R+idx*M+m, m < M, 0).to(tl.float32)
            n = tl.load(N+idx).to(tl.float32)
            beta = tl.load(BE+idx)
            g = tl.load(G+idx*M+m, m < M, 0)
            do = tl.load(DO+idx*DV+v, v < DV, 0).to(tl.float32)
            slot = tl.load(DropSlot+idx)
            e = (m == slot).to(tl.float32)
            a = tl.where(slot >= 0, p-e, 0.)
            b = tl.where(slot >= 0, e, p)
            di = n*do+tl.sum(b[:, None]*adj, 0)
            dc = tl.sum(a[:, None]*adj, 0)
            adj = tl.exp(g)[:, None]*(adj+e[:, None]*dc[None, :]-h[:, None]*(beta*di)[None, :]+r[:, None]*do[None, :])
            tl.store(DI+idx*DV+v, di, v < DV)
            tl.store(DC+idx*DV+v, dc, v < DV)
            tl.store(DVout+idx*DV+v, beta*di, v < DV)


@tr.jit
def affine_combine(a_left, b_left, a_right, b_right):
    return a_right*a_left, b_right+a_right*b_left


@tr.jit
def local_gradients_scan(G, U, Z, Entry, Exit, H, R, BE, DropSlot, DO, DI, DC, DP, DH, DR, DG, DB,
                         T: tl.constexpr, M: tl.constexpr, DV: tl.constexpr, C: tl.constexpr,
                         NC: tl.constexpr, BV: tl.constexpr, NV: tl.constexpr):
    ch, bh, sv = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    slot = sv//NV
    tile = sv % NV
    c = tl.arange(0, C)
    z = tl.arange(0, 2*C)
    v = tile*BV+tl.arange(0, BV)
    t = ch*C+c
    tz = ch*C+z//2  # noqa: F841
    idx = bh*NC+ch
    mask = (t[:, None] < T) & (v[None, :] < DV)
    g = tl.load(G+(bh*T+t)*M+slot, t < T, 0)
    alpha = tl.exp(g)
    zi = tl.load(Z+((bh*T+t[:, None])*2+1)*DV+v[None, :], mask, 0)
    zd = tl.load(Z+(bh*T+t[:, None])*2*DV+v[None, :], mask, 0)
    u0 = tl.load(U+(bh*T+t)*2*M+slot, t < T, 0)
    u1 = tl.load(U+((bh*T+t)*2+1)*M+slot, t < T, 0)
    drive = u0[:, None]*zd+u1[:, None]*zi
    aa = tl.broadcast_to(alpha[:, None], (C, BV))
    ap, bp = tl.associative_scan((aa, drive), 0, affine_combine)
    entry = tl.load(Entry+(idx*M+slot)*DV+v, v < DV, 0)
    states = ap*entry[None, :]+bp
    previous = tl.gather(states, tl.broadcast_to(tl.maximum(c-1, 0)[:, None], (C, BV)), 0)
    x = alpha[:, None]*tl.where(c[:, None] == 0, entry[None, :], previous)
    h = tl.load(H+(bh*T+t)*M+slot, t < T, 0).to(tl.float32)
    r = tl.load(R+(bh*T+t)*M+slot, t < T, 0).to(tl.float32)
    beta = tl.load(BE+bh*T+t, t < T, 0)
    drop = tl.load(DropSlot+bh*T+t, t < T, -1)
    e = (drop == slot).to(tl.float32)
    mask = (t[:, None] < T) & (v[None, :] < DV)
    do = tl.load(DO+(bh*T+t[:, None])*DV+v[None, :], mask, 0).to(tl.float32)
    di = tl.load(DI+(bh*T+t[:, None])*DV+v[None, :], mask, 0)
    dc = tl.load(DC+(bh*T+t[:, None])*DV+v[None, :], mask, 0)
    update = e[:, None]*dc-(h*beta)[:, None]*di+r[:, None]*do
    reverse_index = tl.broadcast_to((C-1-c)[:, None], (C, BV))
    ar = tl.gather(aa, reverse_index, 0)
    br = tl.gather(aa*update, reverse_index, 0)
    ar, br = tl.associative_scan((ar, br), 0, affine_combine)
    exit = tl.load(Exit+(idx*M+slot)*DV+v, v < DV, 0)
    before = tl.gather(ar*exit[None, :]+br, reverse_index, 0)
    following = tl.gather(before, tl.broadcast_to(tl.minimum(c+1, C-1)[:, None], (C, BV)), 0)
    adj = tl.where(c[:, None] == C-1, exit[None, :], following)
    dp = tl.sum(adj*tl.where(drop[:, None] >= 0, zd, zi), 1)
    dh = -tl.sum(x*(beta[:, None]*di), 1)
    dr = tl.sum(x*do, 1)
    dg = tl.sum((adj+update)*x, 1)
    off = (bh*T+t)*M+slot
    tl.atomic_add(DP+off, dp, t < T, sem='relaxed')
    tl.atomic_add(DH+off, dh, t < T, sem='relaxed')
    tl.atomic_add(DR+off, dr, t < T, sem='relaxed')
    tl.atomic_add(DG+off, dg, t < T, sem='relaxed')
    # dbeta = dot(v - h^T x, di), without division by beta (beta may be zero).
    db = -h*tl.sum(x*di, 1)
    tl.atomic_add(DB+bh*T+t, db, t < T, sem='relaxed')


@tr.jit
def scalar_gradients(V, Z, DO, DI, DN, DB, T: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr):
    t, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    v = tl.arange(0, BV)
    idx = bh*T+t
    val = tl.load(V+idx*DV+v, v < DV, 0).to(tl.float32)
    do = tl.load(DO+idx*DV+v, v < DV, 0).to(tl.float32)
    di = tl.load(DI+idx*DV+v, v < DV, 0)
    innovation = tl.load(Z+(idx*2+1)*DV+v, v < DV, 0)
    tl.store(DN+idx, tl.sum(innovation*do, 0))
    tl.atomic_add(DB+idx, tl.sum(val*di, 0), sem='relaxed')
