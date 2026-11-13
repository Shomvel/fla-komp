# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""All-Triton rank-two chunk core; FLA-style prepare/scan/local-backward stages."""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable


@tr.jit
def _inverse(lower, N: tl.constexpr):
    i = tl.arange(0, N)
    inv = (i[:, None] == i[None, :]).to(tl.float32)
    for step in tl.static_range(0, 6):
        w = 1 << step
        cross = tl.where((i[:, None]//(2*w) == i[None, :]//(2*w)) &
                         (i[:, None] % (2*w) >= w) & (i[None, :] % (2*w) < w), lower, 0.)
        if w < N:
            inv += tl.dot(tl.dot(inv, cross, input_precision='tf32x3'), inv, input_precision='tf32x3')
    return inv


@tr.jit
def prepare_kernel(Hist, Proj, Read, Alpha, Beta, Slot, OS, OV, FS, FV,
                   T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
                   BM: tl.constexpr, NC: tl.constexpr):
    block, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    b, head = bh//H, bh % H
    c = tl.arange(0, C)
    n = tl.arange(0, 2*C)
    m = tl.arange(0, BM)
    tok = (b*T+block*C+c)*H+head
    ntok = (b*T+block*C+n//2)*H+head
    valid = block*C+c < T
    nvalid = block*C+n//2 < T
    a = tl.load(Alpha+tok, valid, other=1).to(tl.float32)
    beta = tl.load(Beta+tok, valid, other=0).to(tl.float32)
    nb = tl.load(Beta+ntok, nvalid, other=0).to(tl.float32)
    slot = tl.load(Slot+tok, valid, other=0)
    ns = tl.load(Slot+ntok, nvalid, other=0)
    e = (m[None, :] == slot[:, None]).to(tl.float32)
    ne = (m[None, :] == ns[:, None]).to(tl.float32)
    hist = tl.load(Hist+ntok[:, None]*M+m[None, :], nvalid[:, None] & (m[None, :] < M), 0)
    proj = tl.load(Proj+ntok[:, None]*M+m[None, :], nvalid[:, None] & (m[None, :] < M), 0)
    read = tl.load(Read+tok[:, None]*M+m[None, :], valid[:, None] & (m[None, :] < M), 0)
    x = tl.where(n[:, None] % 2 == 0, proj-ne, -nb[:, None]*ne)
    x = tl.where(nvalid[:, None] & (m[None, :] < M), x, 0.)
    y = tl.where(n[:, None] % 2 == 0, ne, hist)
    write = beta[:, None]*e
    decay = tl.associative_scan(tl.where(c[:, None] > c[None, :], a[:, None], 1.), 0, _mul)
    prefix = tl.associative_scan(a, 0, _mul)
    # Gather expands token indices into the two rank coordinates, without copies.
    dn = tl.gather(decay, tl.broadcast_to((n//2)[:, None], (2*C, C)), 0)
    dnn = tl.gather(dn, tl.broadcast_to((n//2)[None, :], (2*C, 2*C)), 1)
    dc = tl.gather(decay, tl.broadcast_to((n//2)[None, :], (C, 2*C)), 1)
    pn = tl.gather(prefix, n//2, 0)
    lower = tl.dot(y, tl.trans(x), input_precision='tf32x3')*tl.where(n[:, None]//2 > n[None, :]//2, dnn, 0.)
    inv = _inverse(lower, 2*C)
    ys = pn[:, None]*y
    yv = tl.dot(y, tl.trans(write), input_precision='tf32x3')*tl.where(n[:, None]//2 > c[None, :], dn, 0.)
    zs = tl.dot(inv, ys, input_precision='tf32x3')
    zv = tl.dot(inv, yv, input_precision='tf32x3')
    rx = tl.dot(read, tl.trans(x), input_precision='tf32x3')*tl.where(c[:, None] >= n[None, :]//2, dc, 0.)
    rv = tl.dot(read, tl.trans(write), input_precision='tf32x3')*tl.where(c[:, None] >= c[None, :], decay, 0.)
    os = prefix[:, None]*read+tl.dot(rx, zs, input_precision='tf32x3')
    ov = rv+tl.dot(rx, zv, input_precision='tf32x3')
    end = tl.sum(tl.where(c[:, None] == C-1, decay, 0.), 0)
    endn = tl.gather(end, n//2, 0)
    endx = tl.trans(x*endn[:, None])
    pend = tl.sum(tl.where(c == C-1, prefix, 0.), 0)
    fs = pend*(m[:, None] == m[None, :])+tl.dot(endx, zs, input_precision='tf32x3')
    fv = tl.trans(write*end[:, None])+tl.dot(endx, zv, input_precision='tf32x3')
    idx = bh*NC+block
    tl.store(OS+idx*C*M+c[:, None]*M+m[None, :], os, m[None, :] < M)
    tl.store(OV+idx*C*C+c[:, None]*C+c[None, :], ov)
    tl.store(FS+idx*M*M+m[:, None]*M+m[None, :], fs, (m[:, None] < M) & (m[None, :] < M))
    tl.store(FV+idx*M*C+m[:, None]*C+c[None, :], fv, m[:, None] < M)


@tr.jit
def _mul(a, b):
    return a*b


@tr.jit
def prepare_slot_kernel(Hist, Proj, Read, Alpha, Beta, Slot, OS, OV, FS, FV,
                        T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
                        BM: tl.constexpr, NC: tl.constexpr):
    # Compose diagonal+rank-two affine maps within each independently scheduled
    # chunk. No division by gate products, including when a gate is exactly zero.
    block, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    m = tl.arange(0, BM)
    c = tl.arange(0, C)
    valid_m = m < M
    f = (m[:, None] == m[None, :]).to(tl.float32)
    f = tl.where(valid_m[:, None] & valid_m[None, :], f, 0.)
    g = tl.full((BM, C), 0., tl.float32)
    idx = bh*NC+block
    for s in range(C):
        tok = ((bh//H)*T+block*C+s)*H+bh % H
        valid = block*C+s < T
        a = tl.load(Alpha+tok*M+m, valid & valid_m, 1).to(tl.float32)
        h = tl.load(Hist+tok*M+m, valid & valid_m, 0).to(tl.float32)
        p = tl.load(Proj+tok*M+m, valid & valid_m, 0).to(tl.float32)
        r = tl.load(Read+tok*M+m, valid & valid_m, 0).to(tl.float32)
        beta = tl.load(Beta+tok, valid, 0).to(tl.float32)
        slot = tl.load(Slot+tok, valid, 0)
        e = ((m == slot) & valid & valid_m).to(tl.float32)
        f = a[:, None]*f
        g = a[:, None]*g
        old_f = tl.sum(e[:, None]*f, 0)
        old_g = tl.sum(e[:, None]*g, 0)
        pred_f = tl.sum(h[:, None]*f, 0)
        pred_g = tl.sum(h[:, None]*g, 0)
        f = f+(p-e)[:, None]*old_f[None, :]-beta*e[:, None]*pred_f[None, :]
        g = g+(p-e)[:, None]*old_g[None, :]+beta*e[:, None]*((c == s).to(tl.float32)-pred_g)[None, :]
        os = tl.sum(r[:, None]*f, 0)
        ov = tl.sum(r[:, None]*g, 0)
        tl.store(OS+idx*C*M+s*M+m, os, valid_m)
        tl.store(OV+idx*C*C+s*C+c, ov)
    tl.store(FS+idx*M*M+m[:, None]*M+m[None, :], f, valid_m[:, None] & valid_m[None, :])
    tl.store(FV+idx*M*C+m[:, None]*C+c[None, :], g, valid_m[:, None])


@tr.jit
def scan_kernel(OS, OV, FS, FV, Value, Init, Out, Final, Tape,
                T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
                NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr, SAVE: tl.constexpr):
    bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    m = tl.arange(0, BM)
    c = tl.arange(0, C)
    v = tile*BV+tl.arange(0, BV)
    mask = (m[:, None] < M) & (v[None, :] < V)
    state = tl.load(Init+bh*M*V+m[:, None]*V+v[None, :], mask, 0).to(tl.float32)
    for block in range(NC):
        idx = bh*NC+block
        if SAVE:
            tl.store(Tape+idx*M*V+m[:, None]*V+v[None, :], state, mask)
        os = tl.load(OS+idx*C*M+c[:, None]*M+m[None, :], m[None, :] < M, 0)
        ov = tl.load(OV+idx*C*C+c[:, None]*C+c[None, :])
        fs = tl.load(FS+idx*M*M+m[:, None]*M+m[None, :], (m[:, None] < M) & (m[None, :] < M), 0)
        fv = tl.load(FV+idx*M*C+m[:, None]*C+c[None, :], m[:, None] < M, 0)
        tok = ((bh//H)*T+block*C+c)*H+bh % H
        value = tl.load(Value+tok[:, None]*V+v[None, :], (block*C+c[:, None] < T) & (v[None, :] < V), 0).to(tl.float32)
        out = tl.dot(os, state, input_precision='tf32x3')+tl.dot(ov, value, input_precision='tf32x3')
        state = tl.dot(fs, state, input_precision='tf32x3')+tl.dot(fv, value, input_precision='tf32x3')
        tl.store(Out+tok[:, None]*V+v[None, :], out, (block*C+c[:, None] < T) & (v[None, :] < V))
    tl.store(Final+bh*M*V+m[:, None]*V+v[None, :], state, mask)


@tr.jit
def boundary_bwd_kernel(OS, FS, DO, DF, DS, DI,
                        T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
                        NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr):
    bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    m = tl.arange(0, BM)
    c = tl.arange(0, C)
    v = tile*BV+tl.arange(0, BV)
    mask = (m[:, None] < M) & (v[None, :] < V)
    ds = tl.load(DF+bh*M*V+m[:, None]*V+v[None, :], mask, 0).to(tl.float32)
    for step in range(NC):
        block = NC-1-step
        idx = bh*NC+block
        tl.store(DS+idx*M*V+m[:, None]*V+v[None, :], ds, mask)
        os = tl.load(OS+idx*C*M+c[:, None]*M+m[None, :], m[None, :] < M, 0)
        fs = tl.load(FS+idx*M*M+m[:, None]*M+m[None, :], (m[:, None] < M) & (m[None, :] < M), 0)
        tok = ((bh//H)*T+block*C+c)*H+bh % H
        do = tl.load(DO+tok[:, None]*V+v[None, :], (block*C+c[:, None] < T) & (v[None, :] < V), 0).to(tl.float32)
        ds = tl.dot(tl.trans(os), do, input_precision='tf32x3')+tl.dot(tl.trans(fs), ds, input_precision='tf32x3')
    tl.store(DI+bh*M*V+m[:, None]*V+v[None, :], ds, mask)


@tr.jit
def local_bwd_kernel(Hist, Proj, Read, Value, Alpha, Beta, Slot, Tape, DS, DO, Work, DH, DP, DR, DV, DA, DB,
                     B: tl.constexpr, T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr,
                     C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr, SLOT_DECAY: tl.constexpr):
    block, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    m = tl.arange(0, BM)
    v = tile*BV+tl.arange(0, BV)
    mask = (m[:, None] < M) & (v[None, :] < V)
    idx = bh*NC+block
    state = tl.load(Tape+idx*M*V+m[:, None]*V+v[None, :], mask, 0)
    for s in range(min(C, T-block*C)):
        tok = ((bh//H)*T+block*C+s)*H+bh % H
        tl.store(Work+(idx*C+s)*M*V+m[:, None]*V+v[None, :], state, mask)
        h = tl.load(Hist+tok*M+m, m < M, 0)
        p = tl.load(Proj+tok*M+m, m < M, 0)
        val = tl.load(Value+tok*V+v, v < V, 0).to(tl.float32)
        if SLOT_DECAY:
            a = tl.load(Alpha+tok*M+m, m < M, 1).to(tl.float32)[:, None]
        else:
            a = tl.load(Alpha+tok).to(tl.float32)
        beta = tl.load(Beta+tok).to(tl.float32)
        e = (m == tl.load(Slot+tok)).to(tl.float32)
        pre = a*state
        old = tl.sum(pre*e[:, None], 0)
        err = val-tl.sum(pre*h[:, None], 0)
        state = pre+(p-e)[:, None]*old[None, :]+e[:, None]*(beta*err)[None, :]
    ds = tl.load(DS+idx*M*V+m[:, None]*V+v[None, :], mask, 0)
    for step in range(min(C, T-block*C)):
        s = min(C, T-block*C)-1-step
        tok = ((bh//H)*T+block*C+s)*H+bh % H
        prev = tl.load(Work+(idx*C+s)*M*V+m[:, None]*V+v[None, :], mask, 0)
        h = tl.load(Hist+tok*M+m, m < M, 0)
        p = tl.load(Proj+tok*M+m, m < M, 0)
        r = tl.load(Read+tok*M+m, m < M, 0)
        val = tl.load(Value+tok*V+v, v < V, 0).to(tl.float32)
        if SLOT_DECAY:
            a = tl.load(Alpha+tok*M+m, m < M, 1).to(tl.float32)[:, None]
        else:
            a = tl.load(Alpha+tok).to(tl.float32)
        beta = tl.load(Beta+tok).to(tl.float32)
        e = (m == tl.load(Slot+tok)).to(tl.float32)
        do = tl.load(DO+tok*V+v, v < V, 0).to(tl.float32)
        dr = tl.sum(state*do[None, :], 1)
        ds += r[:, None]*do[None, :]
        pre = a*prev
        old = tl.sum(pre*e[:, None], 0)
        err = val-tl.sum(pre*h[:, None], 0)
        selected = tl.sum(ds*e[:, None], 0)
        dp = tl.sum(ds*old[None, :], 1)
        d_old = tl.sum(ds*(p-e)[:, None], 0)
        de = beta*selected
        db = tl.sum(selected*err, 0)
        dh = -tl.sum(pre*de[None, :], 1)
        dpre = ds+e[:, None]*d_old[None, :]-h[:, None]*de[None, :]
        if SLOT_DECAY:
            da = tl.sum(dpre*prev, 1)
        else:
            da = tl.sum(tl.sum(dpre*prev, 1), 0)
        ds = a*dpre
        state = prev
        part = tile*B*T*H+tok
        tl.store(DH+part*M+m, dh, m < M)
        tl.store(DP+part*M+m, dp, m < M)
        tl.store(DR+part*M+m, dr, m < M)
        if SLOT_DECAY:
            tl.store(DA+part*M+m, da, m < M)
        else:
            tl.store(DA+part, da)
        tl.store(DB+part, db)
        tl.store(DV+tok*V+v, de, v < V)


@tr.jit
def reduce_kernel(X, Y, N: tl.constexpr, R: tl.constexpr, BR: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0, BLOCK)
    r = tl.arange(0, BR)
    x = tl.load(X+r[:, None]*N+i[None, :], (r[:, None] < R) & (i[None, :] < N), 0)
    tl.store(Y+i, tl.sum(x, 0), i < N)


def prepare(h, p, r, a, b, slots, c, warps=4):
    batch, t, heads, m = h.shape
    nc = tr.cdiv(t, c)
    alloc = lambda *s: torch.empty(s, device=h.device, dtype=torch.float32)  # noqa: E731
    maps = (alloc(batch, heads, nc, c, m), alloc(batch, heads, nc, c, c),
            alloc(batch, heads, nc, m, m), alloc(batch, heads, nc, m, c))
    kernel = prepare_slot_kernel if a.ndim == 4 else prepare_kernel
    kernel[(nc, batch*heads)](h, p, r, a, b, slots, *maps, t, heads, m, c, max(16, tr.next_power_of_2(m)), nc, num_warps=warps)
    return maps


class _Chunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, p, r, v, a, b, slots, initial, c, save):
        batch, t, heads, m = h.shape
        width = v.shape[-1]
        nc = tr.cdiv(t, c)
        maps = prepare(h, p, r, a, b, slots, c)
        out = torch.empty(v.shape, device=v.device, dtype=torch.float32)
        final = torch.empty(initial.shape, device=v.device, dtype=torch.float32)
        tape = torch.empty((batch, heads, nc, m, width) if save else (1,), device=v.device, dtype=torch.float32)
        scan_kernel[(batch*heads, tr.cdiv(width, 32))](*maps, v, initial, out, final, tape,
                                                       t, heads, m, width, c, nc, max(16, tr.next_power_of_2(m)), 32, save, num_warps=4)
        ctx.save_for_backward(h, p, r, v, a, b, slots, tape, maps[0], maps[2])
        ctx.c = c
        ctx.initial_dtype = initial.dtype
        return out, final

    @staticmethod
    @once_differentiable
    def backward(ctx, do, df):
        h, p, r, v, a, b, slots, tape, os, fs = ctx.saved_tensors
        batch, t, heads, m = h.shape
        width = v.shape[-1]
        c = ctx.c
        nc = tr.cdiv(t, c)
        nv = tr.cdiv(width, 32)
        bm = max(16, tr.next_power_of_2(m))
        do, df = do.contiguous(), df.contiguous()
        ds = torch.empty_like(tape)
        di = torch.empty(df.shape, device=v.device, dtype=ctx.initial_dtype)
        boundary_bwd_kernel[(batch*heads, nv)](os, fs, do, df, ds, di, t, heads, m, width, c, nc, bm, 32, num_warps=4)
        work = torch.empty((batch, heads, nc, c, m, width), device=v.device, dtype=torch.float32)
        partial = [torch.empty((nv, *x.shape), device=v.device, dtype=torch.float32) for x in (h, p, r, a, b)]
        result = [torch.empty_like(x) for x in (h, p, r, a, b)]
        dv = torch.empty_like(v)
        local_bwd_kernel[(nc, batch*heads, nv)](h, p, r, v, a, b, slots, tape, ds, do, work, *partial[:3], dv, *partial[3:],
                                                batch, t, heads, m, width, c, nc, bm, 32, a.ndim == 4, num_warps=4)
        for x, y in zip(partial, result):
            reduce_kernel[(tr.cdiv(y.numel(), 256),)](x, y, y.numel(), nv, tr.next_power_of_2(nv), 256)
        return *result[:3], dv, *result[3:], None, di, None, None


def triton_chunk(h, p, r, v, alpha, beta, slots, initial_values, chunk_size=16):
    """FP32 geometry; alpha is [B,T,H] or [B,T,H,M], beta is [B,T,H]."""
    if alpha.shape not in (h.shape[:3], h.shape):
        raise ValueError('alpha must have shape [B,T,H] or [B,T,H,M]')
    if beta.shape != h.shape[:3] or slots.shape != h.shape[:3]:
        raise ValueError('beta and slots must have shape [B,T,H]')
    if chunk_size not in (16, 32):
        raise ValueError('chunk_size must be 16 or 32')
    inputs = (h, p, r, v, alpha, beta, slots, initial_values)
    save = torch.is_grad_enabled() and any(x.requires_grad for x in inputs)
    with torch.cuda.device(h.device):
        return _Chunk.apply(*[x.contiguous() for x in inputs], chunk_size, save)
