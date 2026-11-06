"""Scalar-decay recurrence with a matrix-form chunk backward.

Forward: the existing per-chunk transfer maps; a serial scan carries only the chunk-entry
state, and outputs are computed for all chunks in parallel. Backward: the serial boundary
scan gives the adjoint at every chunk end; each chunk then recomputes its pseudo-values
U = T (Bs S0 + Bv v) and their adjoints Z = T^T (Xf^T dS + Ox^T dO), and obtains every
coefficient gradient with masked matrix products. Decay products are formed by
multiplicative scans, never by division, so exact zero gates remain supported. The gate
gradient pairs the basis expansions of the state and adjoint, again without dividing by alpha.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from .triton_chunk import _inverse, _mul, prepare, boundary_bwd_kernel, reduce_kernel

PREC = tl.constexpr('tf32x3')


@tr.jit
def _rows2(x, C: tl.constexpr, X: tl.constexpr):
    n = tl.arange(0, 2*C)
    return tl.gather(x, tl.broadcast_to((n//2)[:, None], (2*C, X)), 0)


@tr.jit
def _cols2(x, C: tl.constexpr, X: tl.constexpr):
    n = tl.arange(0, 2*C)
    return tl.gather(x, tl.broadcast_to((n//2)[None, :], (X, 2*C)), 1)


@tr.jit
def _vec2(x, C: tl.constexpr):
    n = tl.arange(0, 2*C)
    return tl.gather(x, n//2, 0)


@tr.jit
def _pick(x, C: tl.constexpr, K: tl.constexpr):
    # Rows 2c+K of a pseudo-row matrix, by an exact 0/1 selection product.
    c = tl.arange(0, C)
    n = tl.arange(0, 2*C)
    sel = (n[None, :] == 2*c[:, None]+K).to(tl.float32)
    return tl.dot(sel, x, input_precision='ieee')


@tr.jit
def _dot(a, b):
    return tl.dot(a, b, input_precision=PREC)


@tr.jit
def scan_state_kernel(FS, FV, Value, Init, Final, Tape,
                      T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
                      NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr):
    bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    m = tl.arange(0, BM)
    c = tl.arange(0, C)
    v = tile*BV+tl.arange(0, BV)
    mask = (m[:, None] < M) & (v[None, :] < V)
    state = tl.load(Init+bh*M*V+m[:, None]*V+v[None, :], mask, 0).to(tl.float32)
    for block in range(NC):
        idx = bh*NC+block
        tl.store(Tape+idx*M*V+m[:, None]*V+v[None, :], state, mask)
        fs = tl.load(FS+idx*M*M+m[:, None]*M+m[None, :], (m[:, None] < M) & (m[None, :] < M), 0)
        fv = tl.load(FV+idx*M*C+m[:, None]*C+c[None, :], m[:, None] < M, 0)
        tok = ((bh//H)*T+block*C+c)*H+bh % H
        value = tl.load(Value+tok[:, None]*V+v[None, :], (block*C+c[:, None] < T) & (v[None, :] < V), 0).to(tl.float32)
        state = _dot(fs, state)+_dot(fv, value)
    tl.store(Final+bh*M*V+m[:, None]*V+v[None, :], state, mask)


@tr.jit
def output_kernel(OS, OV, Value, Tape, Out,
                  T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
                  NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr, P: tl.constexpr = 'tf32x3'):
    block, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    m = tl.arange(0, BM)
    c = tl.arange(0, C)
    v = tile*BV+tl.arange(0, BV)
    idx = bh*NC+block
    state = tl.load(Tape+idx*M*V+m[:, None]*V+v[None, :], (m[:, None] < M) & (v[None, :] < V), 0)
    os = tl.load(OS+idx*C*M+c[:, None]*M+m[None, :], m[None, :] < M, 0).to(tl.float32)
    ov = tl.load(OV+idx*C*C+c[:, None]*C+c[None, :]).to(tl.float32)
    tok = ((bh//H)*T+block*C+c)*H+bh % H
    mask = (block*C+c[:, None] < T) & (v[None, :] < V)
    value = tl.load(Value+tok[:, None]*V+v[None, :], mask, 0).to(tl.float32)
    tl.store(Out+tok[:, None]*V+v[None, :], tl.dot(os, state, input_precision=P)+tl.dot(ov, value, input_precision=P), mask)


@tr.jit
def _coeffs(Hist, Proj, Read, Alpha, Beta, Slot, block, bh,
            T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr, BM: tl.constexpr):
    # Token-major coefficient rows of one chunk; pseudo rows interleave x/y components (2c, 2c+1).
    N: tl.constexpr = 2*C
    b, head = bh//H, bh % H
    c = tl.arange(0, C)
    n = tl.arange(0, N)
    m = tl.arange(0, BM)
    vm = m < M
    t = block*C+c
    valid = t < T
    tok = (b*T+t)*H+head
    nt = block*C+n//2
    nvalid = nt < T
    ntok = (b*T+nt)*H+head
    a = tl.load(Alpha+tok, valid, other=1).to(tl.float32)
    ap = tl.load(Alpha+tok-H, (c > 0) & (t-1 < T), other=1).to(tl.float32)
    beta = tl.load(Beta+tok, valid, other=0).to(tl.float32)
    slot = tl.load(Slot+tok, valid, other=0)
    nbeta = tl.load(Beta+ntok, nvalid, other=0).to(tl.float32)
    nslot = tl.load(Slot+ntok, nvalid, other=0)
    E = ((m[None, :] == slot[:, None]) & valid[:, None]).to(tl.float32)
    nE = ((m[None, :] == nslot[:, None]) & nvalid[:, None]).to(tl.float32)
    nmask = nvalid[:, None] & vm[None, :]
    hN = tl.load(Hist+ntok[:, None]*M+m[None, :], nmask, 0)
    pN = tl.load(Proj+ntok[:, None]*M+m[None, :], nmask, 0)
    XT = tl.where(n[:, None] % 2 == 0, pN-nE, -nbeta[:, None]*nE)
    YT = tl.where(n[:, None] % 2 == 0, nE, hN)
    RT = tl.load(Read+tok[:, None]*M+m[None, :], valid[:, None] & vm[None, :], 0)
    return a, ap, beta, E, XT, YT, RT, valid, tok


@tr.jit
def _decays(a, ap, C: tl.constexpr):
    # Division-free decay products over token windows, 0 outside their causal masks.
    c = tl.arange(0, C)
    ge = c[:, None] >= c[None, :]
    gt = c[:, None] > c[None, :]
    Di = tl.where(ge, tl.associative_scan(tl.where(ge, a[:, None], 1.), 0, _mul), 0.)       # [c',c]
    Dx = tl.where(ge, tl.associative_scan(tl.where(gt, a[:, None], 1.), 0, _mul), 0.)       # (c',c]
    De = tl.where(gt, tl.associative_scan(tl.where(c[:, None]-1 >= c[None, :], ap[:, None], 1.), 0, _mul), 0.)  # [c',c)
    Dxx = tl.where(gt, tl.associative_scan(tl.where(c[:, None]-1 > c[None, :], ap[:, None], 1.), 0, _mul), 0.)  # (c',c)
    pin = tl.associative_scan(a, 0, _mul)
    pex = tl.associative_scan(ap, 0, _mul)
    last = c[:, None] == C-1
    sin = tl.sum(tl.where(last, Di, 0.), 0)
    sex = tl.sum(tl.where(last, Dx, 0.), 0)
    return Di, Dx, De, Dxx, pin, pex, sin, sex


@tr.jit
def bwd_uz_kernel(Hist, Proj, Read, Value, Alpha, Beta, Slot, Tape, DS, DO, UZ,
                  T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr,
                  C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr):
    """Pseudo-values U = T (Bs S0 + Bv v) and adjoints Z = T^T (Xf^T dS + Ox^T dO) of one chunk."""
    block, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    N: tl.constexpr = 2*C
    n = tl.arange(0, N)
    m = tl.arange(0, BM)
    v = tile*BV+tl.arange(0, BV)
    vv = v < V
    a, ap, beta, E, XT, YT, RT, valid, tok = _coeffs(Hist, Proj, Read, Alpha, Beta, Slot, block, bh, T, H, M, C, BM)
    Di, Dx, De, Dxx, pin, pex, sin, sex = _decays(a, ap, C)
    cmask = valid[:, None] & vv[None, :]
    vC = tl.load(Value+tok[:, None]*V+v[None, :], cmask, 0).to(tl.float32)
    dO = tl.load(DO+tok[:, None]*V+v[None, :], cmask, 0).to(tl.float32)
    idx = bh*NC+block
    smask = (m[:, None] < M) & vv[None, :]
    S0 = tl.load(Tape+idx*M*V+m[:, None]*V+v[None, :], smask, 0)
    dSC = tl.load(DS+idx*M*V+m[:, None]*V+v[None, :], smask, 0)
    WT = beta[:, None]*E
    Tm = _inverse(_dot(YT, tl.trans(XT))*_cols2(_rows2(De, C, C), C, N), N)
    U = _dot(Tm, _dot(YT*_vec2(pex, C)[:, None], S0)+_dot(_dot(YT, tl.trans(WT))*_rows2(Dxx, C, C), vC))
    Ox = _dot(RT, tl.trans(XT))*_cols2(Di, C, C)
    Z = _dot(tl.trans(Tm), _vec2(sin, C)[:, None]*_dot(XT, dSC)+_dot(tl.trans(Ox), dO))
    out = UZ+idx*2*N*V+n[:, None]*V+v[None, :]
    tl.store(out, U, vv[None, :])
    tl.store(out+N*V, Z, vv[None, :])


@tr.jit
def bwd_products_kernel(Hist, Proj, Read, Value, Alpha, Beta, Slot, Tape, DS, DO, UZ, DV, Work,
                        T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr,
                        C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr, NW: tl.constexpr):
    """Contractions over the value axis, V-wide outputs dv, and per-tile partial vectors."""
    block, bh, tile = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    N: tl.constexpr = 2*C
    c = tl.arange(0, C)
    n = tl.arange(0, N)
    m = tl.arange(0, BM)
    v = tile*BV+tl.arange(0, BV)
    vv = v < V
    a, ap, beta, E, XT, YT, RT, valid, tok = _coeffs(Hist, Proj, Read, Alpha, Beta, Slot, block, bh, T, H, M, C, BM)
    Di, Dx, De, Dxx, pin, pex, sin, sex = _decays(a, ap, C)
    cmask = valid[:, None] & vv[None, :]
    vC = tl.load(Value+tok[:, None]*V+v[None, :], cmask, 0).to(tl.float32)
    dO = tl.load(DO+tok[:, None]*V+v[None, :], cmask, 0).to(tl.float32)
    idx = bh*NC+block
    smask = (m[:, None] < M) & vv[None, :]
    S0 = tl.load(Tape+idx*M*V+m[:, None]*V+v[None, :], smask, 0)
    dSC = tl.load(DS+idx*M*V+m[:, None]*V+v[None, :], smask, 0)
    src = UZ+idx*2*N*V+n[:, None]*V+v[None, :]
    U = tl.load(src, vv[None, :], 0)
    Z = tl.load(src+N*V, vv[None, :], 0)
    w = Work+((tile*tl.num_programs(1)+bh)*NC+block)*NW
    # Layout of one work record: see _WORK in the host code.
    ccol = c[:, None]*C+c[None, :]
    tl.store(w+0+c[:, None]*N+n[None, :], _dot(dO, tl.trans(U)))
    tl.store(w+C*N+ccol, _dot(dO, tl.trans(vC)))
    tl.store(w+C*N+C*C+c[:, None]*BM+m[None, :], _dot(dO, tl.trans(S0)))
    o1: tl.constexpr = C*N+C*C+C*BM
    tl.store(w+o1+n[:, None]*BM+m[None, :], _dot(U, tl.trans(dSC)))
    o2: tl.constexpr = o1+N*BM
    tl.store(w+o2+n[:, None]*N+n[None, :], _dot(Z, tl.trans(U)))
    o3: tl.constexpr = o2+N*N
    tl.store(w+o3+n[:, None]*C+c[None, :], _dot(Z, tl.trans(vC)))
    o4: tl.constexpr = o3+N*C
    tl.store(w+o4+n[:, None]*BM+m[None, :], _dot(Z, tl.trans(S0)))
    o5: tl.constexpr = o4+N*BM
    XdS = _dot(XT, dSC)
    tl.store(w+o5+n, tl.sum(XdS*U, 1))                                   # p0x
    tl.store(w+o5+N+n, tl.sum(_dot(YT, S0)*Z, 1))                        # py0
    o6: tl.constexpr = o5+2*N
    tl.store(w+o6+c, tl.sum(_dot(RT, S0)*dO, 1))                         # pr0
    EdS = _dot(E, dSC)
    tl.store(w+o6+C+c, tl.sum(beta[:, None]*EdS*vC, 1))                  # p0w
    DxT = tl.trans(Dx)
    B1T = _rows2(DxT, C, C)
    B2T = _cols2(_rows2(tl.trans(Dxx), C, C), C, N)
    Zraw = _vec2(sex, C)[:, None]*XdS+_dot(B1T*tl.trans(_dot(RT, tl.trans(XT))), dO)+_dot(B2T*_dot(XT, tl.trans(YT)), Z)
    tl.store(w+o6+2*C+c, tl.sum(tl.reshape(tl.sum(Zraw*U, 1), (C, 2)), 1))   # term2
    gcy = _cols2(tl.trans(Dxx), C, C)
    Gj = sex[:, None]*EdS+_dot(DxT*_dot(E, tl.trans(RT)), dO)+_dot(gcy*_dot(E, tl.trans(YT)), Z)
    U2 = _pick(U, C, 1)
    tl.store(w+o6+3*C+c, tl.sum(Gj*(vC-a[:, None]*U2), 1))               # dbeta
    tl.store(w+o6+4*C, tl.sum(tl.sum(dSC*S0, 1), 0))                     # p00
    tl.store(DV+tok[:, None]*V+v[None, :], beta[:, None]*Gj, cmask)


@tr.jit
def bwd_coeff_kernel(Hist, Proj, Read, Alpha, Beta, Slot, Work, DH, DP, DR, DA, DB,
                     T: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
                     C: tl.constexpr, NC: tl.constexpr, BM: tl.constexpr, NW: tl.constexpr, NV: tl.constexpr):
    """Coefficient gradients from the value-contracted products; no value-width tensors."""
    block, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    N: tl.constexpr = 2*C
    c = tl.arange(0, C)
    n = tl.arange(0, N)
    m = tl.arange(0, BM)
    a, ap, beta, E, XT, YT, RT, valid, tok = _coeffs(Hist, Proj, Read, Alpha, Beta, Slot, block, bh, T, H, M, C, BM)
    Di, Dx, De, Dxx, pin, pex, sin, sex = _decays(a, ap, C)
    WT = beta[:, None]*E
    o1: tl.constexpr = C*N+C*C+C*BM
    o2: tl.constexpr = o1+N*BM
    o3: tl.constexpr = o2+N*N
    o4: tl.constexpr = o3+N*C
    o5: tl.constexpr = o4+N*BM
    o6: tl.constexpr = o5+2*N
    dOU = tl.zeros((C, N), tl.float32)
    dOv = tl.zeros((C, C), tl.float32)
    dOS0 = tl.zeros((C, BM), tl.float32)
    UdS = tl.zeros((N, BM), tl.float32)
    ZU = tl.zeros((N, N), tl.float32)
    Zv = tl.zeros((N, C), tl.float32)
    ZS0 = tl.zeros((N, BM), tl.float32)
    p0x = tl.zeros((N,), tl.float32)
    py0 = tl.zeros((N,), tl.float32)
    pr0 = tl.zeros((C,), tl.float32)
    p0w = tl.zeros((C,), tl.float32)
    term2 = tl.zeros((C,), tl.float32)
    dB = tl.zeros((C,), tl.float32)
    p00 = 0.
    for tile in tl.static_range(NV):
        w = Work+((tile*tl.num_programs(1)+bh)*NC+block)*NW
        dOU += tl.load(w+c[:, None]*N+n[None, :])
        dOv += tl.load(w+C*N+c[:, None]*C+c[None, :])
        dOS0 += tl.load(w+C*N+C*C+c[:, None]*BM+m[None, :])
        UdS += tl.load(w+o1+n[:, None]*BM+m[None, :])
        ZU += tl.load(w+o2+n[:, None]*N+n[None, :])
        Zv += tl.load(w+o3+n[:, None]*C+c[None, :])
        ZS0 += tl.load(w+o4+n[:, None]*BM+m[None, :])
        p0x += tl.load(w+o5+n)
        py0 += tl.load(w+o5+N+n)
        pr0 += tl.load(w+o6+c)
        p0w += tl.load(w+o6+C+c)
        term2 += tl.load(w+o6+2*C+c)
        dB += tl.load(w+o6+3*C+c)
        p00 += tl.load(w+o6+4*C)
    DxT = tl.trans(Dx)
    Di2 = _cols2(Di, C, C)
    RX = _dot(RT, tl.trans(XT))
    YX = _dot(YT, tl.trans(XT))
    dR = pin[:, None]*dOS0+_dot(Di2*dOU, XT)+_dot(Dx*dOv, WT)
    B1T = _rows2(DxT, C, C)
    B2T = _cols2(_rows2(tl.trans(Dxx), C, C), C, N)
    dXT = _vec2(a, C)[:, None]*(_vec2(sex, C)[:, None]*UdS+_dot(B1T*tl.trans(dOU), RT)+_dot(B2T*tl.trans(ZU), YT))
    dP = _pick(dXT, C, 0)
    Z2S0 = _pick(ZS0, C, 1)
    Z2U = _pick(ZU, C, 1)
    Z2v = _pick(Zv, C, 1)
    scx = _cols2(De, C, C)
    dH = pex[:, None]*Z2S0+_dot(scx*Z2U, XT)+_dot(Dxx*Z2v, WT)
    gcy = _cols2(tl.trans(Dxx), C, C)
    prx = RX*dOU
    prw = _dot(RT, tl.trans(WT))*dOv
    pyx = YX*ZU
    pyw = _dot(YT, tl.trans(WT))*Zv
    t1 = sex*(p00*pex+tl.sum(scx*p0x[None, :], 1)+tl.sum(Dxx*p0w[None, :], 1))
    t1 += tl.sum(DxT*(pex[:, None]*pr0[None, :]+_dot(scx, tl.trans(prx))+_dot(Dxx, tl.trans(prw))), 1)
    t1 += tl.sum(gcy*(pex[:, None]*py0[None, :]+_dot(scx, tl.trans(pyx))+_dot(Dxx, tl.trans(pyw))), 1)
    omask = valid[:, None] & (m[None, :] < M)
    tl.store(DH+tok[:, None]*M+m[None, :], dH, omask)
    tl.store(DP+tok[:, None]*M+m[None, :], dP, omask)
    tl.store(DR+tok[:, None]*M+m[None, :], dR, omask)
    tl.store(DA+tok, t1+term2, valid)
    tl.store(DB+tok, dB, valid)


def _bv(width, c, bm):
    # Large C x M chunks exceed Hopper shared memory with a 64-wide value tile.
    cap = 32 if c*bm >= 32*64 else 64
    return max(16, min(cap, tr.next_power_of_2(width)))


class _Recurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, p, r, v, a, b, slots, initial, c, warps):
        batch, t, heads, m = h.shape
        width = v.shape[-1]
        nc = tr.cdiv(t, c)
        bm = max(16, tr.next_power_of_2(m))
        bv = 32
        os, ov, fs, fv = prepare(h, p, r, a, b, slots, c)
        out = torch.empty(v.shape, device=v.device, dtype=torch.float32)
        final = torch.empty(initial.shape, device=v.device, dtype=torch.float32)
        tape = torch.empty((batch, heads, nc, m, width), device=v.device, dtype=torch.float32)
        scan_state_kernel[(batch*heads, tr.cdiv(width, bv))](fs, fv, v, initial, final, tape,
                                                            t, heads, m, width, c, nc, bm, bv, num_warps=4)
        output_kernel[(nc, batch*heads, tr.cdiv(width, bv))](os, ov, v, tape, out, t, heads, m, width, c, nc, bm, bv, num_warps=4)
        ctx.save_for_backward(h, p, r, v, a, b, slots, tape, os, fs)
        ctx.cfg = (c, initial.dtype, warps)
        return out, final

    @staticmethod
    @once_differentiable
    def backward(ctx, do, df):
        h, p, r, v, a, b, slots, tape, os, fs = ctx.saved_tensors
        c, initial_dtype, warps = ctx.cfg
        batch, t, heads, m = h.shape
        width = v.shape[-1]
        nc = tr.cdiv(t, c)
        bm = max(16, tr.next_power_of_2(m))
        do, df = do.contiguous(), df.contiguous()
        ds = torch.empty_like(tape)
        di = torch.empty(df.shape, device=v.device, dtype=initial_dtype)
        boundary_bwd_kernel[(batch*heads, tr.cdiv(width, 32))](os, fs, do, df, ds, di, t, heads, m, width, c, nc, bm, 32, num_warps=4)
        bv = _bv(width, c, bm)
        nv = tr.cdiv(width, bv)
        n2 = 2*c
        nw = c*n2+c*c+c*bm+n2*bm+n2*n2+n2*c+n2*bm+2*n2+4*c+1
        uz = torch.empty((batch*heads, nc, 2, n2, width), device=v.device, dtype=torch.float32)
        work = torch.empty((nv, batch*heads, nc, nw), device=v.device, dtype=torch.float32)
        dv = torch.empty_like(v)
        outs = [torch.empty_like(x) for x in (h, p, r, a, b)]
        grid = (nc, batch*heads, nv)
        bwd_uz_kernel[grid](h, p, r, v, a, b, slots, tape, ds, do, uz, t, heads, m, width, c, nc, bm, bv, num_warps=4)
        bwd_products_kernel[grid](h, p, r, v, a, b, slots, tape, ds, do, uz, dv, work, t, heads, m, width, c, nc, bm, bv, nw, num_warps=warps)
        bwd_coeff_kernel[(nc, batch*heads)](h, p, r, a, b, slots, work, outs[0], outs[1], outs[2], outs[3], outs[4],
                                            t, heads, m, c, nc, bm, nw, nv, num_warps=4)
        return outs[0], outs[1], outs[2], dv, outs[3], outs[4], None, di, None, None


def triton_recurrence(h, p, r, v, alpha, beta, slots, initial_values, chunk_size=16, warps=4):
    """Scalar-decay recurrence, same contract as triton_chunk; per-slot alpha falls back to triton_chunk."""
    if alpha.ndim == 4:
        from .triton_chunk import triton_chunk
        return triton_chunk(h, p, r, v, alpha, beta, slots, initial_values, chunk_size)
    if alpha.shape != h.shape[:3] or beta.shape != h.shape[:3] or slots.shape != h.shape[:3]:
        raise ValueError('alpha, beta and slots must have shape [B,T,H]')
    if chunk_size not in (16, 32):
        raise ValueError('chunk_size must be 16 or 32')
    inputs = (h, p, r, v, alpha, beta, slots, initial_values)
    with torch.cuda.device(h.device):
        return _Recurrence.apply(*[x.contiguous() for x in inputs], chunk_size, warps)
