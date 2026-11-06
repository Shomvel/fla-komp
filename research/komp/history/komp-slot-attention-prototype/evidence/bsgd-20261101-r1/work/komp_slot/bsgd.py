"""All-Triton forward and first backward for local BSGD-inspired merging.

One CTA owns one batch/head and scans tokens. No Gram solve. FP32 state and tape;
q/k/v may be FP32 or BF16. This correctness prototype stores O(B*T*H*M*(D+V))
backward tape and is not a parallel chunk implementation. See BSGD_MERGING.md.
"""
import math
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable


@tr.jit
def _forward(Q, K, V, G, Beta, Merge, K0, S0, O, KF, SF, KTape, ATape, Choice, Gamma,
             T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
             BM: tl.constexpr, BD: tl.constexpr, BW: tl.constexpr, TAU: tl.constexpr,
             SLOT: tl.constexpr, GRID: tl.constexpr, SEARCH: tl.constexpr, NGRID: tl.constexpr,
             OFFSET: tl.constexpr, SAVE: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    m, d, w = tl.arange(0, BM), tl.arange(0, BD), tl.arange(0, BW)
    kmask = (m[:, None] < M) & (d[None, :] < D)
    smask = (m[:, None] < M) & (w[None, :] < W)
    keys = tl.load(K0+bh*M*D+m[:, None]*D+d[None, :], kmask, 0).to(tl.float32)
    state = tl.load(S0+bh*M*W+m[:, None]*W+w[None, :], smask, 0).to(tl.float32)
    for t in range(T):
        tok = ((bh//H)*T+t)*H+bh%H
        kt = tl.load(K+tok*D+d, d < D, 0).to(tl.float32)
        qt = tl.load(Q+tok*D+d, d < D, 0).to(tl.float32)
        vt = tl.load(V+tok*W+w, w < W, 0).to(tl.float32)
        bt = tl.load(Beta+tok).to(tl.float32)
        if SLOT:
            decay = tl.exp(tl.load(G+tok*M+m, m < M, 0).to(tl.float32))
        else:
            decay = tl.full((BM,), 1., tl.float32)*tl.exp(tl.load(G+tok).to(tl.float32))
        old = state*decay[:, None]
        dist = tl.sum((keys-kt[None, :])*(keys-kt[None, :]), 1)
        h = tl.where(m < M, tl.exp(-dist/(2*TAU)), 0.)
        residual = vt-tl.sum(h[:, None]*old, 0)
        write = bt*residual
        j = (t+OFFSET) % M
        gamma = tl.load(Merge+tok).to(tl.float32)
        if GRID:
            aa = tl.sum(old*old, 1)
            bb = tl.sum(write*write, 0)
            ab = tl.sum(old*write[None, :], 1)
            energy = aa+bb+2*h*ab
            best = tl.full((BM,), float('inf'), tl.float32)
            best_i = tl.full((BM,), 0, tl.int32)
            for gi in range(NGRID):
                y = gi/(NGRID-1.)
                a = tl.exp(-y*y*dist/(2*TAU))
                s = tl.exp(-(1-y)*(1-y)*dist/(2*TAU))
                error = energy-(a*a*aa+s*s*bb+2*a*s*ab)
                improve = error < best
                best_i = tl.where(improve, gi, best_i)
                best = tl.minimum(best, error)
            if SEARCH:
                best = tl.where(m < M, best, float('inf'))
                lowest = tl.min(best, 0)
                j = tl.min(tl.where(best == lowest, m, 2147483647), 0)
            gamma = tl.sum(tl.where(m == j, best_i, 0), 0)/(NGRID-1.)
        sel = m == j
        u = tl.sum(tl.where(sel[:, None], keys, 0.), 0)
        cj = tl.sum(tl.where(sel[:, None], old, 0.), 0)
        z = (1-gamma)*u+gamma*kt
        a = tl.exp(-tl.sum((z-u)*(z-u), 0)/(2*TAU))
        s = tl.exp(-tl.sum((z-kt)*(z-kt), 0)/(2*TAU))
        if SAVE:
            ix = bh*T+t
            tl.store(KTape+ix*M*D+m[:, None]*D+d[None, :], keys, kmask)
            tl.store(ATape+ix*M*W+m[:, None]*W+w[None, :], old, smask)
            tl.store(Choice+ix, j)
            tl.store(Gamma+ix, gamma)
        keys = tl.where(sel[:, None], z[None, :], keys)
        state = tl.where(sel[:, None], (a*cj+s*write)[None, :], old)
        read = tl.where(m < M, tl.exp(-tl.sum((keys-qt[None, :])*(keys-qt[None, :]), 1)/(2*TAU)), 0.)
        out = tl.sum(read[:, None]*state, 0)
        tl.store(O+tok*W+w, out, w < W)
    tl.store(KF+bh*M*D+m[:, None]*D+d[None, :], keys, kmask)
    tl.store(SF+bh*M*W+m[:, None]*W+w[None, :], state, smask)


@tr.jit
def _backward(Q, K, V, G, Beta, KTape, ATape, Choice, Gamma, DO, DKF, DSF,
              DQ, DK, DV, DG, DB, DM, DK0, DS0,
              T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
              BM: tl.constexpr, BD: tl.constexpr, BW: tl.constexpr, TAU: tl.constexpr,
              SLOT: tl.constexpr, GRID: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    m, d, w = tl.arange(0, BM), tl.arange(0, BD), tl.arange(0, BW)
    kmask = (m[:, None] < M) & (d[None, :] < D)
    smask = (m[:, None] < M) & (w[None, :] < W)
    dkeys = tl.load(DKF+bh*M*D+m[:, None]*D+d[None, :], kmask, 0).to(tl.float32)
    dstate = tl.load(DSF+bh*M*W+m[:, None]*W+w[None, :], smask, 0).to(tl.float32)
    for rev in range(T):
        t = T-1-rev
        ix = bh*T+t
        tok = ((bh//H)*T+t)*H+bh%H
        keys = tl.load(KTape+ix*M*D+m[:, None]*D+d[None, :], kmask, 0)
        old = tl.load(ATape+ix*M*W+m[:, None]*W+w[None, :], smask, 0)
        j, gamma = tl.load(Choice+ix), tl.load(Gamma+ix)
        sel = m == j
        kt = tl.load(K+tok*D+d, d < D, 0).to(tl.float32)
        qt = tl.load(Q+tok*D+d, d < D, 0).to(tl.float32)
        vt = tl.load(V+tok*W+w, w < W, 0).to(tl.float32)
        bt = tl.load(Beta+tok).to(tl.float32)
        do = tl.load(DO+tok*W+w, w < W, 0).to(tl.float32)
        h = tl.where(m < M, tl.exp(-tl.sum((keys-kt[None, :])*(keys-kt[None, :]), 1)/(2*TAU)), 0.)
        residual = vt-tl.sum(h[:, None]*old, 0)
        write = bt*residual
        u = tl.sum(tl.where(sel[:, None], keys, 0.), 0)
        cj = tl.sum(tl.where(sel[:, None], old, 0.), 0)
        z = (1-gamma)*u+gamma*kt
        a = tl.exp(-tl.sum((z-u)*(z-u), 0)/(2*TAU))
        s = tl.exp(-tl.sum((z-kt)*(z-kt), 0)/(2*TAU))
        newkeys = tl.where(sel[:, None], z[None, :], keys)
        newstate = tl.where(sel[:, None], (a*cj+s*write)[None, :], old)
        read = tl.where(m < M, tl.exp(-tl.sum((newkeys-qt[None, :])*(newkeys-qt[None, :]), 1)/(2*TAU)), 0.)
        dr = tl.sum(newstate*do[None, :], 1)
        read_dkeys = -read[:, None]*dr[:, None]*(newkeys-qt[None, :])/TAU
        dq = -tl.sum(read_dkeys, 0)
        dkeys += read_dkeys
        dstate += read[:, None]*do[None, :]
        dc = tl.sum(tl.where(sel[:, None], dstate, 0.), 0)
        dz = tl.sum(tl.where(sel[:, None], dkeys, 0.), 0)
        da, ds = tl.sum(dc*cj, 0), tl.sum(dc*write, 0)
        dz += -a*da*(z-u)/TAU-s*ds*(z-kt)/TAU
        du = a*da*(z-u)/TAU+(1-gamma)*dz
        dk = s*ds*(z-kt)/TAU+gamma*dz
        dgamma = tl.sum(dz*(kt-u), 0)
        db = s*dc
        dbeta = tl.sum(db*residual, 0)
        de = bt*db
        dh = -tl.sum(old*de[None, :], 1)
        pred_dkeys = -h[:, None]*dh[:, None]*(keys-kt[None, :])/TAU
        dk -= tl.sum(pred_dkeys, 0)
        dkeys = tl.where(sel[:, None], du[None, :], dkeys)+pred_dkeys
        dold = tl.where(sel[:, None], a*dc[None, :], dstate)-h[:, None]*de[None, :]
        if SLOT:
            decay = tl.exp(tl.load(G+tok*M+m, m < M, 0).to(tl.float32))
            tl.store(DG+tok*M+m, tl.sum(dold*old, 1), m < M)
        else:
            decay = tl.full((BM,), 1., tl.float32)*tl.exp(tl.load(G+tok).to(tl.float32))
            tl.store(DG+tok, tl.sum(tl.sum(dold*old, 1), 0))
        dstate = dold*decay[:, None]
        tl.store(DQ+tok*D+d, dq, d < D)
        tl.store(DK+tok*D+d, dk, d < D)
        tl.store(DV+tok*W+w, de, w < W)
        tl.store(DB+tok, dbeta)
        tl.store(DM+tok, 0. if GRID else dgamma)
    tl.store(DK0+bh*M*D+m[:, None]*D+d[None, :], dkeys, kmask)
    tl.store(DS0+bh*M*W+m[:, None]*W+w[None, :], dstate, smask)


class _Merging(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, merge, keys, values, temperature, mode, routing, grid_size, offset, save):
        b, t, h, d = q.shape
        m, w = keys.shape[-2], v.shape[-1]
        f32 = lambda *s: torch.empty(s, device=q.device, dtype=torch.float32)
        kt, at = (f32(b*h, t, m, d), f32(b*h, t, m, w)) if save else (keys, values)
        choice = torch.empty((b*h, t) if save else (1,), device=q.device, dtype=torch.int32)
        gamma = f32(b*h, t) if save else merge
        out, kf, sf = torch.empty_like(v), f32(b, h, m, d), f32(b, h, m, w)
        cfg = (t, h, m, d, w, tr.next_power_of_2(m), tr.next_power_of_2(d), tr.next_power_of_2(w), temperature, g.ndim == 4, mode == 'grid')
        _forward[(b*h,)](q, k, v, g, beta, merge, keys, values, out, kf, sf, kt, at, choice, gamma,
                          *cfg, routing == 'min_error', grid_size, offset, save, num_warps=4)
        ctx.save_for_backward(q, k, v, g, beta, merge, keys, values, kt, at, choice, gamma)
        ctx.cfg = cfg
        return out, kf, sf

    @staticmethod
    @once_differentiable
    def backward(ctx, do, dkf, dsf):
        q, k, v, g, beta, merge, keys, values, kt, at, choice, gamma = ctx.saved_tensors
        grads = [torch.empty_like(x) for x in (q, k, v, g, beta, merge, keys, values)]
        _backward[(q.shape[0]*q.shape[2],)](q, k, v, g, beta, kt, at, choice, gamma,
                                          do.contiguous(), dkf.contiguous(), dsf.contiguous(), *grads,
                                          *ctx.cfg, num_warps=4)
        return *grads, None, None, None, None, None, None


def bsgd_merging(q, k, v, g, beta, merge, initial_state, *, temperature=.5,
                 mode='gated', routing='cyclic', grid_size=17, offset=0):
    """Return (output, (centers, coefficients)); Gaussian exp(-||x-y||²/(2 temperature)).

    q,k [B,T,H,D], v [B,T,H,V], beta/merge [B,T,H], g log decay
    [B,T,H] or [B,T,H,M]; state [B,H,M,D], [B,H,M,V]. Gates are
    caller-produced (beta/merge in [0,1], g<=0). Keys are NOT normalized internally.
    Grid mode ignores merge and differentiates through the chosen discrete branch.
    For cyclic streaming pass offset equal to the number of preceding tokens.
    First derivatives only. Supports FP32/BF16 inputs and up to 64 slots, D,V<=128.
    """
    keys, values = initial_state
    if q.ndim != 4 or q.shape != k.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError('q/k/v must have matching [B,T,H] axes and equal q/k widths')
    b, t, h, d = q.shape
    if keys.ndim != 4 or values.ndim != 4:
        raise ValueError('initial states must be [B,H,M,D] and [B,H,M,V]')
    m, w = keys.shape[-2], v.shape[-1]
    if keys.shape != (b, h, m, d) or values.shape != (b, h, m, w):
        raise ValueError('state shape mismatch')
    if beta.shape != (b, t, h) or merge.shape != beta.shape or g.shape not in ((b, t, h), (b, t, h, m)):
        raise ValueError('gate shape mismatch')
    if min(b, t, h, m, d, w) < 1 or m > 64 or max(d, w) > 128:
        raise ValueError('nonempty tensors, M<=64 and D,V<=128 required')
    if mode not in ('gated', 'grid') or routing not in ('cyclic', 'min_error') or (mode == 'gated' and routing != 'cyclic'):
        raise ValueError('gated requires cyclic routing; grid supports cyclic/min_error')
    if not isinstance(grid_size, int) or not 2 <= grid_size <= 65 or not isinstance(offset, int) or offset < 0:
        raise ValueError('grid_size must be 2..65 and offset a nonnegative integer')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    xs = (q, k, v, g, beta, merge, keys, values)
    if q.device.type != 'cuda' or any(x.device != q.device or x.dtype not in (torch.float32, torch.bfloat16) for x in xs):
        raise ValueError('all tensors must share a CUDA device and use FP32/BF16')
    save = torch.is_grad_enabled() and any(x.requires_grad for x in xs)
    with torch.cuda.device(q.device):
        out, kf, sf = _Merging.apply(*[x.contiguous() for x in xs], temperature, mode, routing, grid_size, offset % m, save)
    return out, (kf, sf)
