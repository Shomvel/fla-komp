"""Chunk-parallel gated BSGD merging: center trajectories, rank-1 maps and affine scans.

All numeric forward/backward stages are Triton. Cyclic gated policy only; the
coefficient-dependent grid policy remains in bsgd.py. First derivatives only.
"""
import math
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from .utils import zero
from .triton_chunk import _inverse, _mul, local_bwd_kernel, reduce_kernel
from .scan import affine_scan, value_offset_kernel, output_offset_kernel
from .recurrence import output_kernel


@tr.jit
def center_fwd(K, Merge, Initial, Pool, Final, T: tl.constexpr, H: tl.constexpr,
               M: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, OFFSET: tl.constexpr):
    slot, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    d = tl.arange(0, BD)
    u = tl.load(Initial+(bh*M+slot)*D+d, d < D, 0).to(tl.float32)
    tl.store(Pool+(bh*(T+M)+slot)*D+d, u, d < D)
    start = (slot+M-OFFSET) % M
    for t in range(start, T, M):
        tok = ((bh//H)*T+t)*H+bh%H
        k = tl.load(K+tok*D+d, d < D, 0).to(tl.float32)
        gamma = tl.load(Merge+tok).to(tl.float32)
        u = (1-gamma)*u+gamma*k
        tl.store(Pool+(bh*(T+M)+M+t)*D+d, u, d < D)
    tl.store(Final+(bh*M+slot)*D+d, u, d < D)


@tr.jit
def _source(t, slots, M: tl.constexpr, OFFSET: tl.constexpr):
    # Latest update of each physical slot at or before t (t may be -1).
    last = t-((OFFSET+t+M-slots) % M)
    return tl.where(last >= 0, M+last, slots)


@tr.jit
def geometry_fwd(Q, K, Beta, Pool, Hist, Proj, Read, Effective, Slots,
                 T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                 BM: tl.constexpr, BD: tl.constexpr, OFFSET: tl.constexpr, TAU: tl.constexpr):
    t, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    tok = ((bh//H)*T+t)*H+bh%H
    m, d = tl.arange(0, BM), tl.arange(0, BD)
    mask = (m[:, None] < M) & (d[None, :] < D)
    before, after = _source(t-1, m, M, OFFSET), _source(t, m, M, OFFSET)
    old = tl.load(Pool+(bh*(T+M)+before[:, None])*D+d[None, :], mask, 0)
    new = tl.load(Pool+(bh*(T+M)+after[:, None])*D+d[None, :], mask, 0)
    k = tl.load(K+tok*D+d, d < D, 0).to(tl.float32)
    q = tl.load(Q+tok*D+d, d < D, 0).to(tl.float32)
    j = (OFFSET+t) % M
    u = tl.sum(tl.where((m == j)[:, None], old, 0.), 0)
    z = tl.sum(tl.where((m == j)[:, None], new, 0.), 0)
    hist = tl.exp(-tl.sum((old-k[None, :])*(old-k[None, :]), 1)/(2*TAU))
    read = tl.exp(-tl.sum((new-q[None, :])*(new-q[None, :]), 1)/(2*TAU))
    a = tl.exp(-tl.sum((z-u)*(z-u), 0)/(2*TAU))
    s = tl.exp(-tl.sum((z-k)*(z-k), 0)/(2*TAU))
    tl.store(Hist+tok*M+m, hist, m < M)
    tl.store(Read+tok*M+m, read, m < M)
    tl.store(Proj+tok*M+m, tl.where(m == j, a, 0.), m < M)
    tl.store(Effective+tok, tl.load(Beta+tok).to(tl.float32)*s)
    tl.store(Slots+tok, j)


@tr.jit
def exp_gate(G, A, DA, DG, N: tl.constexpr, BACK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0, BLOCK)
    if BACK:
        y = tl.load(A+i, i < N, 0)*tl.load(DA+i, i < N, 0)
        tl.store(DG+i, y, i < N)
    else:
        tl.store(A+i, tl.exp(tl.load(G+i, i < N, 0).to(tl.float32)), i < N)


@tr.jit
def geometry_bwd(Q, K, Beta, Pool, Hist, Read, DH, DP, DR, DE,
                 DPool, DQ, DKDirect, DB,
                 T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
                 BM: tl.constexpr, BD: tl.constexpr, OFFSET: tl.constexpr, TAU: tl.constexpr):
    t, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    tok = ((bh//H)*T+t)*H+bh%H
    m, d = tl.arange(0, BM), tl.arange(0, BD)
    mask = (m[:, None] < M) & (d[None, :] < D)
    before, after = _source(t-1, m, M, OFFSET), _source(t, m, M, OFFSET)
    old = tl.load(Pool+(bh*(T+M)+before[:, None])*D+d[None, :], mask, 0)
    new = tl.load(Pool+(bh*(T+M)+after[:, None])*D+d[None, :], mask, 0)
    k = tl.load(K+tok*D+d, d < D, 0).to(tl.float32)
    q = tl.load(Q+tok*D+d, d < D, 0).to(tl.float32)
    j = (OFFSET+t) % M
    u = tl.sum(tl.where((m == j)[:, None], old, 0.), 0)
    z = tl.sum(tl.where((m == j)[:, None], new, 0.), 0)
    dh = tl.load(DH+tok*M+m, m < M, 0)*tl.load(Hist+tok*M+m, m < M, 0)
    dr = tl.load(DR+tok*M+m, m < M, 0)*tl.load(Read+tok*M+m, m < M, 0)
    dold = -dh[:, None]*(old-k[None, :])/TAU
    dnew = -dr[:, None]*(new-q[None, :])/TAU
    dq, dk = -tl.sum(dnew, 0), -tl.sum(dold, 0)
    a = tl.exp(-tl.sum((z-u)*(z-u), 0)/(2*TAU))
    s = tl.exp(-tl.sum((z-k)*(z-k), 0)/(2*TAU))
    da = tl.load(DP+tok*M+j)
    de = tl.load(DE+tok)
    ds = de*tl.load(Beta+tok).to(tl.float32)
    du = a*da*(z-u)/TAU
    dkk = s*ds*(z-k)/TAU
    dz = -du-dkk
    dold += tl.where((m == j)[:, None], du[None, :], 0.)
    dnew += tl.where((m == j)[:, None], dz[None, :], 0.)
    tl.atomic_add(DPool+(bh*(T+M)+before[:, None])*D+d[None, :], dold, mask, sem='relaxed')
    tl.atomic_add(DPool+(bh*(T+M)+after[:, None])*D+d[None, :], dnew, mask, sem='relaxed')
    tl.store(DQ+tok*D+d, dq, d < D)
    tl.store(DKDirect+tok*D+d, dk+dkk, d < D)
    tl.store(DB+tok, de*s)


@tr.jit
def center_bwd(K, Merge, Pool, DPool, DFinal, DKDirect, DK, DMerge, DInitial,
               T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, D: tl.constexpr,
               BD: tl.constexpr, OFFSET: tl.constexpr):
    slot, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    d = tl.arange(0, BD)
    carry = tl.load(DFinal+(bh*M+slot)*D+d, d < D, 0).to(tl.float32)
    start = (slot+M-OFFSET) % M
    n = tl.maximum(0, tl.cdiv(T-start, M))
    for step in range(n):
        t = start+(n-1-step)*M
        tok = ((bh//H)*T+t)*H+bh%H
        previous = tl.where(t >= M, t, slot)  # pool row M+(t-M), or initial slot
        old = tl.load(Pool+(bh*(T+M)+previous)*D+d, d < D, 0)
        k = tl.load(K+tok*D+d, d < D, 0).to(tl.float32)
        gamma = tl.load(Merge+tok).to(tl.float32)
        dz = carry+tl.load(DPool+(bh*(T+M)+M+t)*D+d, d < D, 0)
        dk = gamma*dz+tl.load(DKDirect+tok*D+d, d < D, 0)
        tl.store(DK+tok*D+d, dk, d < D)
        tl.store(DMerge+tok, tl.sum(dz*(k-old), 0))
        carry = (1-gamma)*dz
    carry += tl.load(DPool+(bh*(T+M)+slot)*D+d, d < D, 0)
    tl.store(DInitial+(bh*M+slot)*D+d, carry, d < D)


class _Geometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, g, beta, merge, initial, tau, offset):
        b, t, h, d = q.shape
        m = initial.shape[-2]
        alloc = lambda *shape: torch.empty(shape, device=q.device, dtype=torch.float32)
        pool, final = alloc(b, h, t+m, d), alloc(b, h, m, d)
        hist, proj, read = [alloc(b, t, h, m) for _ in range(3)]
        alpha, effective = alloc(*g.shape), alloc(b, t, h)
        slots = torch.empty((b, t, h), device=q.device, dtype=torch.int32)
        bd, bm = tr.next_power_of_2(d), tr.next_power_of_2(m)
        center_fwd[(m, b*h)](k, merge, initial, pool, final, t, h, m, d, bd, offset, num_warps=4)
        geometry_fwd[(t, b*h)](q, k, beta, pool, hist, proj, read, effective, slots,
                                t, h, m, d, bm, bd, offset, tau, num_warps=4)
        exp_gate[(tr.cdiv(g.numel(), 256),)](g, alpha, alpha, alpha, g.numel(), False, 256)
        ctx.save_for_backward(q, k, g, beta, merge, initial, pool, hist, read, alpha)
        ctx.cfg = (tau, offset)
        return hist, proj, read, alpha, effective, final, slots

    @staticmethod
    @once_differentiable
    def backward(ctx, dh, dp, dr, da, de, dfinal, dslots):
        q, k, g, beta, merge, initial, pool, hist, read, alpha = ctx.saved_tensors
        tau, offset = ctx.cfg
        b, t, h, d = q.shape
        m = initial.shape[-2]
        dq, dk, dg, db, dm, di = [torch.empty_like(x) for x in (q, k, g, beta, merge, initial)]
        direct = torch.empty(k.shape, device=k.device, dtype=torch.float32)
        dpool = zero(torch.empty_like(pool))
        geometry_bwd[(t, b*h)](q, k, beta, pool, hist, read, dh.contiguous(), dp.contiguous(), dr.contiguous(),
                                de.contiguous(), dpool, dq, direct, db,
                                t, h, m, d, tr.next_power_of_2(m), tr.next_power_of_2(d), offset, tau, num_warps=4)
        center_bwd[(m, b*h)](k, merge, pool, dpool, dfinal.contiguous(), direct, dk, dm, di,
                              t, h, m, d, tr.next_power_of_2(d), offset, num_warps=4)
        exp_gate[(tr.cdiv(g.numel(), 256),)](g, alpha, da.contiguous(), dg, g.numel(), True, 256)
        return dq, dk, dg, db, dm, di, None, None


@tr.jit
def rank1_prepare(Hist, Proj, Read, Alpha, Beta, OS, OV, FS, FV,
                  T: tl.constexpr, H: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
                  BM: tl.constexpr, NC: tl.constexpr, SLOT: tl.constexpr, OFFSET: tl.constexpr):
    block, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    c, m = tl.arange(0, C), tl.arange(0, BM)
    t = block*C+c
    tok = ((bh//H)*T+t)*H+bh%H
    valid = t < T
    slot = (t+OFFSET) % M
    mask = valid[:, None] & (m[None, :] < M)
    hist = tl.load(Hist+tok[:, None]*M+m[None, :], mask, 0)
    proj = tl.load(Proj+tok[:, None]*M+m[None, :], mask, 0)
    read = tl.load(Read+tok[:, None]*M+m[None, :], mask, 0)
    beta = tl.load(Beta+tok, valid, 0)
    e = ((slot[:, None] == m[None, :]) & valid[:, None]).to(tl.float32)
    y = proj-e-beta[:, None]*hist
    # The update is S_t = diag(alpha_t) S_{t-1} + e_t u_t.
    # u_t = y_t^T diag(alpha_t) S_{t-1} + beta_t v_t.
    if SLOT:
        alpha = tl.load(Alpha+tok[:, None]*M+m[None, :], mask, 1.)
        route_alpha = tl.load(Alpha+tok[:, None]*M+slot[None, :], valid[:, None], 1.)
    else:
        a = tl.load(Alpha+tok, valid, 1.)
        alpha = tl.broadcast_to(a[:, None], (C, BM))
        route_alpha = tl.broadcast_to(a[:, None], (C, C))
    prefix = tl.associative_scan(alpha, 0, _mul)
    gt, ge = c[:, None] > c[None, :], c[:, None] >= c[None, :]
    windows = tl.associative_scan(tl.where(gt, route_alpha, 1.), 0, _mul)
    routed_y = tl.gather(y, tl.broadcast_to(slot[None, :], (C, C)), 1)
    routed_r = tl.gather(read, tl.broadcast_to(slot[None, :], (C, C)), 1)
    lower = tl.where(gt, routed_y*windows, 0.)
    inv = _inverse(lower, C)
    us = tl.dot(inv, prefix*y, input_precision='tf32x3')
    uv = inv*beta[None, :]
    response = tl.where(ge & valid[None, :], routed_r*windows, 0.)
    os = prefix*read+tl.dot(response, us, input_precision='tf32x3')
    ov = tl.where(ge, tl.dot(response, uv, input_precision='tf32x3'), 0.)
    end = tl.sum(tl.where((c == C-1)[:, None], windows, 0.), 0)
    endx = tl.trans(e*end[:, None])
    pend = tl.sum(tl.where((c == C-1)[:, None], prefix, 0.), 0)
    fs = pend[:, None]*(m[:, None] == m[None, :])+tl.dot(endx, us, input_precision='tf32x3')
    fv = tl.dot(endx, uv, input_precision='tf32x3')
    idx = bh*NC+block
    tl.store(OS+idx*C*M+c[:, None]*M+m[None, :], os, m[None, :] < M)
    tl.store(OV+idx*C*C+c[:, None]*C+c[None, :], ov)
    tl.store(FS+idx*M*M+m[:, None]*M+m[None, :], fs, (m[:, None] < M) & (m[None, :] < M))
    tl.store(FV+idx*M*C+m[:, None]*C+c[None, :], fv, m[:, None] < M)


class _Recurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hist, proj, read, v, alpha, beta, slots, initial, c, offset):
        b, t, h, m = hist.shape
        w, nc, bm = v.shape[-1], tr.cdiv(t, c), max(16, tr.next_power_of_2(m))
        alloc = lambda *shape: torch.empty(shape, device=v.device, dtype=torch.float32)
        os, ov, fs, fv = alloc(b,h,nc,c,m), alloc(b,h,nc,c,c), alloc(b,h,nc,m,m), alloc(b,h,nc,m,c)
        rank1_prepare[(nc,b*h)](hist,proj,read,alpha,beta,os,ov,fs,fv,t,h,m,c,bm,nc,alpha.ndim==4,offset,num_warps=4)
        out, final = torch.empty_like(v), alloc(b,h,m,w)
        tape, off = alloc(b,h,nc,m,w), alloc(b,h,nc,m,w)
        nv = tr.cdiv(w,32)
        value_offset_kernel[(nc,b*h,nv)](fv,v,off,t,h,m,w,c,nc,bm,32,num_warps=4)
        affine_scan(fs,off,initial,tape,final,False,None,32)
        output_kernel[(nc,b*h,nv)](os,ov,v,tape,out,t,h,m,w,c,nc,bm,32,num_warps=4)
        ctx.save_for_backward(hist,proj,read,v,alpha,beta,slots,tape,os,fs)
        ctx.cfg = c, initial.dtype
        return out,final

    @staticmethod
    @once_differentiable
    def backward(ctx, do, df):
        hist,proj,read,v,alpha,beta,slots,tape,os,fs = ctx.saved_tensors
        c, initial_dtype = ctx.cfg
        b,t,h,m = hist.shape
        w,nc,bm = v.shape[-1],tr.cdiv(t,c),max(16,tr.next_power_of_2(m))
        do,df = do.contiguous(),df.contiguous()
        ds,off = torch.empty_like(tape),torch.empty_like(tape)
        di = torch.empty(df.shape,device=v.device,dtype=initial_dtype)
        output_offset_kernel[(nc,b*h,tr.cdiv(w,32))](os,do,off,t,h,m,w,c,nc,bm,32,num_warps=4)
        affine_scan(fs,off,df,ds,di,True,None,32)
        work = torch.empty((b,h,nc,c,m,w),device=v.device,dtype=torch.float32)
        nv,bv = tr.cdiv(w,64),64
        results = [torch.empty_like(x) for x in (hist,proj,read,alpha,beta)]
        partial = results if nv==1 else [torch.empty((nv,*x.shape),device=v.device,dtype=torch.float32) for x in (hist,proj,read,alpha,beta)]
        dv = torch.empty_like(v)
        local_bwd_kernel[(nc,b*h,nv)](hist,proj,read,v,alpha,beta,slots,tape,ds,do,work,*partial[:3],dv,*partial[3:],
                                      b,t,h,m,w,c,nc,bm,bv,alpha.ndim==4,num_warps=1)
        if nv > 1:
            for x,y in zip(partial,results):
                reduce_kernel[(tr.cdiv(y.numel(),256),)](x,y,y.numel(),nv,tr.next_power_of_2(nv),256)
        return *results[:3],dv,*results[3:],None,di,None,None


def chunk_bsgd_merging(q,k,v,g,beta,merge,initial_state,*,temperature=.5,chunk_size=32,offset=0):
    """Exact gated/cyclic merging semantics; chunk_size changes computation only.

    Input/state contracts match bsgd_merging(mode='gated'). Scalar or per-slot log
    decay, arbitrary token cuts with offset, FP32/BF16, M<=64, D/V<=128. This path
    does not implement value-dependent grid search. See BSGD_CHUNK.md.
    """
    keys,values = initial_state
    if q.ndim!=4 or k.shape!=q.shape or v.ndim!=4 or v.shape[:3]!=q.shape[:3]:
        raise ValueError('q/k/v must have matching [B,T,H] axes')
    b,t,h,d=q.shape
    if keys.ndim!=4 or values.ndim!=4:
        raise ValueError('state must have four axes')
    m,w=keys.shape[-2],v.shape[-1]
    if keys.shape!=(b,h,m,d) or values.shape!=(b,h,m,w):
        raise ValueError('state shape mismatch')
    if beta.shape!=(b,t,h) or merge.shape!=beta.shape or g.shape not in ((b,t,h),(b,t,h,m)):
        raise ValueError('gate shape mismatch')
    if min(b,t,h,d,m,w)<1 or m>64 or max(d,w)>128:
        raise ValueError('nonempty tensors, M<=64, D/V<=128 required')
    if chunk_size not in (16,32,64) or not isinstance(offset,int) or offset<0:
        raise ValueError('chunk_size must be 16/32/64 and offset a nonnegative integer')
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('temperature must be positive and finite')
    xs=(q,k,v,g,beta,merge,keys,values)
    if q.device.type!='cuda' or any(x.device!=q.device or x.dtype not in (torch.float32,torch.bfloat16) for x in xs):
        raise ValueError('all tensors must share a CUDA device and use FP32/BF16')
    q,k,v,g,beta,merge,keys,values=[x.contiguous() for x in xs]
    with torch.cuda.device(q.device):
        hist,proj,read,alpha,effective,final_keys,slots=_Geometry.apply(q,k,g,beta,merge,keys,temperature,offset%m)
        out,final_values=_Recurrence.apply(hist,proj,read,v,alpha,effective,slots,values,chunk_size,offset%m)
    return out,(final_keys,final_values)
