"""Fused recurrent rank-two slot updates with first-order Triton backward.

Layout and value-axis tiling follow FLA's fused recurrent IPLR/GDN operators.
The recurrence is specialized to replacement plus projection plus delta writing.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit(do_not_specialize=['T'])
def slot_fwd_kernel(Hist, Proj, Read, Value, Alpha, Beta, Slot, Init, Out, Final, Tape,
                    T, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr,
                    BM: tl.constexpr, BV: tl.constexpr, SAVE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    iv, bh = pid % nv, pid // nv
    batch, head = bh // H, bh % H
    im = tl.arange(0, BM).to(tl.int64)
    ivs = iv*BV + tl.arange(0, BV).to(tl.int64)
    mask = (im[:, None] < M) & (ivs[None, :] < V)
    state_offset = bh*M*V + im[:, None]*V + ivs[None, :]
    state = tl.load(Init+state_offset, mask, other=0)
    if SAVE:
        off = ((batch*(T+1)*H+head)*M+im[:, None])*V+ivs[None, :]
        tl.store(Tape+off, state, mask)
    for t in tl.range(0, T):
        token = (batch*T+t)*H+head
        h = tl.load(Hist+token*M+im, im < M, other=0)
        p = tl.load(Proj+token*M+im, im < M, other=0)
        r = tl.load(Read+token*M+im, im < M, other=0)
        v = tl.load(Value+token*V+ivs, ivs < V, other=0)
        a = tl.load(Alpha+token)
        b = tl.load(Beta+token)
        j = tl.load(Slot+token)
        e = (im == j).to(tl.float32)
        pre = a*state
        old = tl.sum(pre*e[:, None], axis=0)
        residual = v-tl.sum(pre*h[:, None], axis=0)
        state = pre+(p-e)[:, None]*old[None, :]+e[:, None]*(b*residual)[None, :]
        out = tl.sum(state*r[:, None], axis=0)
        tl.store(Out+token*V+ivs, out, ivs < V)
        if SAVE:
            off = ((batch*(T+1)*H+(t+1)*H+head)*M+im[:, None])*V+ivs[None, :]
            tl.store(Tape+off, state, mask)
    tl.store(Final+state_offset, state, mask)


@triton.jit(do_not_specialize=['T'])
def slot_bwd_kernel(Hist, Proj, Read, Value, Alpha, Beta, Slot, Tape, DOut, DFinal,
                    DH, DP, DR, DV, DA, DB, DInit,
                    T, B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, V: tl.constexpr,
                    BM: tl.constexpr, BV: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    iv, bh = pid % nv, pid // nv
    batch, head = bh // H, bh % H
    im = tl.arange(0, BM).to(tl.int64)
    ivs = iv*BV + tl.arange(0, BV).to(tl.int64)
    mask = (im[:, None] < M) & (ivs[None, :] < V)
    state_offset = bh*M*V+im[:, None]*V+ivs[None, :]
    adjoint = tl.load(DFinal+state_offset, mask, other=0)
    for step in tl.range(0, T):
        t = T-1-step
        token = (batch*T+t)*H+head
        h = tl.load(Hist+token*M+im, im < M, other=0)
        p = tl.load(Proj+token*M+im, im < M, other=0)
        r = tl.load(Read+token*M+im, im < M, other=0)
        v = tl.load(Value+token*V+ivs, ivs < V, other=0)
        a = tl.load(Alpha+token)
        b = tl.load(Beta+token)
        j = tl.load(Slot+token)
        e = (im == j).to(tl.float32)
        off = ((batch*(T+1)*H+t*H+head)*M+im[:, None])*V+ivs[None, :]
        previous = tl.load(Tape+off, mask, other=0)
        current = tl.load(Tape+off+H*M*V, mask, other=0)
        do = tl.load(DOut+token*V+ivs, ivs < V, other=0)
        dr = tl.sum(current*do[None, :], axis=1)
        adjoint += r[:, None]*do[None, :]
        pre = a*previous
        old = tl.sum(pre*e[:, None], axis=0)
        residual = v-tl.sum(pre*h[:, None], axis=0)
        selected_adjoint = tl.sum(adjoint*e[:, None], axis=0)
        dp = tl.sum(adjoint*old[None, :], axis=1)
        d_old = tl.sum(adjoint*(p-e)[:, None], axis=0)
        d_residual = b*selected_adjoint
        db = tl.sum(selected_adjoint*residual, axis=0)
        dh = -tl.sum(pre*d_residual[None, :], axis=1)
        d_pre = adjoint+e[:, None]*d_old[None, :]-h[:, None]*d_residual[None, :]
        da = tl.sum(tl.sum(d_pre*previous, axis=1), axis=0)
        adjoint = a*d_pre
        partial = iv*B*T*H+token
        tl.store(DH+partial*M+im, dh, im < M)
        tl.store(DP+partial*M+im, dp, im < M)
        tl.store(DR+partial*M+im, dr, im < M)
        tl.store(DA+partial, da)
        tl.store(DB+partial, db)
        tl.store(DV+token*V+ivs, d_residual, ivs < V)
    tl.store(DInit+state_offset, adjoint, mask)


class _SlotRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, p, r, v, alpha, beta, slots, initial_values, save):
        batch, t, heads, m = h.shape
        width = v.shape[-1]
        output = torch.empty_like(v)
        final = torch.empty_like(initial_values)
        tape = v.new_empty((batch, t+1, heads, m, width) if save else (1,))
        bv = 32
        with torch.cuda.device(v.device):
            slot_fwd_kernel[(batch*heads*triton.cdiv(width, bv),)](
                h, p, r, v, alpha, beta, slots, initial_values, output, final, tape,
                t, heads, m, width, triton.next_power_of_2(m), bv, save, num_warps=4,
            )
        ctx.save_for_backward(h, p, r, v, alpha, beta, slots, tape)
        ctx.set_materialize_grads(False)
        return output, final

    @staticmethod
    @once_differentiable
    def backward(ctx, do, dfinal):
        h, p, r, v, alpha, beta, slots, tape = ctx.saved_tensors
        batch, t, heads, m = h.shape
        width = v.shape[-1]
        bv = 32
        nv = triton.cdiv(width, bv)
        do = torch.zeros_like(v) if do is None else do.contiguous()
        state_shape = (batch, heads, m, width)
        dfinal = v.new_zeros(state_shape) if dfinal is None else dfinal.contiguous()
        dh, dp, dr = [h.new_empty((nv, *h.shape)) for _ in range(3)]
        da, db = [alpha.new_empty((nv, *alpha.shape)) for _ in range(2)]
        dv = torch.empty_like(v)
        dinitial = v.new_empty(state_shape)
        with torch.cuda.device(v.device):
            slot_bwd_kernel[(batch*heads*nv,)](
                h, p, r, v, alpha, beta, slots, tape, do, dfinal,
                dh, dp, dr, dv, da, db, dinitial,
                t, batch, heads, m, width, triton.next_power_of_2(m), bv, num_warps=4,
            )
        return dh.sum(0), dp.sum(0), dr.sum(0), dv, da.sum(0), db.sum(0), None, dinitial, None


def fused_recurrent_slot(h, p, r, v, alpha, beta, slots, initial_values):
    """FP32 dense CUDA recurrence; integer routes are fixed in backward.

    h,p,r: [B,T,H,M]; v: [B,T,H,V]; alpha,beta,slots: [B,T,H].
    Returns FP32 output [B,T,H,V] and final values [B,H,M,V].
    Training saves all states. Inference allocates only the final recurrent state.
    """
    if h.ndim != 4 or v.ndim != 4:
        raise ValueError('Expected h and v in [B,T,H,D] layout')
    batch, t, heads, m = h.shape
    width = v.shape[-1]
    if not all(d > 0 for d in (batch, t, heads, m, width)) or m > 64:
        raise ValueError('Positive dimensions and at most 64 slots are supported')
    if p.shape != h.shape or r.shape != h.shape or v.shape[:3] != h.shape[:3]:
        raise ValueError('Geometry and value batch/time/head shapes must agree')
    if any(x.shape != h.shape[:3] for x in (alpha, beta, slots)):
        raise ValueError('Gates and slots must have shape [B,T,H]')
    if initial_values.shape != (batch, heads, m, width):
        raise ValueError('initial_values must have shape [B,H,M,V]')
    tensors = (h, p, r, v, alpha, beta, initial_values)
    if any(x.dtype != torch.float32 or x.device != h.device for x in tensors) or h.device.type != 'cuda':
        raise ValueError('The Triton core requires FP32 tensors on the same CUDA device')
    if slots.dtype not in (torch.int32, torch.int64) or slots.device != h.device:
        raise ValueError('slots must contain CUDA int32/int64 indices on the same device')
    # no host synchronization: callers supplying geometry own the route range contract
    save = torch.is_grad_enabled() and any(x.requires_grad for x in tensors)
    return _SlotRecurrence.apply(*[x.contiguous() for x in (h, p, r, v, alpha, beta, slots, initial_values)], save)
