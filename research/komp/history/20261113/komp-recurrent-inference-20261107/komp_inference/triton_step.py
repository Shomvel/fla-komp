"""Single-token fused decay, delta write and read with FP32 state."""

from dataclasses import replace

import torch
import triton
import triton.language as tl

from .recurrent import compressed_state


@triton.jit(do_not_specialize=['N'])
def _step_kernel(Q, K, V, G, Beta, Keys, Coef, Slots, NewCoef, Out,
                 N, D: tl.constexpr, DV: tl.constexpr, M: tl.constexpr,
                 BN: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
                 AXIS: tl.constexpr, TAU: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    n = tl.arange(0, BN).to(tl.int64)
    d = tl.arange(0, BD).to(tl.int64)
    v = tl.arange(0, BV).to(tl.int64)
    old = tl.load(Keys + (bh*N+n[:, None])*D+d[None, :], (n[:, None] < N) & (d[None, :] < D), 0)
    q = tl.load(Q+bh*D+d, d < D, 0)
    k = tl.load(K+bh*D+d, d < D, 0)
    h = tl.exp((tl.sum(old*k[None, :], 1)-1)/TAU)
    r = tl.exp((tl.sum(old*q[None, :], 1)-1)/TAU)
    new_r = tl.exp((tl.sum(q*k, 0)-1)/TAU)
    coefficients = tl.load(Coef+(bh*N+n[:, None])*DV+v[None, :],
                           (n[:, None] < N) & (v[None, :] < DV), 0)
    if AXIS == 1:
        decay = tl.exp(tl.load(G+bh*DV+v, v < DV, 0))[None, :]
    elif AXIS == 0:
        slot = tl.load(Slots+bh*N+n, n < N, 0)
        decay = tl.exp(tl.load(G+bh*M+slot, n < N, 0))[:, None]
    else:
        decay = tl.exp(tl.load(G+bh))
    coefficients = coefficients*decay
    prediction = tl.sum(coefficients*h[:, None], 0)
    value = tl.load(V+bh*DV+v, v < DV, 0)
    write = tl.load(Beta+bh)*(value-prediction)
    output = tl.sum(coefficients*r[:, None], 0)+write*new_r
    tl.store(NewCoef+(bh*(N+1)+n[:, None])*DV+v[None, :], coefficients,
             (n[:, None] < N) & (v[None, :] < DV))
    tl.store(NewCoef+(bh*(N+1)+N)*DV+v, write, v < DV)
    tl.store(Out+bh*DV+v, output, v < DV)


def step(q, k, v, g, beta, score, state):
    b, h, n, d = state.keys.shape
    dv = v.shape[-1]
    coefficients = torch.empty(b, h, n+1, dv, device=v.device, dtype=torch.float32)
    output = torch.empty_like(v)
    _step_kernel[(b*h,)](
        q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(),
        state.keys.contiguous(), state.coefficients.contiguous(), state.slots.contiguous(), coefficients, output,
        n, d, dv, state.num_slots, triton.next_power_of_2(state.num_slots+state.chunk_size),
        triton.next_power_of_2(d), triton.next_power_of_2(dv),
        {'slot': 0, 'value': 1, 'scalar': 2}[state.decay_axis], state.temperature,
        num_warps=4, enable_fp_fusion=False,
    )
    birth = state.births.new_full((b, h, 1), state.position)
    state = replace(state, keys=torch.cat((state.keys, k[..., None, :]), -2), coefficients=coefficients,
                    scores=torch.cat((state.scores, score[..., None]), -1),
                    births=torch.cat((state.births, birth), -1),
                    slots=torch.cat((state.slots, birth % state.num_slots), -1), position=state.position+1)
    if state.position % state.chunk_size == 0:
        state = compressed_state(state)
    return output, state
