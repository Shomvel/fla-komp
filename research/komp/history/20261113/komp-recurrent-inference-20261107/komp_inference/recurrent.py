"""Forward-only KOMP streaming state; compression follows global token boundaries."""

from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class KOMPState:
    keys: torch.Tensor
    coefficients: torch.Tensor
    scores: torch.Tensor
    births: torch.Tensor
    slots: torch.Tensor
    position: int
    num_slots: int
    chunk_size: int
    decay_axis: str
    eviction: str
    temperature: float = .5
    ridge: float = 1e-4

    @property
    def pending(self):
        return self.keys.shape[-2] - self.num_slots

    def clone(self):
        return replace(self, **{name: getattr(self, name).clone() for name in
                               ('keys', 'coefficients', 'scores', 'births', 'slots')})

    def index_select(self, indices):
        """Select/reorder independent sequences, including beam duplication."""
        return replace(self, **{name: getattr(self, name).index_select(0, indices) for name in
                               ('keys', 'coefficients', 'scores', 'births', 'slots')})

    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in
                   (self.keys, self.coefficients, self.scores, self.births, self.slots))


def initialize_state(keys, coefficients, *, chunk_size=128, decay_axis='slot', eviction='fifo',
                     initial_scores=None, temperature=.5, ridge=1e-4):
    if keys.ndim != 4 or coefficients.shape[:3] != keys.shape[:3]:
        raise ValueError('keys/coefficients require [B,H,M,K/V]')
    if chunk_size not in (1, 16, 32, 64, 128) or decay_axis not in ('slot', 'value', 'scalar'):
        raise ValueError('unsupported chunk size or decay axis')
    if eviction not in ('fifo', 'learned') or temperature <= 0 or ridge <= 0:
        raise ValueError('unsupported eviction or nonpositive temperature/ridge')
    dtype = torch.float64 if keys.dtype == coefficients.dtype == torch.float64 else torch.float32
    b, h, m, _ = keys.shape
    if min(keys.shape) < 1 or m > 64 or coefficients.shape[-1] < 1:
        raise ValueError('positive dimensions and M <= 64 required')
    if initial_scores is None:
        if eviction == 'learned':
            raise ValueError('learned eviction requires initial scores')
        initial_scores = keys.new_zeros(b, h, m)
    if initial_scores.shape != (b, h, m):
        raise ValueError('initial scores require [B,H,M]')
    slots = torch.arange(m, device=keys.device).expand(b, h, m).clone()
    return KOMPState(F.normalize(keys.to(dtype), dim=-1, eps=1e-6), coefficients.to(dtype).clone(),
                     initial_scores.to(dtype).clone(), slots - m, slots, 0, m, chunk_size,
                     decay_axis, eviction, temperature, ridge)


def _gather(x, indices):
    return x.gather(-2, indices[..., None].expand(*indices.shape, x.shape[-1]))


def _kernel(x, y, temperature):
    return ((x @ y.transpose(-1, -2) - 1) / temperature).exp()


def _physical_order(keep, slots, m):
    """Retained physical atoms stay put; temporary atoms claim their free positional slot."""
    temporary = keep >= m
    zeros = torch.zeros_like(keep)
    free = zeros.scatter_add(-1, keep.clamp(max=m-1), (~temporary).long()) == 0
    positions = slots.gather(-1, keep)
    earlier = (keep[..., :, None] > keep[..., None, :]) & temporary[..., None, :]
    earlier &= positions[..., :, None] == positions[..., None, :]
    claim = temporary & free.gather(-1, positions) & ~earlier.any(-1)
    claimed = zeros.scatter_add(-1, positions, claim.long()) > 0
    remaining = free & ~claimed
    rank = remaining.long().cumsum(-1) - 1
    by_rank = torch.zeros(*keep.shape[:-1], m+1, device=keep.device, dtype=torch.long)
    by_rank.scatter_(-1, torch.where(remaining, rank, m), torch.arange(m, device=keep.device).expand_as(keep))
    needed_rank = (temporary & ~claim).long().cumsum(-1) - 1
    assigned = by_rank.gather(-1, needed_rank.clamp(min=0))
    destination = torch.where(~temporary, keep, torch.where(claim, positions, assigned))
    return destination.argsort(dim=-1)


def compressed_state(state):
    """Return a terminal compressed view without changing the resumable partial-chunk cache."""
    if not state.pending:
        return state
    m = state.num_slots
    recent = state.births.argsort(dim=-1, descending=True, stable=True)
    rank = state.births if state.eviction == 'fifo' else state.scores
    order = recent.gather(-1, rank.gather(-1, recent).argsort(dim=-1, descending=True, stable=True))
    keep = order[..., :m].sort(dim=-1).values
    drop = order[..., m:].sort(dim=-1).values
    permutation = _physical_order(keep, state.slots, m)
    keep = keep.gather(-1, permutation)
    kept, removed = _gather(state.keys, keep), _gather(state.keys, drop)
    gram = _kernel(kept, kept, state.temperature)
    gram += state.ridge * torch.eye(m, device=gram.device, dtype=gram.dtype)
    projection = torch.linalg.solve(gram, _kernel(kept, removed, state.temperature))
    values = _gather(state.coefficients, keep) + projection @ _gather(state.coefficients, drop)
    slots = torch.arange(m, device=keep.device).expand_as(keep).clone()
    return replace(state, keys=kept, coefficients=values, scores=state.scores.gather(-1, keep),
                   births=state.births.gather(-1, keep), slots=slots)


def _step(q, k, v, g, beta, score, state):
    if state.decay_axis == 'value':
        coefficients = state.coefficients * g.exp()[..., None, :]
    elif state.decay_axis == 'slot':
        coefficients = state.coefficients * g.gather(-1, state.slots).exp()[..., None]
    else:
        coefficients = state.coefficients * g.exp()[..., None, None]
    prediction = (coefficients * _kernel(state.keys, k[..., None, :], state.temperature)).sum(-2)
    write = beta[..., None] * (v - prediction)
    output = (coefficients * _kernel(state.keys, q[..., None, :], state.temperature)).sum(-2)
    output += write * _kernel(k[..., None, :], q[..., None, :], state.temperature).squeeze(-1)
    b, h = beta.shape
    birth = state.births.new_full((b, h, 1), state.position)
    state = replace(state, keys=torch.cat((state.keys, k[..., None, :]), -2),
                    coefficients=torch.cat((coefficients, write[..., None, :]), -2),
                    scores=torch.cat((state.scores, score[..., None]), -1),
                    births=torch.cat((state.births, birth), -1),
                    slots=torch.cat((state.slots, birth % state.num_slots), -1), position=state.position+1)
    if state.position % state.chunk_size == 0:
        state = compressed_state(state)
    return output, state


@torch.no_grad()
def recurrent_komp(q, k, v, log_decay, beta, state, *, scores=None, attention_mask=None, backend='torch'):
    """Consume only new tokens and return outputs plus a bounded resumable state.

    q/k: [B,T,H,K], v: [B,T,H,V]. beta already includes any score write gate.
    Exact uses chunk_size=1. A final short Block chunk stays open across calls.
    Backward, padded batches, and per-sequence unequal positions are unsupported.
    """
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError('invalid operand shapes')
    if attention_mask is not None and (attention_mask.shape != q.shape[:2] or not bool(attention_mask.all())):
        raise ValueError('padding and unequal-length sequences are unsupported')
    b, t, h, d = q.shape
    if state.keys.shape[:2] != (b, h) or state.keys.shape[-1] != d or state.coefficients.shape[-1] != v.shape[-1]:
        raise ValueError('cache shape does not match this batch')
    expected = q.shape[:3] + (() if state.decay_axis == 'scalar' else
                            (state.num_slots if state.decay_axis == 'slot' else v.shape[-1],))
    if log_decay.shape != expected:
        raise ValueError('invalid decay shape for cache axis')
    if state.pending != state.position % state.chunk_size:
        raise ValueError('terminal compressed state cannot resume an unfinished chunk')
    if scores is None:
        if state.eviction == 'learned':
            raise ValueError('learned eviction requires scores')
        scores = torch.zeros_like(beta)
    if scores.shape != beta.shape:
        raise ValueError('scores require [B,T,H]')
    if backend not in ('torch', 'triton'):
        raise ValueError('backend must be torch or triton')
    if any(x.device != state.keys.device for x in (q, k, v, log_decay, beta, scores)):
        raise ValueError('all tensors and cache must share one device')
    if backend == 'triton' and (not q.is_cuda or state.keys.dtype != torch.float32):
        raise ValueError('Triton requires CUDA and FP32 state')
    dtype = state.keys.dtype
    with torch.autocast(q.device.type, enabled=False):
        qn, kn = [F.normalize(x.to(dtype), dim=-1, eps=1e-6) for x in (q, k)]
        operands = [x.to(dtype) for x in (v, log_decay, beta, scores)]
        outputs = []
        for i in range(t):
            if backend == 'triton':
                from .triton_step import step
                output, state = step(qn[:, i], kn[:, i], *(x[:, i] for x in operands), state)
            else:
                output, state = _step(qn[:, i], kn[:, i], *(x[:, i] for x in operands), state)
            outputs.append(output)
        result = torch.stack(outputs, 1) if outputs else v.new_empty(b, 0, h, v.shape[-1])
        return result.to(v.dtype), state
