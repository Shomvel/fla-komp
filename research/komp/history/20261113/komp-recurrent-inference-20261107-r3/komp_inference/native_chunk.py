"""Bounded Block inference preserving the frozen native chunk arithmetic.

Slot decay only. Store raw keys so each native call normalizes retained keys
once, just as the full-prefix operator does. Partial native terminal states
are discarded; only C128 boundaries may replace committed coefficients.
"""
from dataclasses import dataclass, replace

import torch

from .recurrent import _physical_order


@dataclass(frozen=True)
class NativeChunkState:
    raw_keys: torch.Tensor
    coefficients: torch.Tensor
    scores: torch.Tensor
    births: torch.Tensor
    committed: int
    pending_operands: tuple
    eviction: str
    chunk_size: int = 128

    @property
    def pending(self):
        return 0 if not self.pending_operands else self.pending_operands[0].shape[1]

    @property
    def position(self):
        return self.committed+self.pending

    def index_select(self, indices):
        return replace(self, raw_keys=self.raw_keys.index_select(0, indices),
                       coefficients=self.coefficients.index_select(0, indices),
                       scores=self.scores.index_select(0, indices), births=self.births.index_select(0, indices),
                       pending_operands=tuple(None if x is None else x.index_select(0, indices) for x in self.pending_operands))

    def nbytes(self):
        return sum(x.numel()*x.element_size() for x in
                   (self.raw_keys, self.coefficients, self.scores, self.births, *self.pending_operands) if x is not None)


def initialize_native_state(keys, coefficients, *, initial_scores=None, chunk_size=128):
    if chunk_size != 128 or keys.ndim != 4 or keys.shape[:3] != coefficients.shape[:3]:
        raise ValueError('native chunk cache requires C128 and [B,H,M,K/V] state')
    b, h, m, _ = keys.shape
    if 128 % m or m > 64:
        raise ValueError('native chunk cache requires M <=64 dividing 128')
    if keys.dtype != torch.float32 or coefficients.dtype != torch.float32:
        raise ValueError('native chunk cache requires original FP32 keys/state')
    if initial_scores is not None and initial_scores.shape != (b, h, m):
        raise ValueError('initial scores shape mismatch')
    scores = keys.new_zeros(b, h, m) if initial_scores is None else initial_scores.float().clone()
    births = torch.arange(-m, 0, device=keys.device).expand(b, h, m).clone()
    return NativeChunkState(keys.clone(), coefficients.clone(), scores, births, 0, (),
                            'fifo' if initial_scores is None else 'learned')


def _new_metadata(state, k, scores):
    b, n, h, _ = k.shape
    m = state.raw_keys.shape[-2]
    birth = torch.arange(state.committed, state.committed+n, device=k.device).expand(b, h, n)
    births = torch.cat((state.births, birth), -1)
    rank_scores = torch.cat((state.scores, scores.transpose(1, 2)), -1) if scores is not None else births.float()
    recent = births.argsort(dim=-1, descending=True, stable=True)
    order = recent.gather(-1, rank_scores.gather(-1, recent).argsort(dim=-1, descending=True, stable=True))
    return births, rank_scores, order


def native_rank_scores(state, k, scores):
    """Encode true (score,global birth) ordering, avoiding native old-birth reset.

    Only selection sees these exact small integer ranks; beta retains its true
    score-dependent write gate. No model parameter or actual score is changed.
    """
    if state.eviction == 'fifo':
        return {}
    if scores is None:
        raise ValueError('learned native chunk cache requires true scores')
    _, _, order = _new_metadata(state, k, scores)
    size = order.shape[-1]
    rank = torch.zeros_like(order).scatter(-1, order, torch.arange(size, 0, -1, device=k.device).expand_as(order)).float()
    m = state.raw_keys.shape[-2]
    return dict(evict='learned', initial_scores=rank[..., :m].contiguous(),
                scores=rank[..., m:].transpose(1, 2).contiguous())


def _commit(state, operands, final_coefficients):
    q, k, v, g, beta, scores = operands
    if k.shape[1] != state.chunk_size:
        raise ValueError('only complete chunks may be committed')
    m = state.raw_keys.shape[-2]
    births, rank_scores, order = _new_metadata(state, k, scores)
    keep = order[..., :m].sort(dim=-1).values
    # FIFO native final state is age order; at C128 boundaries it coincides
    # with physical slots because 128 is divisible by M.
    if state.eviction == 'learned':
        slots = torch.cat((torch.arange(m, device=k.device).expand_as(state.births), births[..., m:] % m), -1)
        keep = keep.gather(-1, _physical_order(keep, slots, m))
    pool = torch.cat((state.raw_keys, k.transpose(1, 2).float()), -2)
    keys = pool.gather(-2, keep[..., None].expand(*keep.shape, pool.shape[-1])).contiguous()
    true_scores = rank_scores.gather(-1, keep) if scores is not None else torch.zeros_like(state.scores)
    return replace(state, raw_keys=keys, coefficients=final_coefficients.float().clone(),
                   scores=true_scores.contiguous(), births=births.gather(-1, keep).contiguous(),
                   committed=state.committed+state.chunk_size, pending_operands=())


def _call(operator, operands, state):
    q, k, v, g, beta, scores = operands
    return operator(q, k, v, g, beta, (state.raw_keys, state.coefficients), chunk_size=state.chunk_size,
                    **native_rank_scores(state, k, scores))


def _validate(operands, state):
    q, k, v, g, beta, scores = operands
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError('invalid native chunk operand shapes')
    m = state.raw_keys.shape[-2]
    if g.shape != (*q.shape[:3], m):
        raise ValueError('r3 native chunk cache supports slot decay only')
    if state.committed % 128 or not 0 <= state.pending < 128:
        raise ValueError('invalid native chunk boundary')
    if (scores is None) != (state.eviction == 'fifo'):
        raise ValueError('score/eviction contract mismatch')
    if scores is not None and scores.shape != beta.shape:
        raise ValueError('score shape mismatch')


@torch.no_grad()
def native_chunk_append(q, k, v, g, beta, state, operator, *, scores=None):
    """Consume only newly projected tokens, recomputing at most one C128 block.

    May consume multiple chunks in one API call. Persistent state remains
    M atoms plus <128 cached projected operands, independent of total length.
    """
    incoming = (q, k, v, g, beta, scores)
    _validate(incoming, state)
    outputs = []
    cursor = 0
    while cursor < q.shape[1]:
        pending = state.pending
        count = min(128-pending, q.shape[1]-cursor)
        part = tuple(None if x is None else x[:, cursor:cursor+count] for x in incoming)
        if pending:
            part = tuple(None if x is None else torch.cat((old, x), 1) for old, x in zip(state.pending_operands, part))
        output, (_, coef) = _call(operator, part, state)
        outputs.append(output[:, pending:])
        if pending+count == 128:
            state = _commit(state, part, coef)
        else:
            # Clone slices: retaining a view would hold the full prompt backing storage.
            state = replace(state, pending_operands=tuple(None if x is None else x.clone() for x in part))
        cursor += count
    return torch.cat(outputs, 1), state


@torch.no_grad()
def native_chunk_prefill(q, k, v, g, beta, state, operator, *, scores=None):
    """Use unmodified full native prefill output, retaining only committed state and tail."""
    operands = (q, k, v, g, beta, scores)
    _validate(operands, state)
    if state.position:
        return native_chunk_append(q, k, v, g, beta, state, operator, scores=scores)
    # At fresh start native initial births are exactly -M..-1, so true scores
    # preserve original full-prefix dictionary decisions without rank encoding.
    options = dict(evict='learned', scores=scores, initial_scores=state.scores) if scores is not None else {}
    output, (_, final_coef) = operator(q, k, v, g, beta, (state.raw_keys, state.coefficients), chunk_size=128, **options)
    count = q.shape[1]//128*128
    if count:
        full = tuple(None if x is None else x[:, :count] for x in operands)
        if count != q.shape[1]:
            full_options = dict(evict='learned', scores=scores[:, :count], initial_scores=state.scores) if scores is not None else {}
            _, (_, final_coef) = operator(*full[:5], (state.raw_keys, state.coefficients), chunk_size=128, **full_options)
        # Only routing needs iteration here; coefficients come from native full-prefix scan.
        for start in range(0, count, 128):
            part = tuple(None if x is None else x[:, start:start+128] for x in operands)
            state = _commit(state, part, final_coef)
    if count < q.shape[1]:
        state = replace(state, pending_operands=tuple(None if x is None else x[:, count:].clone() for x in operands))
    return output, state
