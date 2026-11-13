"""Reuse frozen native prefill only at valid algorithmic chunk boundaries."""

from dataclasses import replace

import torch

from .recurrent import _physical_order, recurrent_komp


def _metadata(state, scores, count):
    m, c = state.num_slots, state.chunk_size
    for start in range(0, count, c):
        birth = torch.arange(state.position, state.position+c, device=state.births.device)
        birth = birth.expand(*state.births.shape[:-1], c)
        births = torch.cat((state.births, birth), -1)
        new_scores = scores[:, start:start+c].transpose(1, 2) if scores is not None else birth.float()*0
        rank_scores = torch.cat((state.scores, new_scores), -1)
        slots = torch.cat((state.slots, birth % m), -1)
        recent = births.argsort(dim=-1, descending=True, stable=True)
        rank = rank_scores if state.eviction == 'learned' else births
        order = recent.gather(-1, rank.gather(-1, recent).argsort(dim=-1, descending=True, stable=True))
        keep = order[..., :m].sort(dim=-1).values
        keep = keep.gather(-1, _physical_order(keep, slots, m))
        state = replace(state, scores=rank_scores.gather(-1, keep), births=births.gather(-1, keep),
                        position=state.position+c)
    return state


@torch.no_grad()
def native_prefill(q, k, v, g, beta, state, operator, *, scores=None, backend='triton'):
    """Native whole chunks followed by a resumable recurrent tail; never finalize a partial chunk."""
    if state.position != 0 or state.pending:
        return recurrent_komp(q, k, v, g, beta, state, scores=scores, backend=backend)
    t = q.shape[1]
    count = t if state.chunk_size == 1 else t//state.chunk_size*state.chunk_size
    if count == 0:
        return recurrent_komp(q, k, v, g, beta, state, scores=scores, backend=backend)
    args = [x[:, :count] for x in (q, k, v, g, beta)]
    initial = (state.keys, state.coefficients)
    if state.chunk_size == 1:
        if state.eviction != 'fifo' or state.decay_axis != 'slot':
            raise ValueError('native Exact prefill currently supports frozen FIFO slot models')
        output, (keys, coef), metadata = operator(*args, initial, eviction='fifo', start_position=0)
        state = replace(state, keys=keys, coefficients=coef, position=count,
                        scores=metadata['scores'], births=metadata['births'])
    else:
        if state.chunk_size % state.num_slots:
            raise ValueError('native FIFO prefill requires chunk_size divisible by M')
        options = dict(evict='learned', scores=scores[:, :count], initial_scores=state.scores) if scores is not None else {}
        output, (keys, coef) = operator(*args, initial, chunk_size=state.chunk_size, **options)
        state = _metadata(state, scores, count)
        state = replace(state, keys=keys, coefficients=coef)
    if count == t:
        return output, state
    tail, state = recurrent_komp(*(x[:, count:] for x in (q, k, v, g, beta)), state,
                                 scores=None if scores is None else scores[:, count:], backend=backend)
    return torch.cat((output, tail), 1), state
