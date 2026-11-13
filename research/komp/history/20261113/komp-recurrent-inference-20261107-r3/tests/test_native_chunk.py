"""CPU state/ordering properties against the independent frozen token oracle."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from komp_inference.native_chunk import (initialize_native_state, native_chunk_append,
    native_chunk_prefill, native_rank_scores)
from komp_inference.native_model import NativeChunkKOMPLM
from komp_inference.model import RecurrentKOMPLM
from test_recurrent import reference
from test_model import model


def data(length=385, learned=False):
    torch.manual_seed(941)
    b, h, m, d = 2, 2, 4, 7
    q, k, v = [torch.randn(b, length, h, d) for _ in range(3)]
    g = -torch.rand(b, length, h, m)*.03
    beta = torch.rand(b, length, h)*.4
    keys = torch.randn(b, h, m, d)
    coef = torch.randn_like(keys)*.1
    scores = torch.randint(0, 3, beta.shape).float() if learned else None
    initial = torch.zeros(b, h, m) if learned else None
    return (q, k, v, g, beta, scores), initialize_native_state(keys, coef, initial_scores=initial)


def oracle(q, k, v, g, beta, initial_state, chunk_size, evict='fifo', scores=None, initial_scores=None):
    return reference.block_komp_token_reference(q, k, v, g, beta, initial_state, chunk_size=chunk_size,
                                                evict=evict, scores=scores, initial_scores=initial_scores)


@pytest.mark.parametrize('learned', [False, True])
@pytest.mark.parametrize('prefill', [31, 127, 128, 129, 257])
def test_prefill_and_arbitrary_boundary_append(learned, prefill):
    operands, initial = data(learned=learned)
    options = dict(evict='learned', scores=operands[-1], initial_scores=initial.scores) if learned else {}
    expected, _ = oracle(*operands[:5], (initial.raw_keys, initial.coefficients), chunk_size=128, **options)
    out, state = native_chunk_prefill(*(x[:, :prefill] for x in operands[:5]), initial, oracle,
                                      scores=None if not learned else operands[-1][:, :prefill])
    outputs = [out]
    cuts = [prefill]+[c for c in (127, 128, 129, 255, 256, 257, 383, 384, 385) if c > prefill]
    calls = []
    def traced(*args, **kwargs):
        calls.append(args[0].shape[1])
        return oracle(*args, **kwargs)
    for start, end in zip(cuts[:-1], cuts[1:]):
        out, state = native_chunk_append(*(x[:, start:end] for x in operands[:5]), state, traced,
                                         scores=None if not learned else operands[-1][:, start:end])
        outputs.append(out)
        assert state.position == end
        assert state.committed == end//128*128
        assert state.pending == end % 128
        for x in state.pending_operands:
            if x is not None:
                assert x.untyped_storage().nbytes() == x.numel()*x.element_size()
    assert all(1 <= count <= 128 for count in calls)
    torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=3e-6, rtol=3e-5)
    assert state.raw_keys.shape == initial.raw_keys.shape
    assert state.coefficients.shape == initial.coefficients.shape
    assert state.pending == 1


def test_partial_native_terminal_is_discarded_and_raw_keys_retained():
    operands, initial = data(129)
    def poisoned(*args, **kwargs):
        output, (keys, coef) = oracle(*args, **kwargs)
        if args[0].shape[1] < 128:
            coef = torch.full_like(coef, 999999.)
        return output, (keys, coef)
    _, state = native_chunk_prefill(*(x[:, :31] for x in operands[:5]), initial, poisoned)
    assert state.committed == 0
    assert torch.equal(state.coefficients, initial.coefficients)
    _, state = native_chunk_append(*(x[:, 31:128] for x in operands[:5]), state, poisoned)
    assert state.committed == 128 and state.pending == 0
    # Original raw k retained, without premature normalization.
    assert torch.equal(state.raw_keys, operands[1][:, 124:128].transpose(1, 2).float())
    before = state.coefficients.clone()
    _, state = native_chunk_append(*(x[:, 128:] for x in operands[:5]), state, poisoned)
    assert torch.equal(before, state.coefficients)


def test_learned_scores_encode_global_birth_ties_without_changing_beta():
    operands, state = data(4, learned=True)
    state = replace(state, committed=256, births=torch.tensor([33, 200, 12, 145]).expand(2, 2, 4).clone(),
                    scores=torch.tensor([1., 1., 2., 1.]).expand(2, 2, 4).clone())
    operands[-1].fill_(1.)
    original_beta = operands[4].clone()
    encoded = native_rank_scores(state, operands[1], operands[-1])
    both = torch.cat((encoded['initial_scores'], encoded['scores'].transpose(1, 2)), -1)
    # Unique ranks force native's local-old-birth reset to be irrelevant.
    expected_order = [2, 7, 6, 5, 4, 1, 3, 0]
    assert both[0, 0].argsort(descending=True).tolist() == expected_order
    assert both[0, 0].unique().numel() == 8
    assert torch.equal(operands[4], original_beta)
    assert torch.equal(state.scores[0, 0], torch.tensor([1., 1., 2., 1.]))


def test_native_cache_reorder_duplicate_and_bounded_storage():
    operands, state = data(513, learned=True)
    _, state = native_chunk_prefill(*(x[:, :257] for x in operands[:5]), state, oracle, scores=operands[-1][:, :257])
    prior_size = state.nbytes()
    index = torch.tensor([1, 0, 1])
    reordered = state.index_select(index)
    actual, reordered = native_chunk_append(*(x[index, 257:259] for x in operands[:5]), reordered, oracle,
                                             scores=operands[-1][index, 257:259])
    expected, _ = native_chunk_append(*(x[:, 257:259] for x in operands[:5]), state, oracle, scores=operands[-1][:, 257:259])
    torch.testing.assert_close(actual, expected[index], atol=0, rtol=0)
    _, state = native_chunk_append(*(x[:, 257:] for x in operands[:5]), state, oracle, scores=operands[-1][:, 257:])
    assert state.position == 513 and state.nbytes() == prior_size


@pytest.mark.parametrize('learned', [False, True])
def test_native_model_decode_reset_and_unchanged_weights(learned):
    original = model('block', 'slot', learned)
    native = NativeChunkKOMPLM(original, backend='torch', prefill_operator=oracle)
    token = RecurrentKOMPLM(original, backend='torch')
    before = {n: v.clone() for n, v in original.state_dict().items()}
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    expected, _ = token(ids)
    actual, cache = native(ids)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    _, cache = native(torch.tensor([[6]]), cache=cache)
    assert cache.position == 6 and all(x.recurrent_state.pending == 6 for x in cache.layers)
    cache.reset()
    repeat, cache = native(ids, cache=cache)
    torch.testing.assert_close(actual, repeat, atol=0, rtol=0)
    for n, value in original.state_dict().items():
        assert torch.equal(value, before[n])


def test_native_adapter_rejects_unvalidated_value():
    with pytest.raises(NotImplementedError, match='Value decay'):
        NativeChunkKOMPLM(model('block', 'value', True), decay_axis='value', prefill_operator=oracle)
