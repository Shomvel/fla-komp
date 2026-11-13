import importlib.util
import ast
import math
import os
from pathlib import Path

import pytest
import torch

from komp_inference import compressed_state, initialize_state, recurrent_komp
from komp_inference.prefill import native_prefill

torch.set_num_threads(1)
SOURCE_ROOT = Path(os.environ.get('KOMP_SOURCE_ROOT', '/Users/shom/Downloads/fla-komp'))
REFERENCE = SOURCE_ROOT/'fla/ops/komp/block_reference.py'
spec = importlib.util.spec_from_file_location('frozen_block_reference', REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def operands(t=259, dtype=torch.float64, axis='slot'):
    torch.manual_seed(42)
    b, h, d, m, dv = 2, 2, 7, 5, 9
    q, k = [torch.randn(b, t, h, d, dtype=dtype) for _ in range(2)]
    v = torch.randn(b, t, h, dv, dtype=dtype)
    width = m if axis == 'slot' else dv
    g = -torch.rand(b, t, h, width, dtype=dtype)*.03
    beta = torch.rand(b, t, h, dtype=dtype)*.4
    keys = torch.randn(b, h, m, d, dtype=dtype)
    coefficients = torch.randn(b, h, m, dv, dtype=dtype)*.1
    scores = torch.randint(0, 3, (b, t, h)).to(dtype)
    initial_scores = torch.zeros(b, h, m, dtype=dtype)
    return q, k, v, g, beta, keys, coefficients, scores, initial_scores


@pytest.mark.parametrize('chunk,eviction,t', [(1, 'fifo', 35), (1, 'learned', 35),
                                            (128, 'fifo', 259), (128, 'learned', 259)])
def test_frozen_slot_reference(chunk, eviction, t):
    q, k, v, g, beta, keys, values, scores, initial = operands(t=t)
    state = initialize_state(keys, values, chunk_size=chunk, eviction=eviction, initial_scores=initial)
    actual, state = recurrent_komp(q, k, v, g, beta, state, scores=scores)
    expected, (ek, ev) = reference.block_komp_token_reference(
        q, k, v, g, beta, (keys, values), chunk_size=chunk, evict=eviction,
        scores=scores, initial_scores=initial)
    terminal = compressed_state(state)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(terminal.keys, ek, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(terminal.coefficients, ev, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize('axis,eviction,chunk', [('slot', 'fifo', 128), ('slot', 'learned', 128),
                                               ('value', 'fifo', 128), ('value', 'learned', 128),
                                               ('slot', 'fifo', 1), ('value', 'fifo', 1)])
def test_arbitrary_stream_boundaries(axis, eviction, chunk):
    q, k, v, g, beta, keys, values, scores, initial = operands(axis=axis)
    original = initialize_state(keys, values, chunk_size=chunk, decay_axis=axis,
                                eviction=eviction, initial_scores=initial)
    expected, es = recurrent_komp(q, k, v, g, beta, original, scores=scores)
    state, outputs = original.clone(), []
    cuts = [0, 1, 31, 127, 128, 129, 255, 257, 259]
    for start, end in zip(cuts[:-1], cuts[1:]):
        output, state = recurrent_komp(q[:, start:end], k[:, start:end], v[:, start:end],
                                       g[:, start:end], beta[:, start:end], state, scores=scores[:, start:end])
        outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=0, rtol=0)
    for name in ('keys', 'coefficients', 'scores', 'births', 'slots'):
        torch.testing.assert_close(getattr(state, name), getattr(es, name), atol=0, rtol=0)
    assert state.position == es.position == 259
    assert original.position == 0
    assert state.keys.shape[-2] <= state.num_slots+state.chunk_size-1


def test_batch_reorder_and_rejections():
    q, k, v, g, beta, keys, values, scores, initial = operands(t=9)
    state = initialize_state(keys, values, eviction='learned', initial_scores=initial)
    expected, end = recurrent_komp(q, k, v, g, beta, state, scores=scores)
    indices = torch.tensor([1, 0, 1])
    output, reordered = recurrent_komp(*(x[indices] for x in (q, k, v, g, beta)),
                                      state.index_select(indices), scores=scores[indices])
    torch.testing.assert_close(output, expected[indices], atol=0, rtol=0)
    torch.testing.assert_close(reordered.coefficients, end.coefficients[indices], atol=0, rtol=0)
    with pytest.raises(ValueError, match='padding'):
        recurrent_komp(q, k, v, g, beta, state, scores=scores, attention_mask=torch.zeros(2, 9))
    with pytest.raises(ValueError, match='terminal compressed'):
        recurrent_komp(q, k, v, g, beta, compressed_state(end), scores=scores)
    with pytest.raises(ValueError, match='requires scores'):
        recurrent_komp(q, k, v, g, beta, state)


@pytest.mark.parametrize('eviction,t', [('fifo', 259), ('learned', 259)])
def test_independent_value_reference(eviction, t):
    namespace = dict(torch=torch, math=math, F=torch.nn.functional)
    flex = ast.parse((SOURCE_ROOT/'fla/ops/komp/block_flex.py').read_text())
    selected = [node for node in flex.body if isinstance(node, ast.FunctionDef) and node.name in ('_assign', '_gather_rows')]
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'frozen-flex-functions', 'exec'), namespace)
    root = Path(os.environ.get('KOMP_VALUE_LEARNED_ROOT',
                              '/Users/shom/Downloads/komp-value-decay-learned-123-340m-20c-20261107'))
    value = ast.parse((root/'value_learned.py').read_text())
    selected = [node for node in value.body if isinstance(node, ast.FunctionDef)
                and node.name in ('_check', '_kern', '_routing', 'recurrent_value_learned')]
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'frozen-value-functions', 'exec'), namespace)
    q, k, v, g, beta, keys, values, scores, initial = operands(t=t, axis='value')
    # zero ties select newest atoms, reproducing FIFO while exercising the independent learned oracle
    if eviction == 'fifo':
        scores.zero_()
    state = initialize_state(keys, values, decay_axis='value', eviction=eviction, initial_scores=initial)
    actual, state = recurrent_komp(q, k, v, g, beta, state, scores=scores)
    expected, (ek, ev) = namespace['recurrent_value_learned'](
        q, k, v, g, beta, (keys, values), scores=scores, initial_scores=initial)
    terminal = compressed_state(state)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(terminal.keys, ek, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(terminal.coefficients, ev, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize('eviction,t', [('fifo', 259), ('learned', 259), ('learned', 128)])
def test_native_prefill_metadata_and_tail(eviction, t):
    q, k, v, g, beta, keys, values, scores, initial = operands(t=t)
    state = initialize_state(keys, values, chunk_size=128, eviction=eviction, initial_scores=initial)
    # use M=4 so the native C128 physical-slot boundary is aligned
    keys, values, initial = keys[..., :4, :], values[..., :4, :], initial[..., :4]
    g = g[..., :4]
    state = initialize_state(keys, values, chunk_size=128, eviction=eviction, initial_scores=initial)
    expected, es = recurrent_komp(q, k, v, g, beta, state, scores=scores)
    def operator(q, k, v, g, beta, initial_state, chunk_size, evict='fifo', scores=None, initial_scores=None):
        return reference.block_komp_token_reference(q, k, v, g, beta, initial_state, chunk_size=chunk_size,
                                                    evict=evict, scores=scores, initial_scores=initial_scores)
    output, actual = native_prefill(q, k, v, g, beta, state, operator,
                                    scores=scores if eviction == 'learned' else None, backend='torch')
    torch.testing.assert_close(output, expected, atol=2e-11, rtol=2e-11)
    for field in ('keys', 'coefficients', 'births', 'slots'):
        torch.testing.assert_close(getattr(actual, field), getattr(es, field), atol=2e-11, rtol=2e-11)
    if eviction == 'learned':
        torch.testing.assert_close(actual.scores, es.scores, atol=0, rtol=0)
