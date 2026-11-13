"""Diagnostic regressions: failed numeric comparisons remain observable; cache remains resumable."""
import torch
import pytest

from diagnose_model import (difference, logits_details, candidate_operands, make_state,
                            postprocess, NativeOutputProbe, Recorder)
from komp_inference import compressed_state, recurrent_komp
from komp_inference.model import RecurrentKOMPLM
from test_model import model


def test_failure_is_recorded_without_losing_top1(tmp_path):
    expected = torch.tensor([[[1., 2., 3.]]])
    actual = torch.tensor([[[1., 2., 1.]]])
    recorder = Recorder(tmp_path)
    row = recorder.emit('known_failure', actual, expected, **logits_details(actual, expected))
    recorder.stream.close()
    assert not row['within_frozen_model_numeric_gate']
    assert not row['top1_equal']
    assert row['expected_top2_ids'] == [[2, 1]]
    assert row['actual_top2_ids'][0][0] == 1
    assert row['expected_margin'] == [1.]
    assert (tmp_path/'comparisons.jsonl').read_text().count('\n') == 1
    assert not difference(torch.tensor([float('nan')]), torch.ones(1))['finite']


def test_greedy_tie_uses_smallest_argmax_index():
    actual = torch.tensor([[[0., 9.5625, 0., 9.5625]]])
    expected = torch.tensor([[[0., 9.5, 0., 9.5625]]])
    report = logits_details(actual, expected)
    assert report['actual_argmax'] == [1]
    assert report['expected_argmax'] == [3]
    assert report['top1_equal'] is False


@pytest.mark.parametrize('axis,learned', [('slot', False), ('slot', True), ('value', False), ('value', True)])
def test_operand_probe_reproduces_r1_mixer(axis, learned):
    original = model('block', axis, learned)
    adapter = RecurrentKOMPLM(original, decay_axis=axis)
    mixer = original.model.layers[0].attn
    x = original.model.embeddings(torch.tensor([[1, 2, 3, 4, 5]]))
    with torch.inference_mode():
        expected, expected_cache = adapter._mixer(mixer, x, None)
        operands, conv = candidate_operands(mixer, x, axis)
        output, state = recurrent_komp(*operands[:5], make_state(operands, 'block', axis), scores=operands[7])
        actual, _, _ = postprocess(mixer, output, x, original.config)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state.coefficients, expected_cache.recurrent_state.coefficients, rtol=0, atol=0)
    for a, e in zip(conv, expected_cache.conv_state):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_native_output_hypothesis_keeps_partial_cache_open():
    original = model('block', 'slot', False)
    def operator(q, k, v, g, beta, initial, chunk_size):
        from komp_inference import initialize_state
        out, state = recurrent_komp(q, k, v, g, beta, initialize_state(*initial, chunk_size=chunk_size))
        terminal = compressed_state(state)
        return out, (terminal.keys, terminal.coefficients)
    probe = NativeOutputProbe(original, backend='torch', prefill_operator=operator)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.inference_mode():
        _, cache = probe(ids)
        assert all(layer.recurrent_state.pending == 5 for layer in cache.layers)
        _, cache = probe(torch.tensor([[6]]), cache=cache)
        assert cache.position == 6
        assert all(layer.recurrent_state.pending == 6 for layer in cache.layers)
