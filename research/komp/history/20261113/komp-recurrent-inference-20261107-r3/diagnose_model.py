"""Diagnostic only: fixed-input attribution of r1 model numerical drift.

No numeric difference aborts this script. It records existing thresholds but
does not accept a candidate, change a model, or produce evaluation scores.
Run one model in a fresh process, on a GPU selected by the queue owner.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from gpu_validate import MODELS, frozen_operator
from komp_inference import compressed_state, initialize_state, recurrent_komp
from komp_inference.model import RecurrentKOMPLM, LayerCache
from komp_inference.prefill import native_prefill
from verify_sources import fingerprint, verify_candidate, verify_model


def difference(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f'shape mismatch {actual.shape} != {expected.shape}')
    a, e = actual.detach().double(), expected.detach().double()
    d = a-e
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    peak = d.abs().max().item()
    rmsrel = (d.square().mean().sqrt()/e.square().mean().sqrt().clamp_min(1e-12)).item()
    return dict(shape=list(a.shape), actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
                finite=finite, absolute=peak, rmsrel=rmsrel, equal=bool(torch.equal(actual, expected)),
                within_frozen_model_numeric_gate=finite and (peak < .0003 or rmsrel < .003))


def logits_details(actual, expected):
    a, e = actual[:, -1].float(), expected[:, -1].float()
    av, ai = a.topk(2, dim=-1)
    ev, ei = e.topk(2, dim=-1)
    actual_argmax, expected_argmax = a.argmax(-1), e.argmax(-1)
    return dict(top1_equal=bool((actual_argmax == expected_argmax).all()),
                actual_argmax=actual_argmax.cpu().tolist(), expected_argmax=expected_argmax.cpu().tolist(),
                actual_top2_ids=ai.cpu().tolist(), expected_top2_ids=ei.cpu().tolist(),
                actual_top2_values=av.cpu().tolist(), expected_top2_values=ev.cpu().tolist(),
                actual_margin=(av[:, 0]-av[:, 1]).cpu().tolist(),
                expected_margin=(ev[:, 0]-ev[:, 1]).cpu().tolist())


class Recorder:
    def __init__(self, directory):
        self.directory = directory
        self.rows = []
        self.stream = (directory/'comparisons.jsonl').open('x')

    def emit(self, label, actual, expected, **extra):
        row = dict(label=label, **difference(actual, expected), **extra)
        self.rows.append(row)
        self.stream.write(json.dumps(row)+'\n')
        self.stream.flush()
        print(json.dumps(row), flush=True)
        return row


def candidate_operands(mixer, x, axis):
    """Independent reproduction of r1 adapter's fresh-call operand path."""
    b, t, _ = x.shape
    h, m, d = mixer.keys.shape
    projected, conv = [], []
    for projection, convolution in zip((mixer.q, mixer.k, mixer.v), mixer.convs):
        value, final = convolution(projection(x), cache=None, output_final_state=True)
        projected.append(value.reshape(b, t, h, d))
        conv.append(final)
    with torch.autocast(x.device.type, enabled=False):
        z = mixer.decay(x.float()).reshape(b, t, h, d if axis == 'value' else m)
        g = -mixer.A_log.float().exp()*F.softplus(z+mixer.dt_bias.float()) if hasattr(mixer, 'A_log') else F.logsigmoid(z)/8
        beta = F.linear(x.float(), mixer.beta.weight.float(), mixer.beta.bias.float()).sigmoid()
        scores = None
        if hasattr(mixer, 'score'):
            scores = F.linear(x.float(), mixer.score.weight.float(), mixer.score.bias.float())
            beta = beta*(2*scores.sigmoid())
    keys = mixer.keys.float()[None].expand(b, h, m, d)
    coef = x.new_zeros(b, h, m, d, dtype=torch.float32)
    initial = mixer.initial_score[None].expand(b, h, m) if scores is not None else None
    return (*projected, g, beta, keys, coef, scores, initial), tuple(conv)


def unpack_operands(operands):
    if len(operands) == 7:
        return (*operands, None, None)
    if len(operands) != 9:
        raise ValueError('unrecognized frozen operands contract')
    return tuple(operands)


def make_state(operands, kind, axis):
    return initialize_state(operands[5], operands[6], chunk_size=1 if kind == 'exact' else 128,
                            decay_axis=axis, eviction='learned' if operands[7] is not None else 'fifo',
                            initial_scores=operands[8])


def native(operator, operands, kind, normalized_keys=None):
    q, k, v, g, beta, keys, coef, scores, initial = operands
    if normalized_keys is not None:
        keys = normalized_keys
    if kind == 'exact':
        out, state, _ = operator(q, k, v, g, beta, (keys, coef), eviction='fifo')
    else:
        options = dict(evict='learned', scores=scores, initial_scores=initial) if scores is not None else {}
        out, state = operator(q, k, v, g, beta, (keys, coef), chunk_size=128, **options)
    return out, state


def postprocess(mixer, output, x, config):
    b, t, _ = x.shape
    y = mixer.norm(output.float()).to(x.dtype).reshape(b, t, -1)
    gate = mixer.gate(x)
    config = getattr(mixer, 'config', config)
    sigmoid = getattr(mixer, 'output_sigmoid', getattr(config, 'output_sigmoid', False))
    gate = gate.sigmoid() if sigmoid else F.silu(gate)
    return mixer.out(y*gate), y, gate


class NativeOutputProbe(RecurrentKOMPLM):
    """Hypothesis only: native entire prefill output plus independently built open cache.

    It deliberately does not finalize the recurrent cache's partial chunk. Its
    next-token differences are recorded; this is not a validated implementation.
    """
    def _mixer(self, mixer, x, cache):
        if cache is not None:
            return super()._mixer(mixer, x, cache)
        operands, conv = candidate_operands(mixer, x, self.decay_axis)
        kind = getattr(mixer, 'kind', self.model.config.kind)
        state = make_state(operands, kind, self.decay_axis)
        out, _ = native(self.prefill_operator, operands, kind)
        _, state = native_prefill(*operands[:5], state, self.prefill_operator,
                                  scores=operands[7], backend=self.backend)
        update, _, _ = postprocess(mixer, out, x, self.model.config)
        return update, LayerCache(state, conv)


def trace_baseline(model, ids):
    trace = [dict() for _ in model.model.layers]
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def pre(module, args, kwargs, i=i):
            trace[i]['input'] = (args[0] if args else kwargs['hidden_states']).detach().clone()
        def norm(module, args, output, i=i):
            trace[i]['attn_norm'] = output.detach().clone()
        def attn(module, args, output, i=i):
            trace[i]['mixer'] = output[0].detach().clone()
        def post(module, args, output, i=i):
            trace[i]['output'] = output[0].detach().clone()
        hooks.extend((layer.register_forward_pre_hook(pre, with_kwargs=True),
                      layer.attn_norm.register_forward_hook(norm), layer.attn.register_forward_hook(attn),
                      layer.register_forward_hook(post)))
    try:
        hidden = model(ids)
        logits = model.lm_head(hidden[:, -1:]).float()
    finally:
        for hook in hooks:
            hook.remove()
    return trace, hidden, logits


def compare_terminal(recorder, prefix, state, final):
    with torch.autocast(state.keys.device.type, enabled=False):
        terminal = compressed_state(state)
    keys, coef = final
    if state.chunk_size > 1 and state.eviction == 'fifo':
        inverse_age = terminal.births.argsort(dim=-1).argsort(dim=-1)
        keys, coef = (x.gather(-2, inverse_age[..., None].expand_as(x)) for x in (keys, coef))
    recorder.emit(prefix+'/terminal_keys', terminal.keys, keys)
    recorder.emit(prefix+'/terminal_coefficients', terminal.coefficients, coef)


def fixed_layer(recorder, prefix, layer, trace, adapter, operator, kind, axis):
    x = layer.attn_norm(trace['input'])
    recorder.emit(prefix+'/attn_norm_control', x, trace['attn_norm'])
    frozen = unpack_operands(layer.attn.operands(x))
    candidate, _ = candidate_operands(layer.attn, x, axis)
    names = ('q', 'k', 'v', 'g', 'beta', 'initial_keys', 'initial_coef', 'scores', 'initial_scores')
    for name, a, e in zip(names, candidate, frozen):
        if a is not None:
            recorder.emit(prefix+'/operand_'+name, a, e)
    for name, projection, convolution in zip(('q', 'k', 'v'), (layer.attn.q, layer.attn.k, layer.attn.v), layer.attn.convs):
        projected = projection(x)
        without, _ = convolution(projected, cache=None, output_final_state=False)
        with_state, state = convolution(projected, cache=None, output_final_state=True)
        recorder.emit(prefix+'/conv_final_flag_'+name, with_state, without)
        # W raw projected values, left-padded for an input shorter than W.
        width = convolution.kernel_size[0]
        expected = F.pad(projected.transpose(1, 2), (max(0, width-x.shape[1]), 0))[:, :, -width:]
        recorder.emit(prefix+'/conv_cache_'+name, state, expected)
    out, final = native(operator, frozen, kind)
    state0 = make_state(frozen, kind, axis)
    normalized_out, _ = native(operator, frozen, kind, normalized_keys=state0.keys)
    recorder.emit(prefix+'/native_double_normalization', normalized_out, out)
    outputs, states = {}, {}
    for backend in ('torch', 'triton'):
        outputs[backend], states[backend] = recurrent_komp(*frozen[:5], state0.clone(), scores=frozen[7], backend=backend)
        recorder.emit(prefix+'/operator_'+backend, outputs[backend], out)
        compare_terminal(recorder, prefix+'/operator_'+backend, states[backend], final)
    recorder.emit(prefix+'/triton_vs_torch_output', outputs['triton'], outputs['torch'])
    recorder.emit(prefix+'/triton_vs_torch_open_coef', states['triton'].coefficients, states['torch'].coefficients)
    hybrid, open_state = native_prefill(*frozen[:5], state0.clone(), operator, scores=frozen[7], backend='triton')
    recorder.emit(prefix+'/r1_native_prefill', hybrid, out,
                  position=open_state.position, pending=open_state.pending, atoms=open_state.keys.shape[-2])
    # Controls isolate postprocessing from the operator difference on identical x.
    native_update, norm, gate = postprocess(layer.attn, out, x, adapter.model.config)
    recorder.emit(prefix+'/postprocess_control', native_update, trace['mixer'])
    for label, value in [('torch', outputs['torch']), ('triton', outputs['triton']), ('r1_prefill', hybrid)]:
        update, n, g = postprocess(layer.attn, value, x, adapter.model.config)
        recorder.emit(prefix+'/'+label+'_output_norm', n, norm)
        recorder.emit(prefix+'/'+label+'_output_gate', g, gate)
        recorder.emit(prefix+'/'+label+'_out_projection', update, native_update)
        residual = trace['input']+update
        expected_residual = trace['input']+native_update
        recorder.emit(prefix+'/'+label+'_residual', residual, expected_residual)
        result = residual+layer.mlp(layer.mlp_norm(residual))
        recorder.emit(prefix+'/'+label+'_mlp_result', result, trace['output'])
    candidate_update, _ = adapter._mixer(layer.attn, x, None)
    recorder.emit(prefix+'/r1_mixer_fixed_input', candidate_update, trace['mixer'])
    manual = trace['input']+native_update
    manual = manual+layer.mlp(layer.mlp_norm(manual))
    recorder.emit(prefix+'/native_manual_layer_control', manual, trace['output'])


def cascade(recorder, prefix, model, ids, trace, baseline_hidden, baseline_logits, adapter):
    cache = adapter.new_cache()
    hidden = model.model.embeddings(ids)
    for i, layer in enumerate(model.model.layers):
        recorder.emit(f'{prefix}/layer{i:02d}/input', hidden, trace[i]['input'])
        residual = hidden
        x = layer.attn_norm(hidden)
        recorder.emit(f'{prefix}/layer{i:02d}/attn_norm', x, trace[i]['attn_norm'])
        update, cache.layers[i] = adapter._mixer(layer.attn, x, None)
        recorder.emit(f'{prefix}/layer{i:02d}/mixer', update, trace[i]['mixer'])
        hidden = residual+update
        hidden = hidden+layer.mlp(layer.mlp_norm(hidden))
        recorder.emit(f'{prefix}/layer{i:02d}/output', hidden, trace[i]['output'])
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden[:, -1:]).float()
    cache.position = ids.shape[1]
    recorder.emit(prefix+'/final_norm', hidden, baseline_hidden)
    recorder.emit(prefix+'/logits', logits, baseline_logits, **logits_details(logits, baseline_logits))
    return logits, cache


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--base', default='/gpfs/users/linlizhi')
    parser.add_argument('--lengths', type=int, nargs='+', default=[31, 127, 128, 129, 257])
    parser.add_argument('--decode', type=int, default=4)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if any(n < 1 for n in args.lengths) or args.decode < 0:
        raise ValueError('positive lengths and nonnegative decode required')
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', purpose='diagnosis_only_not_acceptance', model=args.model,
                  started=time.time(), args=vars(args), inherited_r1_manifest='f0ebd0c1b3375e671d849a99e70c2019b3cf20be89d588982e4e5fe86dd050f8')
    recorder = Recorder(directory)
    try:
        report['candidate'] = verify_candidate()
        frozen = report['frozen'] = verify_model(args.model, args.base)
        folder, kind, axis, learned = MODELS[args.model]
        source = Path(args.base)/folder
        sys.path.insert(0, str(source))
        module = importlib.import_module('models')
        torch.set_num_threads(2)
        torch.manual_seed(42)
        if torch.cuda.device_count() != 1:
            raise ValueError('queue owner must select exactly one already-authorized GPU')
        report['runtime'] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                 allow_tf32=torch.backends.cuda.matmul.allow_tf32)
        model = module.LM(kind)
        saved = torch.load(frozen['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        if saved['source_sha'] != frozen['source_manifest_sha256'] or saved['data_sha'] != frozen['data_manifest_sha256']:
            raise ValueError('checkpoint internal lineage mismatch')
        model.load_state_dict(saved['model'], strict=True)
        del saved
        model = model.cuda().eval()
        operator = frozen_operator(kind, axis, learned)
        adapter = RecurrentKOMPLM(model, decay_axis=axis, backend='triton', prefill_operator=operator)
        probe = NativeOutputProbe(model, decay_axis=axis, backend='triton', prefill_operator=operator)
        request = Path(args.base)/'komp-flame-downstream-eval-20261107/requests/niah_single_1_1024.jsonl'
        with request.open() as stream:
            row = json.loads(stream.readline())
        report['request'] = dict(file=str(request), sha256=fingerprint(request), request_id=row.get('request_id', row.get('id')),
                                 tokens=row['input_ids'], row=row)
        if max(args.lengths) > len(row['input_ids']):
            raise ValueError('fixed request shorter than diagnostic length')
        tokens = torch.tensor(row['input_ids'], device='cuda')[None]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for length in args.lengths:
                ids = tokens[:, :length]
                trace, baseline_hidden, baseline_logits = trace_baseline(model, ids)
                prefix = f'T{length}'
                for i, layer in enumerate(model.model.layers):
                    fixed_layer(recorder, f'{prefix}/fixed/layer{i:02d}', layer, trace[i], adapter, operator, kind, axis)
                actual, cache = cascade(recorder, prefix+'/r1_cascade', model, ids, trace, baseline_hidden, baseline_logits, adapter)
                proposed, proposed_cache = cascade(recorder, prefix+'/native_output_probe', model, ids, trace, baseline_hidden, baseline_logits, probe)
                # Deterministic teacher forcing isolates cache error from greedy divergence.
                for step in range(args.decode):
                    nxt = baseline_logits[:, -1].argmax(-1)[:, None]
                    ids = torch.cat((ids, nxt), 1)
                    next_hidden = model(ids)
                    recorder.emit(f'{prefix}/frozen_prefix_stability_{step}', next_hidden[:, :-1], baseline_hidden)
                    baseline_hidden = next_hidden
                    baseline_logits = model.lm_head(next_hidden[:, -1:]).float()
                    actual, cache = adapter(nxt, cache=cache, logits_to_keep=1)
                    proposed, proposed_cache = probe(nxt, cache=proposed_cache, logits_to_keep=1)
                    for name, value in [('r1', actual), ('native_output_probe', proposed)]:
                        recorder.emit(f'{prefix}/decode{step}/{name}', value, baseline_logits,
                                      **logits_details(value, baseline_logits))
                del trace
        report['status'] = 'diagnosed'
        report['comparisons'] = len(recorder.rows)
        report['numeric_gate_exceedances'] = sum(not r['within_frozen_model_numeric_gate'] for r in recorder.rows)
        report['top1_mismatches'] = [r['label'] for r in recorder.rows if r.get('top1_equal') is False]
    except Exception as error:
        report['status'], report['error'] = 'diagnostic_execution_failed', repr(error)
        raise
    finally:
        recorder.stream.close()
        report['finished'] = time.time()
        report['comparisons_file_sha256'] = fingerprint(directory/'comparisons.jsonl')
        (directory/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
