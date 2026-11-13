"""Isolated single-GPU acceptance; never writes training or evaluation outputs."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import torch

from komp_inference import compressed_state, initialize_state, recurrent_komp
from komp_inference.model import RecurrentKOMPLM
from verify_sources import verify_candidate, verify_model, plan

MODELS = {
    'block': ('komp-flame-340m-20c-20261106/migration-fla-mb16/runtime', 'block', 'slot', False),
    'exact': ('komp-flame-340m-20c-20261106/migration-fla-mb16/runtime', 'exact', 'slot', False),
    'block123': ('komp-kda-options-ff2272-340m-20c-20261106', 'block', 'slot', False),
    'exact123': ('komp-kda-options-ff2272-340m-20c-20261106', 'exact', 'slot', False),
    'learned': ('komp-gdn-slot-learned-340m-20c-20261107', 'block', 'slot', True),
    'learned123': ('komp-gdn-slot-learned-123-340m-20c-20261107', 'block', 'slot', True),
    'value': ('komp-value-decay-block-340m-20c-20261106', 'block', 'value', False),
    'value_learned123': ('komp-value-decay-learned-123-340m-20c-20261107', 'block', 'value', True),
}


def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def compare(actual, expected, label, relative=.003, absolute=.0003):
    a, e = actual.double(), expected.double()
    difference = a-e
    peak = difference.abs().max().item()
    rmsrel = (difference.square().mean().sqrt()/e.square().mean().sqrt().clamp_min(1e-12)).item()
    result = dict(label=label, absolute=peak, rmsrel=rmsrel,
                  passed=bool(torch.isfinite(a).all()) and (peak < absolute or rmsrel < relative))
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise AssertionError(result)
    return result


@torch.inference_mode()
def frozen_operator(kind, axis, learned):
    module = importlib.import_module('models')
    if not learned and axis == 'slot' and hasattr(module, 'legacy') and hasattr(module.legacy, 'exact_komp'):
        return module.legacy.exact_komp if kind == 'exact' else module.legacy.block_komp_fast
    if axis == 'value' and learned:
        from value_learned import block_komp_value_learned as reference
    elif axis == 'value':
        from fla.ops.komp import block_komp_value_decay as reference
    else:
        from fla.ops.komp import exact_komp, block_komp_fast
        reference = exact_komp if kind == 'exact' else block_komp_fast
    return reference


@torch.inference_mode()
def operators(kind, axis, learned):
    reference = frozen_operator(kind, axis, learned)
    results = []
    configurations = [(1, 33, torch.float32), (2, 129, torch.bfloat16), (1, 259, torch.bfloat16)]
    for b, t, dtype in configurations:
        torch.manual_seed(42)
        h, d, m = 16, 64, 32
        q, k, v = [torch.randn(b, t, h, d, device='cuda', dtype=dtype) for _ in range(3)]
        g = -torch.rand(b, t, h, m if axis == 'slot' else d, device='cuda')*.03
        beta = torch.rand(b, t, h, device='cuda')*.4
        keys = torch.randn(b, h, m, d, device='cuda')
        values = torch.randn(b, h, m, d, device='cuda')*.1
        scores = torch.randint(0, 3, (b, t, h), device='cuda').float()
        initial = torch.zeros(b, h, m, device='cuda')
        options = dict(scores=scores, initial_scores=initial, evict='learned') if learned else {}
        if kind == 'exact':
            out, final, _ = reference(q, k, v, g, beta, (keys, values), eviction='fifo')
        else:
            out, final = reference(q, k, v, g, beta, (keys, values), chunk_size=128, **options)
        state = initialize_state(keys, values, chunk_size=1 if kind == 'exact' else 128,
                                 decay_axis=axis, eviction='learned' if learned else 'fifo', initial_scores=initial)
        actual, state = recurrent_komp(q, k, v, g, beta, state, scores=scores, backend='triton')
        terminal = compressed_state(state)
        # native FIFO Block exposes age order; the cache stores physical slots
        if kind == 'block' and not learned:
            age = terminal.births.argsort(dim=-1)
            final = tuple(x.gather(-2, age.argsort(dim=-1)[..., None].expand_as(x)) for x in final)
        tolerance = .003 if dtype == torch.float32 else .025
        label = f'{kind}-{axis}-learned{learned}-B{b}-T{t}-{dtype}'
        for suffix, a, e in [('output', actual, out), ('keys', terminal.keys, final[0]),
                             ('coefficients', terminal.coefficients, final[1])]:
            results.append(compare(a, e, label+'-'+suffix, relative=tolerance))
        # compare fused token update independently to the pure Torch streaming implementation
        initial_state = initialize_state(keys, values, chunk_size=1 if kind == 'exact' else 128,
                                         decay_axis=axis, eviction='learned' if learned else 'fifo', initial_scores=initial)
        oracle, oracle_state = recurrent_komp(q, k, v, g, beta, initial_state, scores=scores)
        results.append(compare(actual, oracle, label+'-torch-stream', relative=tolerance))
        results.append(compare(state.coefficients, oracle_state.coefficients, label+'-open-state', relative=tolerance))
    return results


@torch.inference_mode()
def model_checks(source, kind, axis, checkpoint):
    module = importlib.import_module('models')
    model = module.LM(kind)
    ck = Path(checkpoint)
    saved = torch.load(ck, map_location='cpu', weights_only=False, mmap=True)
    manifest_path = source/'source-manifest.json'
    if saved['source_sha'] != fingerprint(manifest_path):
        raise ValueError('checkpoint source manifest mismatch')
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files'].items():
        if fingerprint(source/name) != expected:
            raise ValueError(f'frozen source changed: {name}')
    if saved['data_sha'] != fingerprint(source/'configs/data-manifest.json'):
        raise ValueError('checkpoint data manifest mismatch')
    model.load_state_dict(saved['model'], strict=True)
    receipt = dict(checkpoint=str(ck), checkpoint_sha=fingerprint(ck), step=saved['step'], tokens=saved['tokens'],
                   source_sha=saved['source_sha'], data_sha=saved['data_sha'])
    del saved
    model = model.cuda().eval()
    adapter = RecurrentKOMPLM(model, decay_axis=axis, backend='triton',
                             prefill_operator=frozen_operator(kind, axis, hasattr(model.model.layers[0].attn, 'score')))
    manifest = Path('/gpfs/users/linlizhi/komp-flame-downstream-eval-20261107/requests/niah_single_1_1024.jsonl')
    row = json.loads(manifest.open().readline())
    tokens = torch.tensor(row['input_ids'], device='cuda')[None]
    checks = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for length in (31, 127, 128, 129, 257):
            ids = tokens[:, :length]
            baseline = model.lm_head(model(ids)[:, -1:]).float()
            actual, cache = adapter(ids, logits_to_keep=1)
            checks.append(compare(actual, baseline, f'model-prefill-{length}'))
            top1_equal = bool((actual.argmax(-1) == baseline.argmax(-1)).all())
            if not top1_equal:
                raise AssertionError(dict(label=f'prefill-top1-{length}', passed=False))
            for step in range(4):
                nxt = baseline[:, -1].argmax(-1)[:, None]
                ids = torch.cat((ids, nxt), 1)
                baseline = model.lm_head(model(ids)[:, -1:]).float()
                actual, cache = adapter(nxt, cache=cache, logits_to_keep=1)
                checks.append(compare(actual, baseline, f'model-decode-{length}-{step}'))
                if not bool((actual.argmax(-1) == baseline.argmax(-1)).all()):
                    raise AssertionError(dict(label=f'decode-top1-{length}-{step}', passed=False))
            receipt['cache_bytes'] = cache.nbytes()
    return checks, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--base', default='/gpfs/users/linlizhi')
    parser.add_argument('--operators-only', action='store_true')
    parser.add_argument('--model-only', action='store_true')
    parser.add_argument('--operator-receipt')
    parser.add_argument('--evidence-dir', default=str(Path(__file__).parent/'evidence'))
    args = parser.parse_args()
    folder, kind, axis, learned = MODELS[args.model]
    source = Path(args.base)/folder
    candidate_source = verify_candidate()
    frozen = verify_model(args.model, args.base, checkpoint=not args.operators_only)
    if args.operators_only and args.model_only:
        raise ValueError('operator-only and model-only are mutually exclusive')
    if args.model_only:
        if args.operator_receipt is None:
            raise ValueError('model-only requires a passing operator receipt')
        previous = json.loads(Path(args.operator_receipt).read_text())
        if (previous['status'] != 'passed' or previous['model'] != args.model or not previous.get('operators')
                or previous['candidate_source'] != candidate_source or previous['source'] != str(source)):
            raise ValueError('operator receipt does not bind this model and complete source')
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    target = evidence/f'gpu-{args.model}.json'
    # reserve the final evidence path before any CUDA work, preventing accidental duplicate runs
    with target.open('x') as stream:
        stream.write(json.dumps(dict(status='reserved', model=args.model, candidate_source=candidate_source))+'\n')
    sys.path.insert(0, str(source))
    importlib.import_module('models')
    torch.set_num_threads(2)
    # match the original evaluator's Torch defaults; explicit Triton precisions are unchanged
    torch.manual_seed(42)
    report = dict(model=args.model, status='running', torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  torch_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  source=str(source), started=time.time(), candidate_source=candidate_source, frozen=frozen)
    try:
        if not args.model_only:
            report['operators'] = operators(kind, axis, learned)
        else:
            report['operator_receipt_sha256'] = fingerprint(Path(args.operator_receipt))
        if not args.operators_only:
            report['model_checks'], report['checkpoint'] = model_checks(source, kind, axis, frozen['checkpoint'])
        report['status'] = 'passed'
    except Exception as error:
        report['status'], report['error'] = 'failed', repr(error)
        raise
    finally:
        report['finished'] = time.time()
        target.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
