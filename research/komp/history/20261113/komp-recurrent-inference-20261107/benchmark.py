"""Correctness-gated, same-GPU cache-free versus recurrent decode timings."""

import argparse
import gc
import importlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from gpu_validate import MODELS, compare, fingerprint, frozen_operator
from komp_inference.model import RecurrentKOMPLM
from verify_sources import verify_candidate, verify_model


def timed(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, time.perf_counter()-start


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--base', default='/gpfs/users/linlizhi')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--acceptance')
    parser.add_argument('--evidence-dir', default=str(Path(__file__).parent/'evidence'))
    args = parser.parse_args()
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    candidate_source = verify_candidate()
    verify_model(args.model, args.base)
    acceptance = json.loads(Path(args.acceptance or evidence/f'gpu-{args.model}.json').read_text())
    if acceptance['status'] != 'passed' or 'checkpoint' not in acceptance:
        raise ValueError('passing operator and real-checkpoint acceptance is required')
    if acceptance['candidate_source'] != candidate_source:
        raise ValueError('acceptance belongs to a different implementation manifest')
    folder, kind, axis, learned = MODELS[args.model]
    source = Path(args.base)/folder
    sys.path.insert(0, str(source))
    module = importlib.import_module('models')
    model = module.LM(kind)
    checkpoint = Path(acceptance['checkpoint']['checkpoint'])
    if fingerprint(checkpoint) != acceptance['checkpoint']['checkpoint_sha']:
        raise ValueError('checkpoint changed after acceptance')
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model = model.cuda().eval()
    candidate = RecurrentKOMPLM(model, decay_axis=axis, backend='triton',
                               prefill_operator=frozen_operator(kind, axis, learned))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cases = [(1, 128), (2, 128), (4, 128), (8, 128), (1, 2048), (2, 2048),
             (4, 2048), (8, 2048), (1, 8192), (2, 8192), (4, 8192), (8, 8192)]
    results = []
    target = evidence/f'benchmark-{args.model}.json'
    if target.exists():
        raise FileExistsError(target)
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for batch, length in cases:
                ids = torch.randint(3, 32000, (batch, length), device='cuda')
                torch.cuda.reset_peak_memory_stats()
                (logits, cache), prefill_cold = timed(lambda: candidate(ids, logits_to_keep=1))
                _, prefill_warm = timed(lambda: candidate(ids, logits_to_keep=1))
                baseline = model.lm_head(model(ids)[:, -1:]).float()
                compare(logits, baseline, f'benchmark-prefill-B{batch}-T{length}')
                baseline_times, recurrent_times = [], []
                for step in range(args.steps):
                    token = baseline[:, -1].argmax(-1)[:, None]
                    ids = torch.cat((ids, token), 1)
                    baseline, latency = timed(lambda: model.lm_head(model(ids)[:, -1:]).float())
                    baseline_times.append(latency)
                    (logits, cache), latency = timed(lambda: candidate(token, cache=cache, logits_to_keep=1))
                    recurrent_times.append(latency)
                    compare(logits, baseline, f'benchmark-decode-B{batch}-T{length}-step{step}')
                    if not bool((logits.argmax(-1) == baseline.argmax(-1)).all()):
                        raise AssertionError('greedy top1 diverged; timing is not accepted')
                row = dict(batch=batch, prompt_length=length, decode_steps=args.steps,
                           prefill_cold_seconds=prefill_cold, prefill_warm_seconds=prefill_warm,
                           baseline_decode_seconds=baseline_times, recurrent_decode_seconds=recurrent_times,
                           baseline_median_seconds=statistics.median(baseline_times[1:]),
                           recurrent_median_seconds=statistics.median(recurrent_times[1:]),
                           generated_tokens_per_second=batch/statistics.median(recurrent_times[1:]),
                           cache_bytes=cache.nbytes(), peak_allocated=torch.cuda.max_memory_allocated())
                results.append(row)
                print(json.dumps(row), flush=True)
                del cache, ids, logits, baseline
                gc.collect()
        status = 'passed'
    except Exception:
        status = 'failed'
        raise
    finally:
        target.write_text(json.dumps(dict(status=status, cases=results, checkpoint=str(checkpoint), candidate_source=candidate_source,
                                         torch=torch.__version__, gpu=torch.cuda.get_device_name()), indent=2)+'\n')


if __name__ == '__main__':
    main()
