"""Matched 8K operator benchmark. Native upstream GDN and exact per-token KOMP."""
import gc
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import statistics
import sys
import types

import torch
import triton

ROOT = Path(__file__).resolve().parent
FLA_ROOT = ROOT / 'fla-source'
for name, rel in [('fla', 'fla'), ('fla.ops', 'fla/ops'), ('fla.modules', 'fla/modules')]:
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(FLA_ROOT / rel)]
    pkg.__file__ = str(FLA_ROOT / rel / '__init__.py')
    pkg.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    pkg.__version__ = '0.6.0'
    sys.modules[name] = pkg
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from exact_komp import exact_komp
from baseline_komp import exact_komp as baseline

REPORT = {'rows': [], 'checks': [], 'status': 'running', 'method': {
    'shape': 'B1 T8192 H2 K64 V64; KOMP M32',
    'state_elements_per_head': {'gdn': 4096, 'komp_keys_plus_coefficients': 4096},
    'inputs': 'Shared raw q/k/v, scalar FP32 log decay and beta; nonzero FP32 initial states',
    'precision': 'GDN native precision; Optimized KOMP FP32 projection/state; baseline previous optimized KOMP FP32 projection/state',
    'scope': 'Operator including normalization, final state and layout; no model projections, conv or optimizer',
    'backward': 'Random upstream gradients on output and final state; all differentiable inputs including initial state',
    'timing': '5 warmups; 3 trials x 20 iterations; CUDA-event median per trial then median across trials; compilation excluded',
    'graph': 'Single-call CUDA graph replay; one private pool per measurement, freed afterward',
    'memory': 'Additional peak allocated bytes above resident inputs/upstream gradients; eager separate from graph',
    'learned': 'Input score top-M and built-in 2*sigmoid(scores) write gate',
    'fla_source': 'Unchanged operator namespace snapshot from prior validated comparison; all Python files hashed',
    'equal_storage': 'Keys+coefficients only; learned metadata additional; different algorithms, no equal-quality claim',
}}


def save():
    (ROOT / 'evidence' / 'comparison.json').write_text(json.dumps(REPORT, indent=2) + '\n')


def finite(xs):
    assert all(x is None or bool(torch.isfinite(x).all()) for x in xs), 'nonfinite tensor'


def events(fn):
    samples = []
    for _ in range(3):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(20):
            fn()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) / 20)
    return {'ms': statistics.median(samples), 'trials_ms': samples}


def measure(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    gc.collect()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - before) / 2**20
    eager = events(fn)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_result = fn()
    torch.cuda.synchronize()
    for _ in range(5):
        graph.replay()
    graph_time = events(graph.replay)
    del static_result, graph
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {'eager': eager, 'cuda_graph': graph_time, 'extra_peak_MiB': peak}


def inputs(t, dtype):
    torch.manual_seed(981)
    r = lambda *shape: torch.randn(*shape, device='cuda')
    q, k, v = [r(1, t, 2, 64).to(dtype) for _ in range(3)]
    g = torch.full((1, t, 2), -.02, device='cuda')
    beta = torch.full_like(g, .2)
    k0, s0, h0 = r(1, 2, 32, 64), .05*r(1, 2, 32, 64), .05*r(1, 2, 64, 64)
    score = r(1, t, 2)
    return [x.requires_grad_() for x in (q, k, v, g, beta, k0, s0, h0, score)]


def run(name, xs, chunk):
    q, k, v, g, beta, k0, s0, h0, score = xs
    if name == 'gdn':
        return chunk_gated_delta_rule(q, k, v, g, beta, initial_state=h0,
                                     output_final_state=True, use_qk_l2norm_in_kernel=True,
                                     chunk_size=64)
    fn=baseline if name.startswith('baseline_') else exact_komp
    eviction=name.removeprefix('baseline_')
    y, (kf, sf), _ = fn(q, k, v, g, beta, (k0, s0),
                       eviction=eviction, scores=score, compute_chunk=chunk)
    return y, kf, sf


def check_gdn():
    # Confirm this pinned GDN backend is numerically functioning before timing.
    for dtype in (torch.float32, torch.bfloat16):
        xs = inputs(65, dtype)
        q, k, v, g, beta, _, _, h0, _ = xs
        out = run('gdn', xs, 64)
        qn, kn = [torch.nn.functional.normalize(x.float(), dim=-1).to(dtype) for x in (q, k)]
        ref = naive_recurrent_gated_delta_rule(qn, kn, v, beta, g, initial_state=h0, output_final_state=True)
        weights = [torch.randn_like(x) for x in out]
        leaves = (q, k, v, g, beta, h0)
        grads = torch.autograd.grad(out, leaves, weights)
        gold = torch.autograd.grad(ref, leaves, weights)
        for label, a, b in zip(['output','state','dq','dk','dv','dg','dbeta','dh0'], (*out, *grads), (*ref, *gold)):
            finite((a, b))
            err = float((a.float()-b.float()).square().mean().sqrt() / b.float().square().mean().sqrt().clamp_min(1e-6))
            tol = .02 if dtype == torch.bfloat16 else .005
            assert err < tol, (label, dtype, err)
            REPORT['checks'].append({'name': f'gdn/{dtype}/{label}', 'rms_relative': err, 'tolerance': tol})
    save()
    print('PASS GDN reference checks', flush=True)


def main():
    torch.set_num_threads(4)
    torch.utils.deterministic.fill_uninitialized_memory = False
    REPORT['gpu'] = torch.cuda.get_device_name()
    REPORT['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(__file__), *sorted((ROOT/'exact_komp').glob('*.py')), *sorted((ROOT/'baseline_komp').glob('*.py')),
                                         *sorted((FLA_ROOT/'fla').rglob('*.py'))]}
    try:
        check_gdn()
        for dtype in (torch.float32, torch.bfloat16):
            xs = inputs(8192, dtype)
            for name, chunk in [('gdn',64),('baseline_fifo',16),('baseline_learned',16),('fifo',16),('fifo',32),('learned',16),('learned',32)]:
                print('START', str(dtype), name, chunk, flush=True)
                leaves = [xs[i] for i in ([0,1,2,3,4,7] if name == 'gdn' else [0,1,2,3,4,5,6,8])]
                def forward():
                    return run(name, xs, chunk)
                with torch.no_grad():
                    out = forward()
                    finite(out)
                    weights = [torch.randn_like(x)*.01 for x in out]
                    del out
                    fwd = measure(forward)
                def both():
                    return torch.autograd.grad(forward(), leaves, weights, allow_unused=True)
                finite(both())
                fb = measure(both)
                row = {'operator': name, 'dtype': str(dtype), 'compute_chunk': chunk,
                       'B':1, 'T':8192, 'H':2, 'K':64, 'V':64, 'M':32 if name != 'gdn' else None,
                       'forward': fwd, 'forward_backward': fb, 'finite_output_and_gradients': True}
                REPORT['rows'].append(row)
                save()
                print('RESULT', json.dumps(row), flush=True)
                del weights
                gc.collect()
            del xs
        REPORT['status'] = 'passed'
    except Exception as exc:
        REPORT['status'] = 'failed'
        REPORT['error'] = repr(exc)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
