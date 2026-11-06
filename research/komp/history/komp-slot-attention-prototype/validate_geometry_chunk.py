"""Chunked inverse-carry geometry: acceptance against the frozen baseline and Torch/FP64 oracles."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import torch
import triton
from komp_slot.geometry import triton_geometry_serial
from komp_slot.geometry_chunk import triton_geometry_chunked
from komp_slot.reference import prepare_geometry
from komp_slot.triton_chunk import triton_chunk
from validate_geometry_fast import inputs, clone, call, errors, check
from komp_bench.timing import graph_ms

ROOT = Path(__file__).resolve().parent
BLOCKS = (8, 16, 32)
MODES = {'baseline': triton_geometry_serial,
         **{f'chunk{b}': (lambda b: lambda *a, **kw: triton_geometry_chunked(*a, **kw, block=b))(b) for b in BLOCKS}}


def validate(report):
    cases = [(1, 1, 7, torch.float32, 'random'), (19, 5, 33, torch.float32, 'noncontiguous'),
             (33, 16, 64, torch.float32, 'repeated'), (65, 32, 64, torch.float32, 'random'),
             (17, 64, 64, torch.float32, 'random'), (35, 32, 64, torch.bfloat16, 'random'),
             (192, 32, 64, torch.float32, 'random'), (7, 8, 16, torch.float32, 'zero'),
             (129, 32, 64, torch.float32, 'cyclic'), (300, 16, 64, torch.float32, 'random')]
    for t, m, d, dtype, case in cases:
        xs, slots = inputs(t, m, d, dtype, case='random' if case == 'cyclic' else case)
        if case == 'cyclic':
            slots = (torch.arange(t, device='cuda') % m)[None, :, None].expand_as(slots).contiguous()
        xr = clone(xs)
        ref = call(prepare_geometry, xr, slots)
        weights = [torch.randn_like(x) for x in ref]
        rg = torch.autograd.grad(ref, xr, weights)
        for mode, fn in MODES.items():
            x = clone(xs)
            out = call(fn, x, slots)
            grads = torch.autograd.grad(out, x, weights)
            tag = f'{mode}/{t}/{m}/{dtype}/{case}'
            for i, (a, b) in enumerate(zip(out, ref)):
                check(tag+f'/out{i}', a, b, report)
            for i, (a, b) in enumerate(zip(grads, rg)):
                check(tag+f'/grad{i}', a, b, report, 3e-3 if dtype == torch.bfloat16 else 2e-4,
                      1e-2 if dtype == torch.bfloat16 else 1e-3)
        print('PASS geometry', t, m, str(dtype), case, flush=True)
    for t, m, d, seed_case in ((9, 16, 32, 'near_duplicate'), (64, 32, 64, 'near_duplicate'), (130, 64, 64, 'near_duplicate')):
        xs, slots = inputs(t, m, d, case=seed_case)
        xr = clone(xs, torch.float64)
        ref = call(prepare_geometry, xr, slots)
        weights = [torch.randn_like(x).float() for x in ref]
        rg = torch.autograd.grad(ref, xr, [w.double() for w in weights])
        baseline = None
        for mode, fn in MODES.items():
            x = clone(xs)
            out = call(fn, x, slots)
            grads = torch.autograd.grad(out, x, weights)
            measured = []
            for a, b in zip((*out, *grads), (*ref, *rg)):
                assert bool(torch.isfinite(a).all())
                measured.append(errors(a, b))
            if mode == 'baseline':
                baseline = measured
            else:
                for i, (err, base) in enumerate(zip(measured, baseline)):
                    assert err['rms_relative'] <= max(1e-3, base['rms_relative']*1.25), (t, m, mode, i, err, base)
            report['conditioning'].append({'T': t, 'M': m, 'D': d, 'mode': mode, 'errors_vs_fp64': measured,
                                           'criterion': 'RMS relative <= max(.001, 1.25 * baseline error)'})
        print('PASS conditioning', t, m, d, flush=True)
    xs, slots = inputs(67, 32)
    for mode, fn in MODES.items():
        x = clone(xs)
        full = call(fn, x, slots)
        first = fn(x[0][:, :29], x[1][:, :29], slots[:, :29], x[2])
        second = fn(x[0][:, 29:], x[1][:, 29:], slots[:, 29:], first[3])
        for i in range(3):
            check(mode+f'/stream/out{i}', torch.cat((first[i], second[i]), 1), full[i], report)
        check(mode+'/stream/final', second[3], full[3], report)
        gf = torch.autograd.grad(sum(o.square().sum() for o in full), x, retain_graph=True)
        gs = torch.autograd.grad(sum(torch.cat((first[i], second[i]), 1).square().sum() for i in range(3))+second[3].square().sum(), x)
        for i, (a, b) in enumerate(zip(gf, gs)):
            check(mode+f'/stream/grad{i}', a, b, report)
    print('PASS stream', flush=True)


def timeit(fn):
    # Bounded-unroll CUDA Graph replay; callers must not keep eager graphs of the timed leaves alive.
    return graph_ms(fn)


def benchmark(report, quick):
    cases = [(2048, 32)] if quick else [(256, 32), (1024, 16), (1024, 32), (2048, 32), (4096, 32), (1024, 64)]
    for t, m in cases:
        xs, slots = inputs(t, m, dtype=torch.bfloat16, b=2, h=4)
        slots = (torch.arange(t, device='cuda') % m)[None, :, None].expand_as(slots).contiguous()
        row = {'B': 2, 'T': t, 'H': 4, 'D': 64, 'M': m, 'routing': 'cyclic', 'modes': {}}
        for mode, fn in MODES.items():
            gx = clone(xs)
            with torch.no_grad():
                gout = call(fn, gx, slots)
                fwd = timeit(lambda: call(fn, gx, slots))
            gw = [torch.randn_like(o)*.01 for o in gout]
            train = timeit(lambda: torch.autograd.grad(call(fn, gx, slots), gx, gw))
            value = torch.randn(2, t, 4, 64, device='cuda', dtype=torch.bfloat16)*.2
            alpha = torch.full((2, t, 4), .98, device='cuda')
            beta = torch.full_like(alpha, .2)
            init = torch.randn(2, 4, m, 64, device='cuda')*.05
            fx = clone(list(xs)+[value, alpha, beta, init])

            def full(z):
                h, p, r, k = call(fn, z[:3], slots)
                out, s = triton_chunk(h, p, r, z[3], z[4], z[5], slots, z[6])
                return out, k, s
            with torch.no_grad():
                fw = [torch.randn_like(o)*.01 for o in full(fx)]
            full_train = timeit(lambda: torch.autograd.grad(full(fx), fx, fw))
            row['modes'][mode] = {'geometry_fwd_ms': fwd, 'geometry_fwd_bwd_ms': train, 'full_fwd_bwd_ms': full_train}
            print('BENCH', t, m, mode, json.dumps(row['modes'][mode]), flush=True)
        report['benchmark'].append(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default=str(ROOT/'evidence/geometry-chunk-validation.json'))
    p.add_argument('--skip-validate', action='store_true')
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {'checks': [], 'conditioning': [], 'benchmark': [],
              'environment': {'torch': str(torch.__version__), 'triton': triton.__version__,
                              'python': platform.python_version(), 'gpu': torch.cuda.get_device_name()},
              'sources': {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in (Path(__file__), ROOT/'komp_slot/geometry.py', ROOT/'komp_slot/geometry_chunk.py')}}
    try:
        if not args.skip_validate:
            validate(report)
        if args.benchmark:
            benchmark(report, args.quick)
        report['status'] = 'passed'
    except Exception as e:
        report['status'] = 'failed'
        report['error'] = repr(e)
        raise
    finally:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2)+'\n')
        print('REPORT', report['status'], len(report['checks']), flush=True)


if __name__ == '__main__':
    main()
