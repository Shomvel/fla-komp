"""Matrix-form recurrence backward: acceptance against the FP64 token oracle and the frozen all-Triton chunk."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import torch
import triton
import validate as base
from komp_bench.timing import graph_ms
from komp_slot.reference import recurrent_reference
from komp_slot.recurrence import triton_recurrence
from komp_slot.fast_chunk import fast_chunk
import functools
import sys
CANDIDATE = triton_recurrence
from komp_slot.triton_chunk import triton_chunk

ROOT = Path(__file__).resolve().parent
REPORT = base.REPORT


def call(fn, xs, slots, **kw):
    return fn(xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], slots, xs[6], **kw)


def validate():
    cases = [(1, 1, 1, 1, 1), (2, 7, 3, 5, 37), (1, 33, 2, 16, 64), (2, 65, 2, 32, 17),
             (1, 129, 1, 64, 65), (2, 100, 2, 32, 64), (1, 47, 3, 8, 130)]
    for c in (16, 32):
        for n, shape in enumerate(cases):
            xs, slots = base.core_inputs(shape, 'cuda', n)
            if n == 0:
                xs[4].zero_()
                xs[5].fill_(1)
            if n == 1:
                xs[4][:, 0].zero_()
                xs[4][:, -1].fill_(1)
                xs[5][:, 0].zero_()
                slots.fill_(shape[3]-1)
            if n == 5:
                xs[4][:, 3::7].zero_()
                xs[4][:, 5::11].fill_(1)
            cand = base.clone_grad(xs)
            ref = [x.double().requires_grad_() for x in xs]
            out, final = call(CANDIDATE, cand, slots, chunk_size=c)
            ro, rf = call(recurrent_reference, ref, slots)
            tag = f'rec{c}/{shape}'
            base.check(tag+'/output', out, ro)
            base.check(tag+'/state', final, rf)
            go, gf = torch.randn_like(out), torch.randn_like(final)
            cg = torch.autograd.grad((out, final), cand, (go, gf))
            rg = torch.autograd.grad((ro, rf), ref, (go.double(), gf.double()))
            for name, g, e in zip(('h', 'p', 'read', 'v', 'alpha', 'beta', 'initial'), cg, rg):
                base.check(tag+f'/grad_{name}', g, e, atol=6e-5, rtol=6e-4)
            print('PASS', tag, flush=True)
        xs, slots = base.core_inputs((1, 5, 2, 8, 33), 'cuda', 9)
        for which in (0, 1):
            a, b = base.clone_grad(xs), base.clone_grad(xs)
            co = call(CANDIDATE, a, slots, chunk_size=c)[which]
            ro = call(recurrent_reference, b, slots)[which]
            ga = torch.autograd.grad(co.square().sum(), a)
            gb = torch.autograd.grad(ro.square().sum(), b, allow_unused=True)
            for i, (got, r) in enumerate(zip(ga, gb)):
                base.check(f'rec{c}/cotangent{which}/grad{i}', got, torch.zeros_like(got) if r is None else r)
        # Long sequence against the frozen all-Triton chunk implementation, BF16 values.
        xs, slots = base.core_inputs((2, 2048, 4, 32, 64), 'cuda', 11)
        xs[3] = xs[3].bfloat16()
        a, b = base.clone_grad(xs), base.clone_grad(xs)
        out, final = call(CANDIDATE, a, slots, chunk_size=c)
        ro, rf = call(triton_chunk, b, slots, chunk_size=c)
        base.check(f'rec{c}/long/output', out, ro)
        base.check(f'rec{c}/long/state', final, rf)
        go, gf = torch.randn_like(out)*.01, torch.randn_like(final)*.01
        ga = torch.autograd.grad((out, final), a, (go, gf))
        gb = torch.autograd.grad((ro, rf), b, (go, gf))
        for name, g, e in zip(('h', 'p', 'read', 'v', 'alpha', 'beta', 'initial'), ga, gb):
            base.check(f'rec{c}/long/grad_{name}', g.float(), e.float(), atol=6e-5, rtol=6e-4)
        print('PASS long', c, flush=True)


def benchmark(rows):
    from komp_slot.geometry_chunk import triton_geometry_chunked
    from komp_slot.geometry import triton_geometry
    for t, m in ((2048, 32), (1024, 16), (4096, 32)):
        xs, slots = base.core_inputs((2, t, 4, m, 64), 'cuda', 3)
        xs[3] = xs[3].bfloat16()
        row = {'B': 2, 'T': t, 'H': 4, 'M': m, 'V': 64}
        for name, fn in (('triton_chunk', triton_chunk), ('fast16', fast_chunk),
                         ('fast16_bv64', functools.partial(fast_chunk, local_bv=64)),
                         ('fast32', functools.partial(fast_chunk, chunk_size=32)),
                         ('fast32_bv64', functools.partial(fast_chunk, chunk_size=32, local_bv=64))):
            x = base.clone_grad(xs)
            with torch.no_grad():
                row[name+'_fwd'] = graph_ms(lambda: call(fn, x, slots))
                ws = [torch.randn_like(o) for o in call(fn, x, slots)]
            row[name+'_fwd_bwd'] = graph_ms(lambda: torch.autograd.grad(call(fn, x, slots), x, ws))
        # complete operator, cyclic routing, BF16 features
        torch.manual_seed(5)
        q = torch.randn(2, t, 4, 64, device='cuda', dtype=torch.bfloat16)
        k = torch.randn_like(q)
        val = (torch.randn_like(q)*.2)
        alpha = torch.full((2, t, 4), .98, device='cuda')
        beta = torch.full_like(alpha, .2)
        ki = torch.randn(2, 4, m, 64, device='cuda')
        vi = torch.randn(2, 4, m, 64, device='cuda')*.05
        cyc = (torch.arange(t, device='cuda') % m)[None, :, None].expand(2, t, 4).contiguous()
        full_inputs = [q, k, val, alpha, beta, ki, vi]

        def old(z):
            h, p, r, ks = triton_geometry(z[0], z[1], cyc, z[5])
            o, vs = triton_chunk(h, p, r, z[2], z[3], z[4], cyc, z[6])
            return o, ks, vs

        def new(z, block=16, c=16):
            h, p, r, ks = triton_geometry_chunked(z[0], z[1], cyc, z[5], block=block)
            o, vs = fast_chunk(h, p, r, z[2], z[3], z[4], cyc, z[6], chunk_size=c)
            return o, ks, vs
        for name, fn in (('full_old', old), ('full_new', new), ('full_new_c32', lambda z: new(z, 16, 32))):
            x = [y.detach().clone().requires_grad_() for y in full_inputs]
            with torch.no_grad():
                row[name+'_fwd'] = graph_ms(lambda: fn(x))
                ws = [torch.randn_like(o)*.01 for o in fn(x)]
            row[name+'_fwd_bwd'] = graph_ms(lambda: torch.autograd.grad(fn(x), x, ws))
        rows.append(row)
        print('BENCH', json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default=str(ROOT/'evidence/recurrence-validation.json'))
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--skip-validate', action='store_true')
    p.add_argument('--candidate', default='fast')
    args = p.parse_args()
    global CANDIDATE
    CANDIDATE = {'fast': fast_chunk, 'fast64': functools.partial(fast_chunk, local_bv=64), 'matrix': triton_recurrence}[args.candidate]
    REPORT['candidate'] = args.candidate
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    REPORT['environment'] = {'torch': str(torch.__version__), 'triton': triton.__version__,
                             'python': platform.python_version(), 'gpu': torch.cuda.get_device_name()}
    REPORT['sources'] = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in (Path(__file__), ROOT/'komp_slot/recurrence.py', ROOT/'komp_slot/triton_chunk.py')}
    REPORT['benchmark'] = []
    try:
        if not args.skip_validate:
            validate()
        if args.benchmark:
            benchmark(REPORT['benchmark'])
        REPORT['status'] = 'passed'
    except Exception as e:
        REPORT['status'] = 'failed'
        REPORT['error'] = repr(e)
        raise
    finally:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(REPORT, indent=2)+'\n')
        print('REPORT', REPORT['status'], len(REPORT['checks']), flush=True)


if __name__ == '__main__':
    main()
