"""Block-KOMP Triton acceptance against the FP64 token reference (FLA-style RMS tolerances)."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import torch
import triton
from komp_slot.block_komp import block_komp as block_komp_v0
from komp_slot.block_fast import block_komp_fast
block_komp = block_komp_fast
from komp_slot.block_reference import block_komp_reference
from komp_slot.block_flex import block_komp_flex

ROOT = Path(__file__).resolve().parent
REPORT = {'checks': []}


def rms(got, ref):
    got, ref = got.double(), ref.double()
    return float((got-ref).square().mean().sqrt()/(ref.square().mean().sqrt()+1e-30))


def check(name, got, ref, tol):
    assert bool(torch.isfinite(got).all()), name
    err = rms(got, ref)
    REPORT['checks'].append({'name': name, 'rms_relative': err, 'max_abs': float((got.double()-ref.double()).abs().max()), 'tolerance': tol})
    assert err <= tol, (name, err, tol)


def inputs(b, t, h, dk, dv, m, dtype, seed, zero=False, slot=False):
    torch.manual_seed(seed)
    q = torch.randn(b, t, h, dk, device='cuda').to(dtype)
    k = torch.randn(b, t, h, dk, device='cuda').to(dtype)
    v = (torch.randn(b, t, h, dv, device='cuda')*.5).to(dtype)
    g = torch.log(torch.rand(b, t, h, m, device='cuda')*.5+.5) if slot else torch.log(torch.rand(b, t, h, device='cuda')*.1+.9)
    beta = torch.rand(b, t, h, device='cuda')*.5+.1
    if zero:
        g[:, 1] = -30.
        beta[:, 2] = 0.
    k0 = torch.randn(b, h, m, dk, device='cuda')
    v0 = torch.randn(b, h, m, dv, device='cuda')*.1
    return [q, k, v, g, beta, k0, v0]


def run(fn, xs, c):
    o, (ks, vs) = fn(*xs[:5], (xs[5], xs[6]), chunk_size=c)
    return o, ks, vs


def learned():
    # Learned-score eviction (indexed Triton path) against the FP64 flex implementation.
    cases = [(1, 100, 2, 32, 32, 16, 32), (2, 77, 2, 16, 24, 8, 16), (1, 130, 1, 64, 64, 64, 64),
             (1, 150, 2, 32, 48, 32, 64), (2, 96, 2, 128, 128, 64, 64), (1, 70, 2, 24, 16, 48, 32)]
    for i, (b, t, h, dk, dv, m, c) in enumerate(cases):
        for slot in (False, True):
            for dtype in (torch.float32, torch.bfloat16):
                xs = inputs(b, t, h, dk, dv, m, dtype, 40+i, zero=i % 2 == 1, slot=slot)
                torch.manual_seed(90+i)
                tied = i % 2 == 0
                sc = torch.randint(0, 3, (b, t, h), device='cuda').float() if tied else torch.randn(b, t, h, device='cuda')
                isc = torch.randint(0, 3, (b, h, m), device='cuda').float() if tied else torch.randn(b, h, m, device='cuda')
                got_x = [x.detach().clone().requires_grad_() for x in xs]
                ref_x = [x.detach().double().requires_grad_() for x in xs]
                o, (fk, fv) = block_komp(*got_x[:5], (got_x[5], got_x[6]), chunk_size=c, evict='learned',
                                         scores=sc, initial_scores=isc)
                ro, (rk, rv) = block_komp_flex(*ref_x[:5], (ref_x[5], ref_x[6]), chunk_size=c, evict='learned',
                                               scores=sc.double(), initial_scores=isc.double())
                tol = 5e-3 if dtype == torch.float32 else 1e-2
                tag = f'learned/{(b, t, h, dk, dv, m, c)}/{"slot" if slot else "scalar"}/{"tied" if tied else "random"}/{dtype}'
                for name, x, y in zip(('out', 'keys', 'values'), (o, fk, fv), (ro, rk, rv)):
                    check(f'{tag}/{name}', x, y, tol)
                ws = [torch.randn_like(x) for x in (o, fk, fv)]
                ga = torch.autograd.grad((o, fk, fv), got_x, ws)
                gb = torch.autograd.grad((ro, rk, rv), ref_x, [w.double() for w in ws])
                for name, x, y in zip(('dq', 'dk', 'dv', 'dg', 'dbeta', 'dk0', 'dv0'), ga, gb):
                    check(f'{tag}/{name}', x, y, tol)
                print('PASS', tag, flush=True)
    # Scores equal to insertion time reproduce FIFO through the indexed path.
    xs = inputs(1, 200, 2, 32, 32, 32, torch.float32, 7, slot=True)
    sc = torch.arange(200, device='cuda').float()[None, :, None].expand(1, 200, 2).contiguous()
    isc = (torch.arange(32, device='cuda').float()-32)[None, None].expand(1, 2, 32).contiguous()
    a = [x.detach().clone().requires_grad_() for x in xs]
    f = [x.detach().clone().requires_grad_() for x in xs]
    oa, _ = block_komp(*a[:5], (a[5], a[6]), chunk_size=64, evict='learned', scores=sc, initial_scores=isc)
    of, _ = block_komp(*f[:5], (f[5], f[6]), chunk_size=64)
    check('learned_as_fifo/out', oa, of, 1e-4)
    w = torch.randn_like(oa)
    for name, x, y in zip(('dq', 'dk', 'dv', 'dg', 'dbeta', 'dk0', 'dv0'), torch.autograd.grad(oa, a, w), torch.autograd.grad(of, f, w)):
        check(f'learned_as_fifo/{name}', x, y, 1e-4)
    print('PASS learned == fifo when scores are insertion times', flush=True)


def validate():
    cases = [(1, 1, 1, 8, 8, 4, 16), (2, 37, 2, 16, 24, 8, 16), (1, 100, 2, 32, 32, 16, 32),
             (1, 130, 1, 64, 64, 64, 64), (1, 200, 2, 32, 48, 32, 64), (2, 96, 2, 128, 128, 64, 64),
             (1, 75, 3, 33, 17, 5, 16), (1, 90, 2, 32, 32, 48, 32)]
    for i, (b, t, h, dk, dv, m, c) in enumerate(cases):
      for slot in ((False, True) if block_komp is block_komp_fast else (False,)):
        for dtype in (torch.float32, torch.bfloat16):
            xs = inputs(b, t, h, dk, dv, m, dtype, i, zero=i % 2 == 1, slot=slot)
            got_x = [x.detach().clone().requires_grad_() for x in xs]
            ref_x = [x.detach().double().requires_grad_() for x in xs]
            got = run(block_komp, got_x, c)
            ref = run(block_komp_reference, ref_x, c)
            tol = 5e-3 if dtype == torch.float32 else 1e-2
            tag = f'{(b, t, h, dk, dv, m, c)}/{"slot" if slot else "scalar"}/{dtype}'
            for name, x, y in zip(('out', 'keys', 'values'), got, ref):
                check(f'{tag}/{name}', x, y, tol)
            ws = [torch.randn_like(x) for x in got]
            ga = torch.autograd.grad(got, got_x, ws)
            gb = torch.autograd.grad(ref, ref_x, [w.double() for w in ws])
            for name, x, y in zip(('dq', 'dk', 'dv', 'dg', 'dbeta', 'dk0', 'dv0'), ga, gb):
                check(f'{tag}/{name}', x, y, tol)
            print('PASS', tag, max(c['rms_relative'] for c in REPORT['checks'] if c['name'].startswith(tag)), flush=True)
    # Ill-conditioned dictionaries (near-duplicate keys), FP32 inputs: the same BF16-aligned FLA-level
    # RMS tolerance as normal FP32 cases; the error ratio to the all-tf32x3 v0 kernel is recorded.
    for case_m, case_c in ((16, 32), (64, 64)):
        xs = inputs(1, 160, 2, 64, 64, case_m, torch.float32, 21)
        center = torch.randn(1, 1, 2, 64, device='cuda')
        xs[1] = center+.02*xs[1]
        xs[5] = center.transpose(1, 2)+.02*xs[5]
        ref_x = [x.detach().double().requires_grad_() for x in xs]
        ref = run(block_komp_reference, ref_x, case_c)
        ws = [torch.randn_like(x).float() for x in ref]
        gref = torch.autograd.grad(ref, ref_x, [w.double() for w in ws])
        errs = {}
        for name, fn in (('v0', block_komp_v0), ('candidate', block_komp)):
            x = [y.detach().clone().requires_grad_() for y in xs]
            out = run(fn, x, case_c)
            gr = torch.autograd.grad(out, x, ws)
            errs[name] = [rms(a, b) for a, b in zip((*out, *gr), (*ref, *gref))]
            assert all(bool(torch.isfinite(a).all()) for a in (*out, *gr)), name
        for i, (e, base) in enumerate(zip(errs['candidate'], errs['v0'])):
            assert e <= max(5e-3, 1.25*base), ('near_duplicate', case_m, i, e, base)
        REPORT['checks'].append({'name': f'near_duplicate/M{case_m}/C{case_c}', 'candidate_rms': errs['candidate'],
                                 'v0_rms': errs['v0'], 'max_ratio_to_v0': max(e/b for e, b in zip(errs['candidate'], errs['v0'])),
                                 'tolerance': 'max(5e-3, 1.25*v0)'})
        print('PASS near-duplicate', case_m, case_c, [f'{e:.1e}' for e in errs['candidate']], flush=True)
    # Chunk-aligned streaming is exact; future inputs do not change earlier outputs.
    xs = inputs(1, 128, 2, 32, 32, 16, torch.float32, 9)
    full = run(block_komp, xs, 32)
    first = block_komp(*[x[:, :64] for x in xs[:5]], (xs[5], xs[6]), chunk_size=32)
    second = block_komp(*[x[:, 64:] for x in xs[:5]], first[1], chunk_size=32)
    # Split and unsplit calls group the chunk scan differently; tf32 re-association only.
    check('stream/out', torch.cat((first[0], second[0]), 1), full[0], 1e-3)
    check('stream/values', second[1][1], full[2], 1e-3)
    ys = [x.clone() for x in xs]
    ys[2][:, 100:] += 1.
    pert = run(block_komp, ys, 32)
    assert torch.equal(pert[0][:, :96], full[0][:, :96]), 'chunk-local causality'
    REPORT['checks'].append({'name': 'causality/earlier_chunks_unchanged', 'passed': True})
    print('PASS stream and causality', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default=str(ROOT/'evidence/block-komp-validation.json'))
    p.add_argument('--impl', choices=('fast', 'v0'), default='fast')
    args = p.parse_args()
    global block_komp
    block_komp = block_komp_fast if args.impl == 'fast' else block_komp_v0
    REPORT['impl'] = args.impl
    torch.backends.cuda.matmul.allow_tf32 = False
    REPORT['environment'] = {'torch': str(torch.__version__), 'triton': triton.__version__,
                             'python': platform.python_version(), 'gpu': torch.cuda.get_device_name()}
    REPORT['sources'] = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in (Path(__file__), ROOT/'komp_slot/block_komp.py', ROOT/'komp_slot/block_fast.py', ROOT/'komp_slot/block_reference.py')}
    try:
        if block_komp is block_komp_fast:
            learned()
        validate()
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
