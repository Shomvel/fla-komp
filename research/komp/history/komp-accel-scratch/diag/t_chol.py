"""CHOL_INV: near-duplicate accuracy vs FP64 (against the sweep), then timing and per-kernel time at 8k."""
import collections, itertools, torch
from torch.profiler import profile, ProfilerActivity
from komp_slot import block_fast as BF
from komp_slot.block_komp import block_komp as block_komp_v0
from komp_slot.block_reference import block_komp_reference
from validate_block_komp import inputs, run as vrun, rms
from benchmark_block import make, run
from komp_bench.timing import graph_ms

names = ('out', 'keys', 'values', 'dq', 'dk', 'dv', 'dg', 'dbeta', 'dk0', 'dv0')
for case_m, case_c in ((16, 32), (64, 64)):
    xs = inputs(1, 160, 2, 64, 64, case_m, torch.float32, 21)
    center = torch.randn(1, 1, 2, 64, device='cuda')
    xs[1] = center+.02*xs[1]
    xs[5] = center.transpose(1, 2)+.02*xs[5]
    ref_x = [x.detach().double().requires_grad_() for x in xs]
    ref = vrun(block_komp_reference, ref_x, case_c)
    ws = [torch.randn_like(x).float() for x in ref]
    gref = torch.autograd.grad(ref, ref_x, [w.double() for w in ws])

    def errs(fn):
        x = [y.detach().clone().requires_grad_() for y in xs]
        out = vrun(fn, x, case_c)
        return [rms(a, b) for a, b in zip((*out, *torch.autograd.grad(out, x, ws)), (*ref, *gref))]

    base = errs(block_komp_v0)
    print(case_m, case_c, 'v0  ', ' '.join(f'{n}={e:.1e}' for n, e in zip(names, base)), flush=True)
    for chol in (False, True):
        BF.CHOL_INV = chol
        e = errs(BF.block_komp_fast)
        print(case_m, case_c, f'chol={chol:d}', ' '.join(f'{n}={x:.1e}' for n, x in zip(names, e)),
              f'worst/limit {max(a/max(5e-3, 1.25*b) for a, b in zip(e, base)):.2f}', flush=True)

for cfg, unroll in (((1, 8192, 64, 128, 128, 64), 2), ((2, 2048, 4, 64, 64, 32), 10), ((64, 192, 2, 64, 64, 32), 10)):
    data, slots = make(*cfg)
    for label, name, sc in (('scalar', 'block', None), ('slot', 'block_slot', None), ('learned', 'block_slot', data['scores'])):
        for chol, w in [(False, 1)]+[(True, w) for w in (1, 2, 4)]:
            BF.CHOL_INV, BF.WARPS['chol'] = chol, w
            x = [y.detach().clone().requires_grad_() for y in data[name]]
            f = lambda: run(name, x, slots, 64, sc)
            with torch.no_grad():
                fw = graph_ms(f, unroll=unroll, replays=5)
                ws = [torch.randn_like(o)*.01 for o in f()]
            step = lambda: torch.autograd.grad(f(), x, ws)
            fb = graph_ms(step, unroll=unroll, replays=5)
            extra = ''
            if cfg[1] == 8192 and label == 'scalar':
                step(); torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    step(); torch.cuda.synchronize()
                agg = collections.defaultdict(float)
                for e in prof.events():
                    if e.device_type == torch.autograd.DeviceType.CUDA:
                        agg[e.name] += e.device_time/1000
                extra = ' '.join(f'{k[:11]}={v:.3f}' for k, v in agg.items() if k.startswith(('proj_kernel', 'chol_kernel')))
            print(cfg[:2], label, f'chol={chol} warps={w} fwd {fw:.3f} fwd+bwd {fb:.3f}', extra, flush=True)
BF.CHOL_INV, BF.WARPS['chol'] = False, 1
