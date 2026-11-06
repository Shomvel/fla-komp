"""One-pass serial scan vs two-level scan: 8k / main / train timing and agreement."""
import itertools, torch
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms


def rms(a, b):
    return float(((a.double()-b.double()).square().mean()/(b.double().square().mean()+1e-30)).sqrt())


for cfg, unroll in (((1, 8192, 64, 128, 128, 64), 2), ((2, 2048, 4, 64, 64, 32), 10), ((64, 192, 2, 64, 64, 32), 10)):
    data, slots = make(*cfg)
    for name in ('block', 'block_slot'):
        base = None
        for serial, bv, stages, w in [(0, 64, 2, 4)]+list(itertools.product((1,), (32, 64), (1, 2, 3), (4, 8))):
            BF.SCAN_SERIAL, BF.SCAN_BV, BF.SCAN_STAGES, BF.WARPS['scan'] = serial, bv, stages, w
            x = [y.detach().clone().requires_grad_() for y in data[name]]
            f = lambda: run(name, x, slots, 64)
            try:
                with torch.no_grad():
                    fw = graph_ms(f, unroll=unroll, replays=5)
                    torch.manual_seed(1)
                    ws = [torch.randn_like(o)*.01 for o in f()]
                step = lambda: torch.autograd.grad(f(), x, ws)
                fb = graph_ms(step, unroll=unroll, replays=5)
                res = [o.detach() for o in f()]+[g.detach() for g in step()]
                base = base or res
                err = max(rms(a, b) for a, b in zip(res, base))
                print(cfg[:2], name, f'serial={serial} bv={bv} stages={stages} warps={w}', f'fwd {fw:.3f} fwd+bwd {fb:.3f} rms-vs-two-level {err:.1e}', flush=True)
            except Exception as e:
                print(cfg[:2], name, f'serial={serial} bv={bv} stages={stages} warps={w}', 'error', repr(e)[:150], flush=True)
BF.SCAN_SERIAL, BF.SCAN_BV, BF.SCAN_STAGES, BF.WARPS['scan'] = 0, 64, 2, 4
