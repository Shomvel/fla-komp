"""A/B of TAPE_BF16 and SHARE_GRAM: timing (8k, main) and agreement with both flags off."""
import itertools, torch
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms

FLAGS = ('TAPE_BF16', 'SHARE_GRAM')


def rms(a, b):
    return float(((a.double()-b.double()).square().mean()/(b.double().square().mean()+1e-30)).sqrt())


for cfg, unroll in (((1, 8192, 64, 128, 128, 64), 2), ((2, 2048, 4, 64, 64, 32), 10)):
    data, slots = make(*cfg)
    for label, name, sc in (('scalar', 'block', None), ('slot', 'block_slot', None), ('learned', 'block_slot', data['scores'])):
        base = None
        for vals in itertools.product((False, True), repeat=len(FLAGS)):
            for f, v in zip(FLAGS, vals):
                setattr(BF, f, v)
            x = [y.detach().clone().requires_grad_() for y in data[name]]
            f = lambda: run(name, x, slots, 64, sc)
            with torch.no_grad():
                fw = graph_ms(f, unroll=unroll, replays=5)
                torch.manual_seed(1)
                ws = [torch.randn_like(o)*.01 for o in f()]
            step = lambda: torch.autograd.grad(f(), x, ws)
            fb = graph_ms(step, unroll=unroll, replays=5)
            res = [o.detach() for o in f()]+[g.detach() for g in step()]
            base = base or res
            err = max(rms(a, b) for a, b in zip(res, base))
            print(cfg[1], label, dict(zip(FLAGS, vals)), f'fwd {fw:.3f} fwd+bwd {fb:.3f} max-rms-vs-off {err:.1e}', flush=True)
