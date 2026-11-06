"""OVERLAP_DICT on/off for learned eviction: timing (8k / main / train) and forward bitwise agreement."""
import torch
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms

for cfg, unroll in (((1, 8192, 64, 128, 128, 64), 2), ((2, 2048, 4, 64, 64, 32), 10), ((64, 192, 2, 64, 64, 32), 10)):
    data, slots = make(*cfg)
    base = None
    for ov in (False, True):
        BF.OVERLAP_DICT = ov
        x = [y.detach().clone().requires_grad_() for y in data['block_slot']]
        f = lambda: run('block_slot', x, slots, 64, data['scores'])
        with torch.no_grad():
            fw = graph_ms(f, unroll=unroll, replays=5)
            torch.manual_seed(1)
            ws = [torch.randn_like(o)*.01 for o in f()]
            outs = [o.detach() for o in f()]
        fb = graph_ms(lambda: torch.autograd.grad(f(), x, ws), unroll=unroll, replays=5)
        base = base or outs
        print(cfg[:2], f'overlap={ov} fwd {fw:.3f} fwd+bwd {fb:.3f} forward bitwise {all(torch.equal(a, b) for a, b in zip(outs, base))}', flush=True)
