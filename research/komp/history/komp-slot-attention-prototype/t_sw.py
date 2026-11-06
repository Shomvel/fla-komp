import torch
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms
data, slots = make(1, 8192, 64, 128, 128, 64)
x = [y.detach().clone().requires_grad_() for y in data['block']]
f = lambda: run('block', x, slots, 64)
with torch.no_grad():
    ws = [torch.randn_like(o)*.01 for o in f()]
def bench(tag):
    with torch.no_grad():
        fw = graph_ms(f, unroll=2, replays=5)
    fb = graph_ms(lambda: torch.autograd.grad(f(), x, ws), unroll=2, replays=5)
    print(f'{tag:30s} fwd {fw:.3f} fwd+bwd {fb:.3f}', flush=True)
bench('default')
for key, vals in (('chunk_fwd', (2, 4)), ('proj_bwd_fused', (2, 4)), ('chunk_bwd', (2, 4)), ('scan', (2, 8))):
    old = BF.WARPS[key]
    for w in vals:
        BF.WARPS[key] = w; bench(f'{key}={w}')
    BF.WARPS[key] = old
