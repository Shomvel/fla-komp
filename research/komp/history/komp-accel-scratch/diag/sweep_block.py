import itertools, json, torch
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms
data, slots = make(1, 8192, 64, 128, 128, 64)
x = [y.detach().clone().requires_grad_() for y in data['block']]
with torch.no_grad():
    ws = [torch.randn_like(o)*.01 for o in run('block', x, slots, 64)]
def bench(tag):
    with torch.no_grad():
        f = graph_ms(lambda: run('block', x, slots, 64), unroll=2, replays=5)
    fb = graph_ms(lambda: torch.autograd.grad(run('block', x, slots, 64), x, ws), unroll=2, replays=5)
    print(f'{tag:40s} fwd {f:.3f} fwd+bwd {fb:.3f}', flush=True)
bench('default (blocked sweep)')
BF.BLOCK_SWEEP = False
bench('scalar sweep')
BF.BLOCK_SWEEP = True
for w in (2, 8):
    BF.WARPS['proj'] = w; bench(f'blocked proj_warps={w}')
BF.WARPS['proj'] = 4
