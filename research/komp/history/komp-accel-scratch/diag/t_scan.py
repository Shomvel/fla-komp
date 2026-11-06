"""8k tuning sweep: scan group size / value tile / warps, and dict_scan warps (learned)."""
import itertools, torch
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms

data, slots = make(1, 8192, 64, 128, 128, 64)


def timed(name, sc=None):
    x = [y.detach().clone().requires_grad_() for y in data[name]]
    f = lambda: run(name, x, slots, 64, sc)
    with torch.no_grad():
        fw = graph_ms(f, unroll=2, replays=5)
        ws = [torch.randn_like(o)*.01 for o in f()]
    fb = graph_ms(lambda: torch.autograd.grad(f(), x, ws), unroll=2, replays=5)
    return fw, fb


for grp, bv, w in itertools.product((0, 4, 16), (32, 64, 128), (2, 4, 8)):
    BF.SCAN_GROUP, BF.SCAN_BV, BF.WARPS['scan'] = grp, bv, w
    try:
        fw, fb = timed('block')
        print(f'scan group={grp} bv={bv} warps={w} fwd {fw:.3f} fwd+bwd {fb:.3f}', flush=True)
    except Exception as e:
        print(f'scan group={grp} bv={bv} warps={w} error {repr(e)[:120]}', flush=True)
BF.SCAN_GROUP, BF.SCAN_BV, BF.WARPS['scan'] = 0, 64, 4
for w in (1, 2, 4, 8):
    BF.WARPS['dict'] = w
    fw, fb = timed('block_slot', data['scores'])
    print(f'learned dict warps={w} fwd {fw:.3f} fwd+bwd {fb:.3f}', flush=True)
