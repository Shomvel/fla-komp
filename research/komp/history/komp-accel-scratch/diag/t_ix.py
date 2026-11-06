"""Warps for the learned-path proj / proj_bwd variants, and DOV_BF16 per decay mode (chunk_bwd_slot = 8)."""
import itertools, torch
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms

data, slots = make(1, 8192, 64, 128, 128, 64)


def timed(name, sc=None):
    x = [y.detach().clone().requires_grad_() for y in data[name]]
    f = lambda: run(name, x, slots, 64, sc)
    with torch.no_grad():
        ws = [torch.randn_like(o)*.01 for o in f()]
    return graph_ms(lambda: torch.autograd.grad(f(), x, ws), unroll=2, replays=5)


for wp, wb in itertools.product((4, 8), (4, 8)):
    BF.WARPS['proj_ix'], BF.WARPS['proj_bwd_ix'] = wp, wb
    print(f'learned proj_ix={wp} proj_bwd_ix={wb} fwd+bwd {timed("block_slot", data["scores"]):.3f}', flush=True)
BF.WARPS['proj_ix'], BF.WARPS['proj_bwd_ix'] = 4, 4
for dov in (False, True):
    BF.DOV_BF16 = dov
    print(f'DOV_BF16={dov} scalar {timed("block"):.3f} slot {timed("block_slot"):.3f}', flush=True)
