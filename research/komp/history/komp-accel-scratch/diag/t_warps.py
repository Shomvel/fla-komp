"""8k one-at-a-time sweep of per-kernel warps and the key-gradient block width."""
import torch
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


default = dict(BF.WARPS)
for name in ('block', 'block_slot'):
    print(name, 'default', f'{timed(name):.3f}', flush=True)
for key in ('att', 'proj', 'ut', 'compose', 'mapgrad', 'proj_bwd', 'chunk_bwd', 'keygrad'):
    for w in (2, 4, 8):
        if w == default[key]:
            continue
        BF.WARPS[key] = w
        try:
            print(key, w, ' '.join(f'{timed(n):.3f}' for n in ('block', 'block_slot')), flush=True)
        except Exception as e:
            print(key, w, 'error', repr(e)[:100], flush=True)
    BF.WARPS[key] = default[key]
for kb in (64, 128):
    for w in (4, 8):
        BF.KEY_BLOCK, BF.WARPS['keygrad'] = kb, w
        try:
            print('key_block', kb, 'warps', w, ' '.join(f'{timed(n):.3f}' for n in ('block', 'block_slot')), flush=True)
        except Exception as e:
            print('key_block', kb, w, 'error', repr(e)[:100], flush=True)
