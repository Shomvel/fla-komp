"""Registers / spills / shared memory of every compiled block_fast kernel, scalar vs per-slot vs learned 8k,
plus the A/B of the cached-diagonal slot decay and BF16 dOV (bitwise agreement)."""
import torch
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms
import triton

data, slots = make(1, 8192, 64, 128, 128, 64)
for label, name, sc in (('scalar', 'block', None), ('slot', 'block_slot', None), ('learned', 'block_slot', data['scores'])):
    base = None
    for dov in (False, True):
        BF.DOV_BF16 = dov
        x = [y.detach().clone().requires_grad_() for y in data[name]]
        f = lambda: run(name, x, slots, 64, sc)
        with torch.no_grad():
            torch.manual_seed(1)
            ws = [torch.randn_like(o)*.01 for o in f()]
        step = lambda: torch.autograd.grad(f(), x, ws)
        fb = graph_ms(step, unroll=2, replays=5)
        res = [o.detach() for o in f()]+[g.detach() for g in step()]
        base = base or res
        print(label, f'DOV_BF16={dov} fwd+bwd {fb:.3f} bitwise-vs-off {all(torch.equal(a, b) for a, b in zip(res, base))}', flush=True)
for kname in dir(BF):
    kern = getattr(BF, kname)
    if not isinstance(kern, triton.runtime.jit.JITFunction) or not kname.endswith('_kernel'):
        continue
    for dev, caches in kern.device_caches.items():
        for key, k in caches[0].items():
            print(f'{kname:24s} regs {k.n_regs:3d} spills {k.n_spills:4d} smem {k.metadata.shared:6d} warps {k.metadata.num_warps}', flush=True)
