"""Register/spill effect of the live-range edits (broadcast scalar Gam, Dp/Dl at use) and chunk_bwd warps."""
import sys, torch, triton
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms

data, slots = make(1, 8192, 64, 128, 128, 64)
for w in (4, 8):
    BF.WARPS['chunk_bwd'] = w
    for label, name, sc in (('scalar', 'block', None), ('slot', 'block_slot', None), ('learned', 'block_slot', data['scores'])):
        x = [y.detach().clone().requires_grad_() for y in data[name]]
        f = lambda: run(name, x, slots, 64, sc)
        with torch.no_grad():
            torch.manual_seed(1)
            ws = [torch.randn_like(o)*.01 for o in f()]
        fb = graph_ms(lambda: torch.autograd.grad(f(), x, ws), unroll=2, replays=5)
        torch.save([o.detach().cpu() for o in f()]+[g.detach().cpu() for g in torch.autograd.grad(f(), x, ws)],
                   f'/tmp/regs2-{sys.argv[1]}-{label}-{w}.pt')
        print(sys.argv[1], 'chunk_bwd warps', w, label, f'fwd+bwd {fb:.3f}', flush=True)
BF.WARPS['chunk_bwd'] = 4
for kname in ('chunk_bwd_kernel', 'keygrad_kernel', 'proj_bwd_a_kernel', 'proj_kernel', 'compose_kernel', 'ut_kernel'):
    kern = getattr(BF, kname)
    for dev, caches in kern.device_caches.items():
        for key, k in caches[0].items():
            print(f'{kname:20s} regs {k.n_regs:3d} spills {k.n_spills:4d} warps {k.metadata.num_warps}', flush=True)
