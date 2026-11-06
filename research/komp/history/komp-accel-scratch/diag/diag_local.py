import torch, triton
import validate as base
from komp_bench.timing import graph_ms
from komp_slot.triton_chunk import local_bwd_kernel
t, m, c = 2048, 32, 16
xs, slots = base.core_inputs((2, t, 4, m, 64), 'cuda', 3)
h, p, r, v, a, b, init = xs
v = v.bfloat16(); nc = t//c
tape = torch.randn(2, 4, nc, m, 64, device='cuda'); ds = torch.randn_like(tape); do = torch.randn(v.shape, device='cuda')
work = torch.empty((2, 4, nc, c, m, 64), device='cuda'); dv = torch.empty_like(v)
for bv in (16, 32, 64):
    nv = 64//bv
    part = [torch.empty((nv, *x.shape), device='cuda') for x in (h, p, r, a, b)]
    for warps in (1, 2, 4, 8):
        run = lambda: local_bwd_kernel[(nc, 8, nv)](h, p, r, v, a, b, slots, tape, ds, do, work, *part[:3], dv, *part[3:], 2, t, 4, m, 64, c, nc, 32, bv, False, num_warps=warps)
        k = run()
        print(f'bv={bv} warps={warps} regs {k.n_regs} spills {k.n_spills} ms {graph_ms(run):.4f}', flush=True)
