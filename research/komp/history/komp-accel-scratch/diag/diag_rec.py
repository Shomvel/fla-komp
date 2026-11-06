import sys, torch, triton
import validate as base
from komp_bench.timing import graph_ms
from komp_slot import recurrence as R
t, m = 2048, 32
xs, slots = base.core_inputs((2, t, 4, m, 64), 'cuda', 3)
h, p, r, v, a, b, init = xs
v = v.bfloat16()
for c in (16, 32):
    nc = triton.cdiv(t, c); bm = max(16, triton.next_power_of_2(m)); n2 = 2*c
    tape = torch.randn(2, 4, nc, m, 64, device='cuda'); ds = torch.randn_like(tape); do = torch.randn(v.shape, device='cuda')
    outs = [torch.empty_like(x) for x in (h, p, r, a, b)]; dv = torch.empty_like(v)
    for warps in (4, 8):
        bv = R._bv(64, c, bm); nv = triton.cdiv(64, bv)
        nw = c*n2+c*c+c*bm+n2*bm+n2*n2+n2*c+n2*bm+2*n2+4*c+1
        uz = torch.empty((8, nc, 2, n2, 64), device='cuda'); work = torch.empty((nv, 8, nc, nw), device='cuda')
        grid = (nc, 8, nv)
        runs = {
          'uz': lambda: R.bwd_uz_kernel[grid](h, p, r, v, a, b, slots, tape, ds, do, uz, t, 4, m, 64, c, nc, bm, bv, num_warps=warps),
          'products': lambda: R.bwd_products_kernel[grid](h, p, r, v, a, b, slots, tape, ds, do, uz, dv, work, t, 4, m, 64, c, nc, bm, bv, nw, num_warps=warps),
          'coeff': lambda: R.bwd_coeff_kernel[(nc, 8)](h, p, r, a, b, slots, work, *outs, t, 4, m, c, nc, bm, nw, nv, num_warps=warps)}
        for name, run in runs.items():
            k = run()
            print('C', c, 'warps', warps, name, 'regs', k.n_regs, 'spills', k.n_spills, 'ms', round(graph_ms(run), 4), flush=True)
