import torch, triton
from komp_bench.timing import graph_ms
from komp_slot import geometry_chunk as G
from komp_slot.geometry import routes_parallel_into, normalize_kernel
b, t, h, d, m = 2, 2048, 4, 64, 32
torch.manual_seed(0)
q = torch.randn(b, t, h, d, device='cuda'); k = torch.randn_like(q); init = torch.randn(b, h, m, d, device='cuda')
slots = (torch.arange(t, device='cuda') % m)[None, :, None].expand(b, t, h).contiguous()
qn = torch.empty_like(q); pool = torch.empty(b, h, t+m, d, device='cuda')
normalize_kernel[(t+m, b*h)](q, k, init, qn, pool, t, h, m, d, 64)
src = torch.empty((b, t, h, m), device='cuda', dtype=torch.int32); rem = torch.empty((b, t, h), device='cuda', dtype=torch.int32)
last = torch.empty((b, h, m), device='cuda', dtype=torch.int32)
routes_parallel_into(slots, src, rem, last, m)
hist, proj, read = [torch.empty(b, t, h, m, device='cuda') for _ in range(3)]
dh, dp, dr = [torch.randn(b, t, h, m, device='cuda') for _ in range(3)]
print('--- split backward', flush=True)
z = torch.empty_like(proj)
for L in (16, 32):
    nc = triton.cdiv(t, L)
    p0 = torch.empty(b, h, nc, m, m, device='cuda')
    G.geometry_chunk_fwd_kernel[(nc, b*h)](qn, pool, src, rem, slots, hist, proj, read, p0, t, h, m, d, L, nc, 32, 64, .5, 1e-4, True, num_warps=4)
    for warps in (1, 2, 4):
        za = lambda: G.geometry_chunk_z_kernel[(nc, b*h)](pool, src, rem, slots, p0, dp, z, t, h, m, d, L, nc, 32, 64, .5, 1e-4, num_warps=warps)
        dqn = torch.empty_like(qn); dpool = torch.zeros_like(pool)
        gb = lambda: G.geometry_chunk_grad_kernel[(nc, b*h)](qn, pool, src, rem, slots, hist, proj, read, z, dh, dr, dqn, dpool, t, h, m, d, L, nc, 32, 64, .5, 1e-4, num_warps=warps)
        ka, kb = za(), gb()
        print(f'L={L} warps={warps} z regs {ka.n_regs} spills {ka.n_spills} ms {graph_ms(za):.4f} | grad regs {kb.n_regs} spills {kb.n_spills} ms {graph_ms(gb):.4f}', flush=True)
