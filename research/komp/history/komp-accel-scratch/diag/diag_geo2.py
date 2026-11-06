import itertools, torch, triton
from komp_bench.timing import graph_ms
from komp_slot import geometry_chunk as G
b, t, h, d, m = 2, 2048, 4, 64, 32
torch.manual_seed(0)
q = torch.randn(b, t, h, d, device='cuda', requires_grad=True); k = torch.randn(b, t, h, d, device='cuda', requires_grad=True)
init = torch.randn(b, h, m, d, device='cuda', requires_grad=True)
slots = (torch.arange(t, device='cuda') % m)[None, :, None].expand(b, t, h).contiguous()
with torch.no_grad():
    ws = [torch.randn_like(x) for x in G.triton_geometry_chunked(q, k, slots, init)]
for block in (16, 32):
    for fw, gw in itertools.product((1, 2, 4), (1, 2, 4)):
        G.WARPS.update(fwd=fw, z=gw, grad=gw)
        with torch.no_grad():
            f = graph_ms(lambda: G.triton_geometry_chunked(q, k, slots, init, block=block))
        fb = graph_ms(lambda: torch.autograd.grad(G.triton_geometry_chunked(q, k, slots, init, block=block), (q, k, init), ws))
        print(f'block={block} fwd_warps={fw} bwd_warps={gw} fwd {f:.4f} fwd+bwd {fb:.4f}', flush=True)
