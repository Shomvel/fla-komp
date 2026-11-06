import itertools, torch
from komp_bench.timing import graph_ms
from komp_slot import geometry_chunk as G
b, t, h, d, m = 2, 2048, 4, 64, 32
torch.manual_seed(0)
q = torch.randn(b, t, h, d, device='cuda', requires_grad=True); k = torch.randn(b, t, h, d, device='cuda', requires_grad=True)
init = torch.randn(b, h, m, d, device='cuda', requires_grad=True)
slots = (torch.arange(t, device='cuda') % m)[None, :, None].expand(b, t, h).contiguous()
with torch.no_grad():
    ws = [torch.randn_like(x) for x in G.triton_geometry_chunked(q, k, slots, init)]
for sw, fw in itertools.product((1, 2, 4, 8), (1, 2, 4, 8)):
    G.WARPS.update(start=sw, finish=fw)
    with torch.no_grad():
        f = graph_ms(lambda: G.triton_geometry_chunked(q, k, slots, init))
    fb = graph_ms(lambda: torch.autograd.grad(G.triton_geometry_chunked(q, k, slots, init), (q, k, init), ws))
    print(f'start={sw} finish={fw} fwd {f:.4f} fwd+bwd {fb:.4f}', flush=True)
