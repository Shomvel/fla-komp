import sys, itertools, torch
sys.path.insert(0, sys.argv[1])
from komp_slot.block_reference import block_komp_token_reference
from komp_slot.block_flex import block_komp_flex
torch.manual_seed(0)
def state_fn(K, S, probe):
    kp = torch.nn.functional.normalize(probe, dim=-1)
    return torch.einsum('bhmv,bhpm->bhpv', S, torch.exp((torch.einsum('bhmd,pd->bhpm', K, kp)-1)/.5))
worst = 0
for (B, T, H, D, V, M, C), slot, evict, proj in itertools.product(((1, 37, 2, 6, 5, 6, 8), (2, 29, 1, 5, 4, 4, 16), (1, 23, 2, 7, 3, 8, 4)),
                                                              (False, True), ('fifo', 'residual', 'coef_norm', 'learned'), (False, True)):
    q, k = torch.randn(B, T, H, D, dtype=torch.float64), torch.randn(B, T, H, D, dtype=torch.float64)
    v = torch.randn(B, T, H, V, dtype=torch.float64); be = torch.rand(B, T, H, dtype=torch.float64)
    g = -torch.rand(B, T, H, M, dtype=torch.float64)*.4 if slot else -torch.rand(B, T, H, dtype=torch.float64)*.3
    K0 = torch.randn(B, H, M, D, dtype=torch.float64); S0 = torch.randn(B, H, M, V, dtype=torch.float64)
    sc = torch.randint(0, 3, (B, T, H)).double(); isc = torch.randint(0, 3, (B, H, M)).double()
    xa = [x.clone().requires_grad_() for x in (q, k, v, g, be, K0, S0)]
    xb = [x.clone().requires_grad_() for x in (q, k, v, g, be, K0, S0)]
    kw = dict(chunk_size=C, evict=evict, projectron=proj, scores=sc, initial_scores=isc)
    a = block_komp_token_reference(*xa[:5], (xa[5], xa[6]), **kw)
    b = block_komp_flex(*xb[:5], (xb[5], xb[6]), **kw)
    probe = torch.randn(3, D, dtype=torch.float64)
    fo = float((a[0]-b[0]).abs().max())
    fs = float((a[1][0]-b[1][0]).abs().max()) + float((a[1][1]-b[1][1]).abs().max())
    w = torch.randn_like(a[0]); wk = torch.randn_like(a[1][0]); wv = torch.randn_like(a[1][1])
    ga = torch.autograd.grad((a[0], a[1][0], a[1][1]), xa, (w, wk, wv), allow_unused=True)
    gb = torch.autograd.grad((b[0], b[1][0], b[1][1]), xb, (w, wk, wv), allow_unused=True)
    ge = max(float((x-y).abs().max()/(x.abs().max()+1e-30)) for x, y in zip(ga, gb) if x is not None)
    worst = max(worst, fo, fs, ge)
    print(f'{(B,T,H,D,V,M,C)} slot={slot} {evict:9s} proj={proj} out {fo:.1e} state {fs:.1e} grad {ge:.1e}', flush=True)
print('WORST', worst)
