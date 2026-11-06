"""FP64 CPU: block_flex chunk-end dedupe vs the token reference (outputs, final state, all gradients), merge
rates, and the default path against the pre-change block_flex (bitwise)."""
import importlib.util, itertools, sys, torch
sys.path.insert(0, sys.argv[1])
from komp_slot.block_reference import block_komp_token_reference
from komp_slot.block_flex import block_komp_flex
spec = importlib.util.spec_from_file_location('komp_slot.flex_old', sys.argv[2])
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
torch.manual_seed(0)
worst = 0.
for (B, T, H, D, V, M, C), slot, evict, coh in itertools.product(
        ((1, 37, 2, 6, 5, 6, 8), (2, 29, 1, 5, 4, 4, 16), (1, 23, 2, 7, 3, 8, 4), (1, 40, 1, 4, 3, 4, 8)),
        (False, True), ('fifo', 'learned'), (None, .3, .6, .9)):
    dd = coh
    q, k = torch.randn(B, T, H, D, dtype=torch.float64), torch.randn(B, T, H, D, dtype=torch.float64)
    k[:, 1::3] = k[:, ::3][:, :k[:, 1::3].shape[1]]+.05*torch.randn_like(k[:, 1::3])   # near-repeats
    v = torch.randn(B, T, H, V, dtype=torch.float64); be = torch.rand(B, T, H, dtype=torch.float64)
    g = -torch.rand(B, T, H, M, dtype=torch.float64)*.4 if slot else -torch.rand(B, T, H, dtype=torch.float64)*.3
    K0 = torch.randn(B, H, M, D, dtype=torch.float64); S0 = torch.randn(B, H, M, V, dtype=torch.float64)
    sc = torch.randint(0, 3, (B, T, H)).double(); isc = torch.randint(0, 3, (B, H, M)).double()
    xa = [x.clone().requires_grad_() for x in (q, k, v, g, be, K0, S0)]
    xb = [x.clone().requires_grad_() for x in (q, k, v, g, be, K0, S0)]
    kw = dict(chunk_size=C, evict=evict, scores=sc, initial_scores=isc, dedupe=dd)
    stats = {}
    a = block_komp_token_reference(*xa[:5], (xa[5], xa[6]), **kw)
    b = block_komp_flex(*xb[:5], (xb[5], xb[6]), **kw, stats=stats)
    fo = float((a[0]-b[0]).abs().max())
    fs = float((a[1][0]-b[1][0]).abs().max())+float((a[1][1]-b[1][1]).abs().max())
    w, wk, wv = torch.randn_like(a[0]), torch.randn_like(a[1][0]), torch.randn_like(a[1][1])
    ga = torch.autograd.grad((a[0], a[1][0], a[1][1]), xa, (w, wk, wv), allow_unused=True)
    gb = torch.autograd.grad((b[0], b[1][0], b[1][1]), xb, (w, wk, wv), allow_unused=True)
    ge = max(float((x-y).abs().max()/(x.abs().max()+1e-30)) for x, y in zip(ga, gb) if x is not None)
    worst = max(worst, fo, fs, ge)
    rate = stats.get('evicted_dup', 0)/max(1, stats.get('candidates', 1))
    extra = ''
    if coh is None:
        xc = [x.clone() for x in (q, k, v, g, be, K0, S0)]
        c = old.block_komp_flex(*xc[:5], (xc[5], xc[6]), chunk_size=C, evict=evict, scores=sc, initial_scores=isc)
        extra = f'default bitwise-equal-old {torch.equal(c[0], b[0].detach()) and torch.equal(c[1][1], b[1][1].detach())}'
    print(f'{(B,T,H,D,V,M,C)} slot={slot:d} {evict:7s} dedupe={dd} dup-evicted {rate:.0%} out {fo:.1e} state {fs:.1e} grad {ge:.1e} {extra}', flush=True)
print('WORST', worst)
