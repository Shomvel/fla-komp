"""Round-5 probes: standalone sweep kernels, then op-level flag A/B with bitwise agreement checks."""
import itertools, torch, triton, triton.language as tl
from komp_slot import block_fast as BF
from komp_slot.geometry_chunk import _sweep_inverse
from benchmark_block import make, run
from komp_bench.timing import graph_ms


@triton.jit
def sweep_kernel(G, P, M: tl.constexpr, FUSED: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, M)
    g = tl.load(G+i*M*M+r[:, None]*M+r[None, :])
    if FUSED:
        x = BF._sweep_inverse_fused(g, M, M)
    else:
        x = _sweep_inverse(g, M, M)
    tl.store(P+i*M*M+r[:, None]*M+r[None, :], x)


torch.manual_seed(0)
k = torch.nn.functional.normalize(torch.randn(8192, 64, 128, device='cuda'), dim=-1)
G = torch.exp((k@k.transpose(1, 2)-1)/.5)+1e-4*torch.eye(64, device='cuda')
ref = None
for fused, w in itertools.product((False, True), (1, 2, 4, 8)):
    P = torch.empty_like(G)
    f = lambda: sweep_kernel[(8192,)](G, P, 64, fused, num_warps=w)
    ms = triton.testing.do_bench(f, warmup=5, rep=50)
    ref = P.clone() if ref is None else ref
    print(f'sweep fused={fused} warps={w} {ms:.3f} ms bitwise-equal-to-first {torch.equal(P, ref)}', flush=True)

FLAGS = ('SWEEP_FUSED', 'QN_BF16', 'SKIP_ZERO', 'SLOT_GC_CACHE')
VARIANTS = [dict.fromkeys(FLAGS, False)]+[{**dict.fromkeys(FLAGS, False), f: True} for f in FLAGS]+[dict.fromkeys(FLAGS, True)]
for cfg, unroll in (((1, 8192, 64, 128, 128, 64), 2), ((2, 2048, 4, 64, 64, 32), 10), ((64, 192, 2, 64, 64, 32), 10)):
    data, slots = make(*cfg)
    for label, name, sc in (('scalar', 'block', None), ('slot', 'block_slot', None), ('learned', 'block_slot', data['scores'])):
        base = None
        for flags in VARIANTS:
            for key, val in flags.items():
                setattr(BF, key, val)
            x = [y.detach().clone().requires_grad_() for y in data[name]]
            f = lambda: run(name, x, slots, 64, sc)
            with torch.no_grad():
                fw = graph_ms(f, unroll=unroll, replays=5)
                torch.manual_seed(1)
                ws = [torch.randn_like(o)*.01 for o in f()]
            step = lambda: torch.autograd.grad(f(), x, ws)
            fb = graph_ms(step, unroll=unroll, replays=5)
            res = [o.detach() for o in f()]+[g.detach() for g in step()]
            base = base or res
            same = all(torch.equal(a, b) for a, b in zip(res, base))
            on = [k for k, v in flags.items() if v] or ['none']
            print(cfg[:2], label, '+'.join(on), f'fwd {fw:.3f} fwd+bwd {fb:.3f} bitwise {same}', flush=True)
