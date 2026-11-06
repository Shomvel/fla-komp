"""A/B of block_fast flags (rank-1 Gram sweep, pool+normalize backward fusion) plus the 8k kernel breakdown."""
import collections, torch
from torch.profiler import profile, ProfilerActivity
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms

VARIANTS = [dict(RANK1_SWEEP=False, FUSE_POOL_NORM=False, MERGE_KEYGRAD=False, ROW_NORMALIZE=False), dict(RANK1_SWEEP=True, FUSE_POOL_NORM=True, MERGE_KEYGRAD=False, ROW_NORMALIZE=True), dict(RANK1_SWEEP=True, FUSE_POOL_NORM=True, MERGE_KEYGRAD=True, ROW_NORMALIZE=True)]


def rms(a, b):
    return float(((a.double()-b.double()).square().mean()/(b.double().square().mean()+1e-30)).sqrt())


for cfg, unroll in (((1, 8192, 64, 128, 128, 64), 2), ((2, 2048, 4, 64, 64, 32), 10)):
    data, slots = make(*cfg)
    for label, name, sc in (('scalar', 'block', None), ('slot', 'block_slot', None), ('learned', 'block_slot', data['scores'])):
        base = None
        for i, flags in enumerate(VARIANTS):
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
            res = [o.detach() for o in f()]+[gr.detach() for gr in step()]
            base = base or res
            err = max(rms(a, b) for a, b in zip(res, base))
            print(cfg[1], label, flags, f'fwd {fw:.3f} fwd+bwd {fb:.3f} max-rms-vs-first {err:.1e}', flush=True)
            if i == len(VARIANTS)-1 and cfg[1] == 8192:
                step(); torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    step(); torch.cuda.synchronize()
                agg = collections.defaultdict(float)
                for e in prof.events():
                    if e.device_type == torch.autograd.DeviceType.CUDA:
                        agg[e.name] += e.device_time/1000
                print('  total', round(sum(agg.values()), 3), 'kernels', len([e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]), flush=True)
                for k, v in sorted(agg.items(), key=lambda z: -z[1])[:14]:
                    print(f'  {v:8.3f} ms  {k[:60]}', flush=True)
