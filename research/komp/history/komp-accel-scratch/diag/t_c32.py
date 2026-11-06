"""8k C=32 scalar regression hunt: forward per-kernel device time under flag toggles (run per code version)."""
import collections, sys, torch
from torch.profiler import profile, ProfilerActivity
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms

data, slots = make(1, 8192, 64, 128, 128, 64)
variants = [{}]
for flag in ('QN_BF16', 'DOV_BF16', 'SLOT_GC_CACHE', 'SKIP_ZERO'):
    if hasattr(BF, flag):
        variants.append({flag: False})
for v in variants:
    saved = {k: getattr(BF, k) for k in v}
    for k, val in v.items():
        setattr(BF, k, val)
    x = [y.detach().clone().requires_grad_() for y in data['block']]
    f = lambda: run('block', x, slots, 32)
    with torch.no_grad():
        fw = graph_ms(f, unroll=2, replays=3)
        f(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            f(); torch.cuda.synchronize()
    agg = collections.defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            agg[e.name] += e.device_time/1000
    print(sys.argv[1], v or 'defaults', f'fwd {fw:.3f}', ' '.join(f'{k[:16]}={t:.2f}' for k, t in sorted(agg.items(), key=lambda z: -z[1])[:6]), flush=True)
    for k, val in saved.items():
        setattr(BF, k, val)
