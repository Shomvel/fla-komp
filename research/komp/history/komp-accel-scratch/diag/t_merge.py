"""8k: merged key-gradient path on/off, timing and keygrad device time."""
import collections, torch
from torch.profiler import profile, ProfilerActivity
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms

data, slots = make(1, 8192, 64, 128, 128, 64)
for name in ('block', 'block_slot'):
    for merge in (False, True):
        BF.MERGE_KEYGRAD = merge
        x = [y.detach().clone().requires_grad_() for y in data[name]]
        f = lambda: run(name, x, slots, 64)
        with torch.no_grad():
            ws = [torch.randn_like(o)*.01 for o in f()]
        step = lambda: torch.autograd.grad(f(), x, ws)
        fb = graph_ms(step, unroll=2, replays=5)
        step(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step(); torch.cuda.synchronize()
        agg = collections.defaultdict(float)
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                agg[e.name] += e.device_time/1000
        print(name, f'merge={merge}', f'fwd+bwd {fb:.3f}', ' '.join(f'{k[:12]}={v:.3f}' for k, v in agg.items()
                                                                  if k.startswith(('keygrad', 'pool_norm', 'proj_kernel', 'att_kernel', 'block_mapgrad'))), flush=True)
