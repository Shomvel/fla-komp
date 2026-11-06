"""KEYGRAD_DEST on/off: 8k timing, keygrad + pool_norm device time, bitwise agreement."""
import collections, torch
from torch.profiler import profile, ProfilerActivity
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms

for cfg in ((1, 8192, 64, 128, 128, 64), (2, 4096, 8, 64, 64, 64)):
    data, slots = make(*cfg)
    for name in ('block', 'block_slot'):
        base = None
        for dest in (False, True):
            BF.KEYGRAD_DEST = dest
            x = [y.detach().clone().requires_grad_() for y in data[name]]
            f = lambda: run(name, x, slots, 64)
            with torch.no_grad():
                torch.manual_seed(1)
                ws = [torch.randn_like(o)*.01 for o in f()]
            step = lambda: torch.autograd.grad(f(), x, ws)
            fb = graph_ms(step, unroll=2, replays=5)
            res = [o.detach() for o in f()]+[g.detach() for g in step()]
            base = base or res
            same = all(torch.equal(a, b) for a, b in zip(res, base))
            step(); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                step(); torch.cuda.synchronize()
            agg = collections.defaultdict(float)
            for e in prof.events():
                if e.device_type == torch.autograd.DeviceType.CUDA:
                    agg[e.name] += e.device_time/1000
            print(cfg[:3], name, f'dest={dest} fwd+bwd {fb:.3f} bitwise {same}',
                  ' '.join(f'{k[:14]}={v:.3f}' for k, v in agg.items() if 'keygrad' in k or 'pool_norm' in k), flush=True)
