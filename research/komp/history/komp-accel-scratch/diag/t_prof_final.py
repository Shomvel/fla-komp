"""Per-kernel device time of one 8k forward+backward with the default block_fast settings."""
import collections, torch
from torch.profiler import profile, ProfilerActivity
from benchmark_block import make, run

data, slots = make(1, 8192, 64, 128, 128, 64)
for label, name, sc in (('scalar', 'block', None), ('slot', 'block_slot', None), ('learned', 'block_slot', data['scores'])):
    x = [y.detach().clone().requires_grad_() for y in data[name]]
    f = lambda: run(name, x, slots, 64, sc)
    with torch.no_grad():
        ws = [torch.randn_like(o)*.01 for o in f()]
    step = lambda: torch.autograd.grad(f(), x, ws)
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    agg = collections.defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            agg[e.name] += e.device_time/3/1000
    print(label, 'total', f'{sum(agg.values()):.3f}', flush=True)
    for k, v in sorted(agg.items(), key=lambda z: -z[1]):
        if v >= .05:
            print(f'  {v:7.3f} ms  {k[:50]}', flush=True)
