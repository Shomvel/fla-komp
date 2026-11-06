import sys, collections, torch
from torch.profiler import profile, ProfilerActivity
from benchmark_block import make, run
cfgs = {'main': (2, 2048, 4, 64, 64, 32), '8k': (1, 8192, 64, 128, 128, 64)}
for name in sys.argv[1:]:
    b, t, h, dk, dv, m = cfgs[name]
    data, slots = make(b, t, h, dk, dv, m)
    x = [y.detach().clone().requires_grad_() for y in data['block']]
    with torch.no_grad():
        ws = [torch.randn_like(o) for o in run('block', x, slots, 64)]
    step = lambda: torch.autograd.grad(run('block', x, slots, 64), x, ws)
    for _ in range(3): step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3): step()
        torch.cuda.synchronize()
    agg = collections.defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            agg[e.name] += e.device_time/3/1000
    print(name, 'total', round(sum(agg.values()), 4))
    for k, v in sorted(agg.items(), key=lambda x: -x[1])[:14]:
        print(f'  {v:9.4f} ms  {k[:70]}')
