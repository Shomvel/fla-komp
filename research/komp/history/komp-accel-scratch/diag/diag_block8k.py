import collections, sys, torch
from torch.profiler import profile, ProfilerActivity
from komp_slot import block_fast as BF
from benchmark_block import make
from komp_bench.timing import graph_ms
data, slots = make(1, 8192, 64, 128, 128, 64)
xs = data['block']
for mode in ('scalar', 'slot'):
    x = [y.detach().clone().requires_grad_() for y in xs]
    if mode == 'slot':
        x[3] = (x[3].detach()[..., None].expand(*x[3].shape, 64)+.01*torch.randn(*x[3].shape, 64, device='cuda').clamp(max=0)).contiguous().requires_grad_()
    run = lambda: BF.block_komp_fast(*x[:5], (x[5], x[6]), chunk_size=64)
    with torch.no_grad():
        ws = [torch.randn_like(o)*.01 for o in (run()[0], *run()[1])]
    step = lambda: torch.autograd.grad((lambda r: (r[0], *r[1]))(run()), x, ws)
    for _ in range(2): step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step(); torch.cuda.synchronize()
    agg = collections.defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            agg[e.name] += e.device_time/1000
    print(mode, 'total', round(sum(agg.values()), 3), flush=True)
    for k, v in sorted(agg.items(), key=lambda z: -z[1])[:16]:
        print(f'  {v:8.3f} ms  {k[:60]}', flush=True)
for kern in (BF.att_kernel, BF.proj_kernel, BF.ut_kernel, BF.compose_kernel, BF.proj_bwd_kernel, BF.ut_bwd_kernel, BF.keygrad_kernel):
    for dev, caches in kern.device_caches.items():
        for key, k in caches[0].items():
            print(kern.fn.__name__, 'regs', k.n_regs, 'spills', k.n_spills, 'smem', k.metadata.shared, flush=True)
