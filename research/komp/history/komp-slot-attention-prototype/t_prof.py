import collections, sys, torch
from torch.profiler import profile, ProfilerActivity
from komp_slot import block_fast as BF
from benchmark_block import make, run
data, slots = make(1, 8192, 64, 128, 128, 64)
for fuse in (False, True):
    BF.FUSE, BF.BF16 = fuse, True
    x = [y.detach().clone().requires_grad_() for y in data['block']]
    f = lambda: run('block', x, slots, 64)
    with torch.no_grad():
        ws = [torch.randn_like(o)*.01 for o in f()]
    step = lambda: torch.autograd.grad(f(), x, ws)
    step(); step(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step(); torch.cuda.synchronize()
    agg = collections.defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            agg[e.name] += e.device_time/1000
    print('FUSE', fuse, round(sum(agg.values()), 3))
    for k, v in sorted(agg.items(), key=lambda z: -z[1])[:12]:
        print(f'  {v:7.3f}  {k}')
for kern in (BF.chunk_fwd_kernel, BF.bscan_compose_kernel, BF.bscan_expand_kernel, BF.proj_bwd_kernel, BF.chunk_bwd_kernel):
    for dev, caches in kern.device_caches.items():
        for key, k in caches[0].items():
            print(kern.fn.__name__, 'regs', k.n_regs, 'spills', k.n_spills, 'smem', k.metadata.shared, 'warps', k.metadata.num_warps)
