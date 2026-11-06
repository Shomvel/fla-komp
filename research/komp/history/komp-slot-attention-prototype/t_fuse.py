import collections, torch
from torch.profiler import profile, ProfilerActivity
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms

def count(name, xs, slots, c=64, **kw):
    x = [y.detach().clone().requires_grad_() for y in xs]
    f = lambda: run(name, x, slots, c)
    with torch.no_grad():
        ws = [torch.randn_like(o)*.01 for o in f()]
    step = lambda: torch.autograd.grad(f(), x, ws)
    step(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step(); torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    names = collections.Counter(e.name for e in ev)
    return len(ev), sum(e.device_time for e in ev)/1000, names

for cfg, unroll in (((2, 2048, 4, 64, 64, 32), 10), ((1, 8192, 64, 128, 128, 64), 2)):
    data, slots = make(*cfg)
    for name in ('gdn', 'gsa'):
        n, ms, names = count(name, data[name], slots)
        print(cfg[1], name, 'kernels', n, 'kernel-sum ms', round(ms, 3), flush=True)
    for fuse, bf in ((False, False), (True, False), (True, True)):
        BF.FUSE, BF.BF16 = fuse, bf
        for dname in ('block', 'block_slot'):
            n, ms, names = count(dname, data[dname], slots)
            x = [y.detach().clone().requires_grad_() for y in data[dname]]
            f = lambda: run(dname, x, slots, 64)
            with torch.no_grad():
                fw = graph_ms(f, unroll=unroll, replays=5)
                ws = [torch.randn_like(o)*.01 for o in f()]
            fb = graph_ms(lambda: torch.autograd.grad(f(), x, ws), unroll=unroll, replays=5)
            print(cfg[1], dname, f'fuse={fuse} bf16={bf}', 'kernels', n, f'fwd {fw:.3f} fwd+bwd {fb:.3f}', flush=True)
            if fuse and bf and dname == 'block':
                print('   ', dict(names), flush=True)
