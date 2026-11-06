import collections, itertools, torch
from torch.profiler import profile, ProfilerActivity
from komp_slot import block_fast as BF
from benchmark_block import make, run
from komp_bench.timing import graph_ms
for cfg, unroll in (((1, 8192, 64, 128, 128, 64), 2), ((2, 2048, 4, 64, 64, 32), 10)):
    data, slots = make(*cfg)
    for fs_, fp, fc, bf in ((False, False, False, True), (True, False, False, True), (True, True, False, True),
                            (True, False, True, True), (True, True, True, True), (True, True, True, False)):
        BF.FUSE_SCAN, BF.FUSE_PROJ_BWD, BF.FUSE_CHUNK_BWD, BF.BF16 = fs_, fp, fc, bf
        x = [y.detach().clone().requires_grad_() for y in data['block']]
        f = lambda: run('block', x, slots, 64)
        with torch.no_grad():
            fw = graph_ms(f, unroll=unroll, replays=5)
            ws = [torch.randn_like(o)*.01 for o in f()]
        step = lambda: torch.autograd.grad(f(), x, ws)
        fb = graph_ms(step, unroll=unroll, replays=5)
        step(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step(); torch.cuda.synchronize()
        n = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
        print(cfg[1], f'scan={fs_} projbwd={fp} chunkbwd={fc} bf16={bf}', 'kernels', n, f'fwd {fw:.3f} fwd+bwd {fb:.3f}', flush=True)
