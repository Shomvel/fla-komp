"""Per-kernel CUDA time of full KOMP (geometry + chunk) forward/backward at the FLA comparison shape."""
import sys, json, collections
import torch
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, '.')
from benchmark_fla import inputs, clone
from komp_slot.geometry_chunk import triton_geometry_chunked
from komp_slot.triton_chunk import triton_chunk
from komp_slot.fast_chunk import fast_chunk

def run(name, xs, slots):
    h, p, r, ks = triton_geometry_chunked(xs[0], xs[1], slots, xs[5], block=BLOCK)
    o, vs = fast_chunk(h, p, r, xs[2], xs[3], xs[4], slots, xs[6], local_bv=64)
    return o, ks, vs
BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 16

def main(t=2048, m=32):
    data, slots = inputs(t, torch.bfloat16, m=m)
    xs = clone(data['komp'])
    out = run('komp', xs, slots)
    ws = [torch.randn_like(o) * .01 for o in out]
    def step():
        o = run('komp', xs, slots)
        torch.autograd.grad(o, xs, ws)
    for _ in range(5): step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10): step()
        torch.cuda.synchronize()
    agg = collections.defaultdict(lambda: [0.0, 0])
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            agg[e.name][0] += e.device_time / 10 / 1000; agg[e.name][1] += 1
    total = sum(v[0] for v in agg.values())
    print(f'T={t} M={m} total kernel ms/iter {total:.4f}')
    for name, (ms, n) in sorted(agg.items(), key=lambda x: -x[1][0]):
        print(f'{ms:9.4f} ms  x{n//10:3d}  {name[:90]}')

if __name__ == '__main__':
    for t, m in ((2048, 32), (1024, 16)):
        main(t, m)
