"""Cost of a [64,64] axis-0 scan (as in the per-slot dg suffix sum) vs triangular-ones products."""
import torch, triton, triton.language as tl


@triton.jit
def scan_kernel(X, Y, MODE: tl.constexpr, C: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, C)
    x = tl.load(X+i*C*C+r[:, None]*C+r[None, :])
    if MODE == 0:
        y = tl.sum(x, 0)[None, :]-tl.cumsum(x, 0)+x
    elif MODE == 1:
        up = (r[:, None] <= r[None, :]).to(tl.float32)
        y = tl.dot(tl.trans(up), x, input_precision='tf32x3')
    elif MODE == 2:
        up = (r[:, None] <= r[None, :]).to(tl.float32)
        y = tl.dot(tl.trans(up), x, input_precision='ieee')
    else:
        y = x*2.
    tl.store(Y+i*C*C+r[:, None]*C+r[None, :], y)


x = torch.randn(8192, 64, 64, device='cuda')
ref = x.flip(1).cumsum(1).flip(1)
for mode, name in ((3, 'copy'), (0, 'tl.cumsum suffix'), (1, 'dot tf32x3'), (2, 'dot ieee')):
    for w in (2, 4, 8):
        y = torch.empty_like(x)
        ms = triton.testing.do_bench(lambda: scan_kernel[(8192,)](x, y, mode, 64, num_warps=w), warmup=5, rep=50)
        err = float((y-ref).abs().max()) if mode != 3 else 0.
        print(f'{name:18s} warps={w} {ms:.3f} ms max err {err:.1e}', flush=True)
