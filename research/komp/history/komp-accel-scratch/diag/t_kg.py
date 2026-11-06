"""block_fast flag combinations against the all-off path (fifo and learned), several shapes."""
import itertools, torch
from komp_slot import block_fast as BF
from validate_block_komp import inputs

FLAGS = ('MERGE_KEYGRAD', 'FUSE_POOL_NORM', 'ROW_NORMALIZE')


def rms(a, b):
    return float(((a.double()-b.double()).square().mean()/(b.double().square().mean()+1e-30)).sqrt())


def grads(xs, c, learned):
    x = [y.detach().clone().requires_grad_() for y in xs]
    kw = {}
    if learned:
        torch.manual_seed(11)
        b, t, h = xs[0].shape[:3]
        kw = dict(evict='learned', scores=torch.randn(b, t, h, device='cuda'),
                  initial_scores=torch.randn(b, h, xs[5].shape[-2], device='cuda'))
    o, (fk, fv) = BF.block_komp_fast(*x[:5], (x[5], x[6]), chunk_size=c, **kw)
    torch.manual_seed(3)
    ws = [torch.randn_like(z) for z in (o, fk, fv)]
    return [o.detach(), fk.detach(), fv.detach(), *torch.autograd.grad((o, fk, fv), x, ws)]


names = ('out', 'keys', 'values', 'dq', 'dk', 'dv', 'dg', 'dbeta', 'dk0', 'dv0')
bad = 0
for shape in [(1, 200, 2, 32, 32, 32, 64), (1, 150, 2, 64, 64, 64, 64), (2, 77, 2, 16, 24, 8, 16),
              (1, 90, 2, 32, 32, 48, 32), (1, 512, 4, 128, 128, 64, 64), (1, 100, 1, 32, 32, 16, 32),
              (1, 40, 2, 32, 32, 64, 64)]:
    *dims, c = shape
    for learned, slot, dtype in itertools.product((False, True), (False, True), (torch.float32, torch.bfloat16)):
        xs = inputs(*dims, dtype, 5, slot=slot)
        for f in FLAGS:
            setattr(BF, f, False)
        base = grads(xs, c, learned)
        for f in FLAGS:
            setattr(BF, f, True)
        got = grads(xs, c, learned)
        errs = {n: rms(a, b) for n, a, b in zip(names, got, base)}
        e = max(errs.values())
        ok = e < (2e-5 if dtype == torch.float32 else 5e-3)
        bad += not ok
        print('OK' if ok else 'DIFF', shape, 'learned' if learned else 'fifo', 'slot' if slot else 'scalar', dtype,
              f'{e:.1e}', max(errs, key=errs.get), flush=True)
print('DIFFS', bad)
